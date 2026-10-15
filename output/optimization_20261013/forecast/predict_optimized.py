"""Read new optimized saved models and reproduce the current fixed future path.

This CLI is separate from the previous modeling CLI and never opens May labels.
An optional history CSV must match this experiment's known historical endpoint;
general operational updates/new future measurements are outside this interface.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import argparse
import json
import numpy as np
import pandas as pd
from opt_common import OUT,load_target,future,prediction_frame,write_json
from classical_opt import build_future,calendar

def model_directory(scenario,name):return OUT/scenario/('baselines' if name=='baseline' else name)

def checkpoint_path(directory,info):
    if isinstance(info,str) and Path(info).exists():return Path(info)
    if isinstance(info,dict):
        for key in ['path','model_path','checkpoint_path']:
            if key in info and Path(info[key]).exists():return Path(info[key])
    choices=list(directory.glob('*.pt'))+list((directory/'saved_model').glob('*.pt'))+list(directory.glob('*.pth'))
    if len(choices)!=1:raise ValueError('Could not uniquely locate the final neural checkpoint')
    return choices[0]

def forecast_saved(scenario,name):
    directory=model_directory(scenario,name)
    result=json.loads((directory/'results.json').read_text(encoding='utf-8'))
    y=load_target();points=24 if scenario=='dayahead24' else 745
    idx=future(y.index[-1]+pd.Timedelta(hours=1),points)
    if name in ['arima','xgboost','baseline']:
        metadata=json.loads((directory/'saved_model/model_metadata.json').read_text(encoding='utf-8'))
        if pd.Timestamp(metadata['history_end'])!=y.index[-1]:raise ValueError('Saved model endpoint changed')
        cfg=metadata['config']
        if name=='arima':
            from statsmodels.tsa.arima.model import ARIMAResults
            fitted=ARIMAResults.load(metadata['model_path']);p=np.asarray(fitted.forecast(steps=points),dtype=np.float64)
        elif name=='xgboost':
            from xgboost import XGBRegressor
            fitted=XGBRegressor();fitted.load_model(metadata['model_path']);fitted.set_params(n_jobs=4)
            X=build_future(y,idx,cfg['feature_set'])
            p=metadata['target_center']+cfg['residual_shrinkage']*np.asarray(fitted.predict(X),dtype=np.float64)
        elif cfg['name']=='calendar_ridge':
            p=metadata['history_mean']+calendar(idx)@np.asarray(metadata['ridge_coef'])+metadata['ridge_intercept']
        else:
            if cfg['name']=='train_mean':level=metadata['history_mean']
            elif cfg['name']=='train_median':level=metadata['history_median']
            elif cfg['name']=='expanding_mean':level=float(y.mean())
            elif cfg['name']=='recent_mean':level=float(y.iloc[-cfg['window']:].mean())
            elif cfg['name']=='ema':level=metadata.get('ema_last',float(y.ewm(span=cfg['window'],adjust=False).mean().iloc[-1]))
            else:raise ValueError('Unknown saved baseline')
            p=np.repeat(level,points).astype(np.float64)
    elif name=='chronos2':
        from chronos_opt import pipeline,raw_predict,calibrate
        metadata=json.loads((directory/'saved_model/model_metadata.json').read_text(encoding='utf-8'))
        cfg=metadata['selected_config'];pipe,_=pipeline();raw=raw_predict(pipe,y,points,cfg['context']);p=calibrate(raw,y,cfg).astype(np.float64)
    else:
        from neural_hpo import predict_saved_checkpoint
        saved=checkpoint_path(directory,result.get('saved_model'))
        observed_idx,p,info=predict_saved_checkpoint(saved)
        if not pd.DatetimeIndex(observed_idx).equals(idx):raise ValueError('Saved neural forecast index mismatch')
        p=np.asarray(p,dtype=np.float64)
    reference=pd.read_csv(directory/'future_path.csv')['predicted_kwh'].to_numpy(dtype=np.float64)
    if p.shape!=reference.shape or not np.allclose(p,reference,rtol=0,atol=1e-8):
        raise ValueError(f'Saved-model inference differs from frozen future path: {np.max(np.abs(p-reference))}')
    return idx,p,{'model':name,'scenario':scenario,'path_rows':len(p),'reload_max_abs_diff_kwh':float(np.max(np.abs(p-reference))),
                  'forecast_origin':str(idx[0]),'forecast_end':str(idx[-1]),'May_labels_read':False,'new_optimized_models_only':True}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scenario',required=True,choices=['dayahead24','long745'])
    parser.add_argument('--model',default='selected',choices=['selected','arima','xgboost','patchtst','itransformer','chronos2','baseline'])
    parser.add_argument('--history',default=None);parser.add_argument('--output',default=None)
    args=parser.parse_args();name=args.model
    if name=='selected':
        lock=json.loads((OUT/'comparison_selection_lock.json').read_text(encoding='utf-8'));name=lock['scenarios'][args.scenario]['selected_model']
    if args.history:
        raw=pd.read_csv(args.history);candidate=pd.Series(raw.iloc[:,1].to_numpy(dtype=float),index=pd.to_datetime(raw.iloc[:,0])).sort_index();expected=load_target()
        if not candidate.index.equals(expected.index) or not np.allclose(candidate.values,expected.values,rtol=0,atol=1e-8):
            raise ValueError('Optional history must match this experiment; new operational context is not supported')
    idx,p,verification=forecast_saved(args.scenario,name)
    output=Path(args.output) if args.output else OUT/'cli_predictions'/f'{args.scenario}_{name}.csv'
    output.parent.mkdir(parents=True,exist_ok=True)
    prediction_frame(name,args.scenario,'saved_inference',0,idx[0],idx,np.repeat(np.nan,len(idx)),p).to_csv(output,index=False,encoding='utf-8-sig')
    verification['output']=str(output);write_json(output.with_suffix('.verification.json'),verification)
    print(json.dumps(verification,indent=2))

if __name__=='__main__':main()
