"""Forecast from saved models without training or reading future target labels.

Custom aggregate input CSV: timestamp, energy_kwh (hourly complete kWh).
For iTransformer also supply a wide module-history CSV with timestamp and the
13 module columns recorded in final_model.pt. Outputs include project factors.
"""
from pathlib import Path
from types import SimpleNamespace
import sys, json, argparse
import numpy as np
import pandas as pd
from common import OUT, load_training, future_index, baseline_predict, prediction_frame

def read_history(path):
    if path is None:return load_training()
    df=pd.read_csv(path)
    missing=set(['timestamp','energy_kwh'])-set(df)
    if missing:raise ValueError(f'Input CSV missing {sorted(missing)}')
    s=pd.Series(df.energy_kwh.to_numpy(dtype=float),index=pd.to_datetime(df.timestamp),name='energy_kwh').sort_index()
    if len(s)<960 or not s.index.is_unique or not s.index.equals(pd.date_range(s.index[0],s.index[-1],freq='h')) or not np.isfinite(s).all():
        raise ValueError('Need at least 960 complete, unique, finite consecutive hourly targets')
    return s

def run(model,history,module_history=None):
    import torch
    torch.set_num_threads(4)
    future=future_index(history.index[-1]+pd.Timedelta(hours=1)); quantiles={}
    if model=='chronos2':
        from chronos2_experiment import pipeline,predict
        meta=json.loads((OUT/'chronos2/provenance.json').read_text(encoding='utf-8'))
        pipe=pipeline(Path(meta['local_snapshot']));output,_=predict(pipe,history)
        pred=output.predictions.to_numpy(dtype=float)
        quantiles={f'q{q.replace(".","")}':output[q].to_numpy(dtype=float) for q in ['0.1','0.5','0.9']}
    elif model=='arima':
        from statsmodels.tsa.arima.model import ARIMAResults
        fit=ARIMAResults.load(OUT/'saved_models/final_arima.pkl')
        # Apply saved parameters to this observation history, without refitting.
        fit=fit.apply(history.to_numpy(dtype=float),refit=False)
        pred=np.asarray(fit.forecast(745),dtype=float)
    elif model=='xgboost':
        from xgboost import XGBRegressor
        from arima_xgb import direct_future_features
        meta=json.loads((OUT/'saved_models/final_xgboost_metadata.json').read_text(encoding='utf-8'))
        fitted=XGBRegressor();fitted.load_model(OUT/'saved_models/final_xgboost.json')
        fitted.set_params(n_jobs=4)
        pred=fitted.predict(direct_future_features(history,future)).astype(float)+meta['target_center']
    elif model=='patchtst':
        sys.path.insert(0,str(OUT/'patchtst'))
        from patchtst_vendor.models.PatchTST import Model
        saved=torch.load(OUT/'patchtst/checkpoints/final_3600.pt',map_location='cpu',weights_only=True)
        fitted=Model(SimpleNamespace(**saved['config']));fitted.load_state_dict(saved['state_dict']);fitted.eval()
        context=saved['config']['seq_len'];mean=saved['normalization_mean'];sd=saved['normalization_std']
        xb=torch.tensor((history.iloc[-context:].to_numpy()-mean)/sd,dtype=torch.float32).reshape(1,context,1)
        with torch.no_grad():pred=fitted(xb)[0,:,0].numpy().astype(float)*sd+mean
    elif model=='itransformer':
        from itransformer_experiment import load,infer,module_data
        fitted,saved=load(OUT/'itransformer/final_model.pt')
        if module_history is None:
            if not history.index.equals(load_training().index):raise ValueError('Custom iTransformer aggregate history requires --module-history')
            wide=module_data(history)
        else:
            df=pd.read_csv(module_history);df['timestamp']=pd.to_datetime(df.timestamp)
            wide=df.set_index('timestamp').reindex(history.index)[saved['columns']]
            if wide.isna().any().any():raise ValueError('Module history contains missing times/values')
            np.testing.assert_allclose(wide.sum(axis=1),history,rtol=0,atol=1e-6)
        pred=infer(fitted,wide,saved['config'],np.array(saved['mean']),np.array(saved['scale'])).sum(axis=1)
    elif model in ['baseline_mean','baseline_median','baseline_weekday_hour_mean']:
        pred=baseline_predict(model.replace('baseline_',''),history,future)
    else:raise ValueError(f'Unknown model {model}')
    if len(pred)!=745 or not np.isfinite(pred).all():raise ValueError('Invalid saved-model output')
    return future,np.asarray(pred,dtype=float),quantiles

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='selected',choices=['selected','arima','xgboost','patchtst','itransformer','chronos2','baseline_mean','baseline_median','baseline_weekday_hour_mean'])
    p.add_argument('--history',type=Path);p.add_argument('--module-history',type=Path)
    p.add_argument('--steps',type=int,default=745)
    p.add_argument('--output',type=Path,default=OUT/'inference_latest.csv')
    args=p.parse_args()
    if not 1<=args.steps<=745:raise ValueError('Saved direct models support 1..745 hourly outputs')
    model=args.model
    if model=='selected':model=json.loads((OUT/'deployment_selection_lock.json').read_text(encoding='utf-8'))['selected_model']
    history=read_history(args.history);future,pred,quantiles=run(model,history,args.module_history)
    rows=prediction_frame(model,'inference',future[0],future[:args.steps],np.full(args.steps,np.nan),pred[:args.steps])
    rows['cost_krw_project_rate']=pred[:args.steps]*180
    rows['carbon_kgco2_project_factor']=pred[:args.steps]*.424
    for key,value in quantiles.items():rows[key]=value[:args.steps]
    args.output.parent.mkdir(parents=True,exist_ok=True);rows.to_csv(args.output,index=False,encoding='utf-8-sig')
    print(f'{model}: {len(rows)} hourly predictions saved to {args.output}')

if __name__=='__main__':main()
