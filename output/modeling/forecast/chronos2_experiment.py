"""Official Chronos-2 frozen zero-shot forecast, local CPU inference only."""
from pathlib import Path
import os, time, argparse, json, hashlib
from common import (ROOT, OUT, load_training, future_index, history_before, actual_at,
                    score_prefixes, prediction_frame, write_json,VALIDATION_ORIGIN,TEST_ORIGIN)
import numpy as np
import pandas as pd

DEST=OUT/'chronos2'
CACHE=ROOT/'output/modeling/.model_cache'
os.environ['HF_HOME']=str(CACHE)
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['DO_NOT_TRACK']='1'
REVISION='29ec3766d36d6f73f0696f85560a422f50e8498c'

def prepare():
    from huggingface_hub import snapshot_download
    path=snapshot_download('amazon/chronos-2',revision=REVISION,
            allow_patterns=['config.json','model.safetensors','README.md'],
            cache_dir=str(CACHE),token=False)
    return Path(path)

def pipeline(path):
    import torch
    torch.set_num_threads(4)
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained(str(path),device_map='cpu',local_files_only=True)

def predict(pipe,history):
    data=pd.DataFrame({'id':'aggregate','timestamp':history.index,'target':history.values})
    started=time.perf_counter()
    output=pipe.predict_df(data,id_column='id',timestamp_column='timestamp',target='target',
            prediction_length=745,quantile_levels=[0.1,0.5,0.9],batch_size=1,
            context_length=8192,freq='h')
    expected=future_index(history.index[-1]+pd.Timedelta(hours=1))
    output=output.sort_values('timestamp')
    if not pd.DatetimeIndex(output['timestamp']).equals(expected): raise ValueError('Chronos timestamps mismatch')
    if 'predictions' not in output: raise ValueError(f'Unexpected output {list(output.columns)}')
    # Official library names this point output "mean", but returns its median
    # (training quantile 0.5). This forecast therefore optimizes a median point.
    return output,time.perf_counter()-started

def main(final=False,download_only=False):
    DEST.mkdir(parents=True,exist_ok=True)
    path=prepare(); print('Official Chronos-2 weights cached at immutable revision',REVISION,flush=True)
    provenance={'model_id':'amazon/chronos-2','revision':REVISION,'local_snapshot':str(path),
       'license':'Apache-2.0','mode':'Frozen zero-shot univariate, no finetuning',
       'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()},
       'data_transfer':'Only public model files downloaded. Input observations processed locally.'}
    write_json(DEST/'provenance.json',provenance)
    if download_only:return
    series=load_training(); pipe=pipeline(path)
    if final:
        if not (DEST/'selection_lock.json').exists():raise ValueError('Internal protocol lock missing')
        output,elapsed=predict(pipe,series); origin=series.index[-1]+pd.Timedelta(hours=1)
        frame=prediction_frame('Chronos-2','may_full_path',origin,pd.DatetimeIndex(output.timestamp),
                               np.full(745,np.nan),output.predictions.values)
        for col in ['0.1','0.5','0.9']:
            if col in output:frame['q'+col.replace('.','')]=output[col].to_numpy(dtype=float)
        frame.to_csv(DEST/'may_full_path.csv',index=False)
        write_json(DEST/'final_fit.json',{'inference_seconds':elapsed,'context_hours':len(series),'trainable':False})
        print('Chronos-2 final forecast saved before May scoring.',flush=True);return
    write_json(DEST/'selection_lock.json',{'model':'Chronos-2','model_id':'amazon/chronos-2','revision':REVISION,
           'context_length':8192,'prediction_length':745,'quantile_levels':[0.1,0.5,0.9],
           'selection_basis':'Predeclared official frozen model; no tuning','mode':'zero-shot','status':'frozen_before_May_labels'})
    frames=[]; result={'model':'Chronos-2','splits':{}}
    for name,origin in [('validation',VALIDATION_ORIGIN),('internal_test',TEST_ORIGIN)]:
        hist=history_before(series,origin); output,elapsed=predict(pipe,hist)
        actual=actual_at(series,pd.DatetimeIndex(output.timestamp)); pred=output.predictions.values
        metrics=score_prefixes(actual,pred)
        interval={}
        if '0.1' in output and '0.9' in output:
            lo=output['0.1'].values;hi=output['0.9'].values
            interval={'nominal_coverage':0.8,'empirical_coverage':float(((actual>=lo)&(actual<=hi)).mean()),'mean_width_kwh':float((hi-lo).mean()),'calibrated_on_this_project':False}
        result['splits'][name]={'metrics':metrics,'inference_seconds':elapsed,'context_hours':len(hist),'interval':interval}
        frame=prediction_frame('Chronos-2',name,origin,pd.DatetimeIndex(output.timestamp),actual,pred)
        for col in ['0.1','0.5','0.9']:
            if col in output:frame['q'+col.replace('.','')]=output[col].to_numpy(dtype=float)
        frames.append(frame)
        print('Chronos-2',name,'P745MAE',metrics['P745']['mae_kwh'],'seconds',elapsed,flush=True)
    pd.concat(frames).to_csv(DEST/'internal_predictions.csv',index=False)
    write_json(DEST/'results.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');p.add_argument('--download-only',action='store_true')
    args=p.parse_args();main(args.final,args.download_only)
