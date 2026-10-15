"""Infer saved 7-detector current-observation scores on a new RTU CSV.

Input: original module(equipment), localtime and the 14 sensor columns.
Output: original key, one row per observation, float64 scores/alarms per model.
Scores do not represent verified real-fault probabilities. No training occurs.
"""
from pathlib import Path
import argparse,json,pickle
import numpy as np
import pandas as pd
import torch
from torch import nn
from xgboost import XGBClassifier

OUT=Path(__file__).resolve().parent
KEY='module(equipment)'

class AE(nn.Module):
    def __init__(self,p):
        super().__init__();self.network=nn.Sequential(nn.Linear(p,32),nn.ReLU(),nn.Linear(32,12),nn.ReLU(),nn.Linear(12,4),nn.ReLU(),nn.Linear(4,12),nn.ReLU(),nn.Linear(12,32),nn.ReLU(),nn.Linear(32,p))
    def forward(self,x):return self.network(x)

def load():
    torch.set_num_threads(4)
    locks=json.loads((OUT/'threshold_locks.json').read_text(encoding='utf-8'));saved={}
    for name,meta in locks.items():
        d=dict(meta)
        if d['kind']=='xgb':
            model=XGBClassifier(n_jobs=4);model.load_model(OUT/'models'/f'{name}.json');model.set_params(n_jobs=4);d['model']=model
            assert model.get_booster().feature_names==d['features']
        elif d['kind']=='ae':
            ck=torch.load(OUT/'models'/f'{name}.pt',map_location='cpu',weights_only=True)
            model=AE(len(d['features']));model.load_state_dict(ck['state_dict']);model.eval();d['model']=model
            with (OUT/'models'/f'{name}_scaler.pkl').open('rb') as f:d['scaler']=pickle.load(f)
            assert ck['features']==d['features'] and ck['threshold']==d['threshold']
        elif d['kind']=='if':
            with (OUT/'models'/f'{name}.pkl').open('rb') as f:d['model']=pickle.load(f)
        else:
            z=json.loads((OUT/'models'/'robust_z.json').read_text(encoding='utf-8'))
            assert z['features']==d['features'];d['median']=np.asarray(z['median']);d['mad_scale']=np.asarray(z['mad_scale'])
        saved[name]=d
    return saved

def infer(frame,saved):
    required=list(dict.fromkeys([KEY,'localtime',*[f for d in saved.values() for f in d['features']]]))
    missing=[f for f in required if f not in frame]
    if missing:raise ValueError('Missing columns: '+', '.join(missing))
    if frame[[KEY,'localtime']].isna().any().any():raise ValueError('Missing source keys')
    features=[f for f in required if f not in [KEY,'localtime']]
    values=frame[features].apply(pd.to_numeric,errors='coerce')
    if not np.isfinite(values.to_numpy()).all():raise ValueError('Sensors contain missing, nonnumeric, or nonfinite values')
    result=frame[[KEY,'localtime']].copy()
    for name,d in saved.items():
        v=values[d['features']]
        if d['kind']=='xgb':scores=d['model'].predict_proba(v)[:,1].astype(np.float64)
        elif d['kind']=='if':scores=-d['model'].score_samples(v).astype(np.float64)
        elif d['kind']=='z':scores=np.abs((v.to_numpy()-d['median'])/d['mad_scale']).max(axis=1).astype(np.float64)
        else:
            t=torch.from_numpy(d['scaler'].transform(v).astype(np.float32))
            with torch.no_grad():scores=(d['model'](t)-t).pow(2).mean(dim=1).numpy().astype(np.float64)
        result[name+'_score']=scores;result[name+'_alarm']=(scores>=float(d['threshold'])).astype(np.int8)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--chunksize',type=int,default=10000);args=p.parse_args()
    source=Path(args.input).resolve();dest=Path(args.output).resolve()
    if source==dest or dest.exists():p.error('Output must be a new file different from the input')
    if args.chunksize<=0:p.error('chunksize must be positive')
    torch.set_num_threads(4);saved=load();parts=[];rows=0
    for frame in pd.read_csv(source,chunksize=args.chunksize,dtype={KEY:'string','localtime':'string'}):
        result=infer(frame,saved);result.insert(0,'input_row_index',np.arange(rows,rows+len(frame)));parts.append(result);rows+=len(frame)
    if not parts:p.error('No input observations')
    dest.parent.mkdir(parents=True,exist_ok=True)
    pd.concat(parts,ignore_index=True).to_csv(dest,index=False,float_format='%.17g',compression='infer')
    print(json.dumps({'input_rows':rows,'models':len(saved),'output':str(dest),'meaning':'Current observation scores; not verified fault probabilities'},ensure_ascii=False))

if __name__=='__main__':main()
