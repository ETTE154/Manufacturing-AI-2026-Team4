"""Frozen official Chronos-2 context and past-only level-blending search.

No finetuning and no May-label access. Alpha=0 is explicitly baseline-equivalent,
not a learned foundation-model improvement. All input processing is local.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import os
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
from opt_common import (ROOT,OUT,THREADS,DAY_FOLD_STARTS,APRIL_TEST_ORIGINS,
                        LONG_VALID_ORIGIN,LONG_TEST_ORIGIN,load_target,history,
                        future,actual,metrics,prediction_frame,day_summary,
                        write_json,require_source_approval)

os.environ['HF_HOME']=str(OUT/'runtime_cache')
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['DO_NOT_TRACK']='1'
CONTEXTS=[168,720,'full']
CONFIGS=[]
for context in CONTEXTS:
    CONFIGS.append({'context':context,'calibration':'none','alpha':1.0})
    for calibration in ['historical_mean','ema720']:
        for alpha in [0.0,.25,.5,.75]:
            CONFIGS.append({'context':context,'calibration':calibration,'alpha':alpha})

def pipeline():
    import torch
    torch.set_num_threads(THREADS)
    from chronos import Chronos2Pipeline
    provenance=json.loads((ROOT/'output/modeling/forecast/chronos2/provenance.json').read_text(encoding='utf-8'))
    snapshot=Path(provenance['local_snapshot'])
    for filename,expected in provenance['files'].items():
        with (snapshot/filename).open('rb') as handle:observed=hashlib.file_digest(handle,'sha256').hexdigest()
        if observed!=expected:raise ValueError('Frozen official Chronos file checksum changed')
    model=Chronos2Pipeline.from_pretrained(str(snapshot),device_map='cpu',local_files_only=True)
    write_json(OUT/'chronos_model_provenance.json',provenance)
    return model,provenance

def raw_predict(pipe,h,points,context):
    count=len(h) if context=='full' else min(int(context),len(h))
    selected=h.iloc[-count:]
    data=pd.DataFrame({'id':'aggregate','timestamp':selected.index,'target':selected.values})
    # Predict one origin at a time: later-origin contexts can never influence an
    # earlier origin via joint series/group attention or shared data frames.
    result=pipe.predict_df(data,id_column='id',timestamp_column='timestamp',target='target',
                           prediction_length=points,quantile_levels=[.1,.5,.9],batch_size=1,
                           context_length=8192 if context=='full' else int(context),freq='h')
    result=result.sort_values('timestamp')
    expected=future(h.index[-1]+pd.Timedelta(hours=1),points)
    if not pd.DatetimeIndex(result.timestamp).equals(expected):raise ValueError('Chronos forecast clock differs')
    point=result.predictions.to_numpy(dtype=float)
    if not np.isfinite(point).all():raise ValueError('Nonfinite foundation forecast')
    return point

def calibrate(raw,h,cfg):
    if cfg['calibration']=='none':return raw.copy()
    if cfg['calibration']=='historical_mean':level=float(h.mean())
    elif cfg['calibration']=='ema720':level=float(h.ewm(span=720,adjust=False).mean().iloc[-1])
    else:raise ValueError(cfg['calibration'])
    return cfg['alpha']*raw+(1-cfg['alpha'])*level

def raw_paths(pipe,y,origins,points,context,destination,split):
    filename=destination/f'raw_{split}_context_{context}.csv'
    if filename.exists():
        old=pd.read_csv(filename);old['id']=pd.to_datetime(old['id']);old['origin']=pd.to_datetime(old['origin'])
        result={origin:g.sort_values('id')['predicted_kwh'].to_numpy(dtype=float) for origin,g in old.groupby('origin')}
        if set(result)==set(origins) and all(len(path)==points for path in result.values()):return result
        raise ValueError('Cached raw forecast does not match declared origins')
    paths={};frames=[];started=time.monotonic()
    for i,origin in enumerate(origins):
        h=history(y,origin);idx=future(origin,points);path=raw_predict(pipe,h,points,context);paths[origin]=path
        frames.append(prediction_frame('chronos2_raw','dayahead24' if points==24 else 'long745',split,0,origin,idx,actual(y,idx) if idx[-1]<=y.index[-1] else np.repeat(np.nan,points),path))
        if i%14==0:print(f'Chronos context{context} {split} origin{i+1}/{len(origins)} {time.monotonic()-started:.1f}s',flush=True)
    pd.concat(frames,ignore_index=True).to_csv(filename,index=False,encoding='utf-8-sig')
    return paths

def frames_for_config(y,origins,paths,cfg,points,split):
    frames=[]
    for origin in origins:
        idx=future(origin,points);h=history(y,origin);path=calibrate(paths[origin],h,cfg)
        a=actual(y,idx) if idx[-1]<=y.index[-1] else np.repeat(np.nan,points)
        fold=0
        if points==24 and split=='validation':
            fold=next(i for i,start in enumerate(DAY_FOLD_STARTS,1) if start<=origin<start+pd.Timedelta(days=14))
        frames.append(prediction_frame('chronos2','dayahead24' if points==24 else 'long745',split,fold,origin,idx,a,path))
    return pd.concat(frames,ignore_index=True)

def run(scenario):
    require_source_approval();y=load_target();pipe,provenance=pipeline();started=time.monotonic()
    destination=OUT/scenario/'chronos2';destination.mkdir(parents=True,exist_ok=True)
    if scenario=='dayahead24':
        origins=pd.DatetimeIndex(np.concatenate([pd.date_range(t,periods=14,freq='D').values for t in DAY_FOLD_STARTS]));points=24
    else:origins=pd.DatetimeIndex([LONG_VALID_ORIGIN]);points=745
    raw={context:raw_paths(pipe,y,origins,points,context,destination,'validation') for context in CONTEXTS}
    records=[];frames=[]
    for candidate,cfg in enumerate(CONFIGS):
        frame=frames_for_config(y,origins,raw[cfg['context']],cfg,points,'validation')
        if scenario=='dayahead24':stats=day_summary(frame);objective=stats['mean_origin_path_mae_kwh']
        else:
            a=frame.actual_kwh.to_numpy();p=frame.predicted_kwh.to_numpy();stats={'path_P745':metrics(a,p),'late_P505':metrics(a[240:745],p[240:745])};objective=stats['path_P745']['mae_kwh']
        records.append({'candidate':candidate,'config':cfg,'objective_mae_kwh':objective,'validation':stats})
        frames.append(frame)
    chosen=min(records,key=lambda r:r['objective_mae_kwh']);cfg=chosen['config']
    locked={'model':'chronos2','scenario':scenario,'selected_config':cfg,'selection_uses_May_labels':False,
            'selection_uses_April_test':False,'configuration_count':len(CONFIGS),'candidate_results':records,
            'selection_basis':'70 day-origin mean path MAE' if points==24 else 'Single Feb25 P745 MAE',
            'mode':'Official frozen pretrained zero-shot checkpoint plus validation-selected past-only level blend; no weight finetuning',
            'degenerate_blend_note':'alpha=0 exactly equals the chosen simple level estimator; it cannot support a foundation-model skill claim',
            'point_semantics':'Official predictions column corresponds to the model median (0.5 quantile)',
            'provenance':provenance}
    write_json(destination/'selection_lock.json',locked)
    frames[chosen['candidate']].to_csv(destination/'validation_predictions.csv',index=False,encoding='utf-8-sig')
    test_origins=APRIL_TEST_ORIGINS if points==24 else pd.DatetimeIndex([LONG_TEST_ORIGIN])
    test_raw=raw_paths(pipe,y,test_origins,points,cfg['context'],destination,'test')
    test=frames_for_config(y,test_origins,test_raw,cfg,points,'April_test' if points==24 else 'internal_test')
    test.to_csv(destination/'test_predictions.csv',index=False,encoding='utf-8-sig')
    if points==24:test_stats=day_summary(test)
    else:
        a=test.actual_kwh.to_numpy();p=test.predicted_kwh.to_numpy();test_stats={'path_P745':metrics(a,p),'late_P505':metrics(a[240:745],p[240:745])}
    origin=y.index[-1]+pd.Timedelta(hours=1);idx=future(origin,points)
    final_raw=raw_predict(pipe,y,points,cfg['context']);final_path=calibrate(final_raw,y,cfg)
    prediction_frame('chronos2',scenario,'future_unlabelled',0,origin,idx,np.repeat(np.nan,points),final_path).to_csv(destination/'future_path.csv',index=False,encoding='utf-8-sig')
    model_info={'checkpoint_snapshot_read_only':provenance['local_snapshot'],'revision':provenance['revision'],
                'checkpoint_files_sha256':provenance['files'],'selected_config':cfg,'history_end':str(y.index[-1]),
                'history_count':len(y),'history_values_kwh':y.to_list(),'forecast_points':points,
                'alpha_zero_baseline_equivalent':cfg['alpha']==0,'trainable_weights':False}
    write_json(destination/'saved_model'/'model_metadata.json',model_info)
    result={'model':'chronos2','scenario':scenario,'selected_config':cfg,'validation':chosen['validation'],
            'test':test_stats,'saved_model':model_info,'runtime_seconds':time.monotonic()-started,'may_labels_read':False,
            'trials_count':len(CONFIGS),'input_scope':'Aggregate univariate past target only; chronological past-only level calibration',
            'selection_lock':str(destination/'selection_lock.json'),'model_card_notes':[locked['mode'],locked['degenerate_blend_note'],locked['point_semantics'],
            'Only point forecasts are optimized; blended quantiles are not claimed as calibrated uncertainty intervals.']}
    write_json(destination/'results.json',result)
    print(json.dumps({'selected_config':cfg,'validation':chosen['validation'],'test':test_stats},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scenario',choices=['dayahead24','long745'],required=True)
    p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    if args.prepare_only:write_json(OUT/'chronos_search_plan.json',{'configs':CONFIGS,'weight_finetuning':False});print('Chronos plan prepared; no inference/May labels')
    else:run(args.scenario)
