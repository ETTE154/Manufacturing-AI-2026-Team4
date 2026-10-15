"""Frozen local Chronos2 H/D contexts and train-only mean-blend validation.

The official point column is a median. Alpha0 is exactly a mean baseline,
and cannot establish foundation-model skill. No download, upload or May reads.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import argparse,hashlib,json,os,time
import numpy as np
import pandas as pd
from neural_resampling import (OUT,EXP,ROOT,OLD,VALID,TEST,FINAL,POINTS,THREADS,read,write,sha,
                               prefix,complete_periods,metrics,summary,pred_frame,native_csv,hourly_map,update_summary)
from resample_common import load_hourly_target
os.environ['HF_HOME']=str(OUT/'runtime_cache')
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['DO_NOT_TRACK']='1'

def pipeline():
    import torch
    torch.set_num_threads(THREADS)
    from chronos import Chronos2Pipeline
    p=read(OLD/'chronos2/provenance.json');snapshot=Path(p['local_snapshot'])
    for filename,expected in p['files'].items():
        with (snapshot/filename).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        assert actual==expected,filename
    model=Chronos2Pipeline.from_pretrained(str(snapshot),device_map='cpu',local_files_only=True)
    return model,p

def native_series(h,resolution):
    return h if resolution=='hourly' else complete_periods(h,'D').rate

def raw_predict(pipe,h,resolution,context):
    s=native_series(h,resolution);count=len(s) if context=='full' else min(int(context),len(s))
    selected=s.iloc[-count:];points=745 if resolution=='hourly' else 32;freq='h' if resolution=='hourly' else 'D'
    frame=pd.DataFrame({'id':'aggregate','timestamp':selected.index,'target':selected.to_numpy(float)})
    out=pipe.predict_df(frame,id_column='id',timestamp_column='timestamp',target='target',
                       prediction_length=points,quantile_levels=[.1,.5,.9],batch_size=1,
                       context_length=8192 if context=='full' else int(context),freq=freq).sort_values('timestamp')
    origin=h.index[-1]+pd.Timedelta(hours=1);expected=pd.date_range(origin,periods=points,freq=freq)
    assert pd.DatetimeIndex(out.timestamp).equals(expected)
    p=out.predictions.to_numpy(float);assert np.isfinite(p).all()
    # Verify official median interpretation on each returned path, if exposed.
    if '0.5' in out.columns:assert np.array_equal(p,out['0.5'].to_numpy(float))
    return p

def cached_raw(pipe,h,resolution,context,path):
    if path.exists():
        f=pd.read_csv(path,float_precision='round_trip');p=f.forecast_rate_kwh_per_hour.to_numpy(float)
        expected=pd.date_range(h.index[-1]+pd.Timedelta(hours=1),periods=745 if resolution=='hourly' else 32,
                               freq='h' if resolution=='hourly' else 'D')
        assert pd.DatetimeIndex(pd.to_datetime(f.timestamp)).equals(expected)
        return p
    p=raw_predict(pipe,h,resolution,context);native_csv(path,p,h.index[-1]+pd.Timedelta(hours=1),resolution);return p

def calibrated(raw,h,resolution,alpha):
    level=float(native_series(h,resolution).mean())
    return float(alpha)*np.asarray(raw,float)+(1-float(alpha))*level

def run(resolution,pipe,provenance,y):
    dest=OUT/resolution/'chronos2';dest.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    contexts=['full',720 if resolution=='hourly' else 28]
    configs=[{'context':context,'mean_blend_alpha':alpha,'mapping_profile':'uniform'} for context in contexts for alpha in [0.,.25,1.]]
    write(dest/'search_plan.json',{'configs':configs,'validation_origins':list(VALID),'selection':'Mean validation-origin Late505 MAE only',
         'weights_frozen':True,'checkpoint_revision':provenance['revision'],'point':'official median0.5','May_labels_read':False})
    raws={}
    for context in contexts:
        for fold,origin in enumerate(VALID,1):
            h=prefix(y,origin);raws[(context,origin)]=cached_raw(pipe,h,resolution,context,dest/f'raw_context{context}_fold{fold}.csv')
            print(f'Chronos2 {resolution} context{context} fold{fold} completed {time.perf_counter()-started:.1f}s',flush=True)
    records=[];frames={}
    for ci,cfg in enumerate(configs):
        f=pd.concat([pred_frame('chronos2',resolution,'validation',fold,origin,y,
                     calibrated(raws[(cfg['context'],origin)],prefix(y,origin),resolution,cfg['mean_blend_alpha']))
                     for fold,origin in enumerate(VALID,1)],ignore_index=True)
        csv=dest/f'candidate{ci}_validation_predictions.csv';f.to_csv(csv,index=False,encoding='utf-8-sig')
        stats=summary(f);records.append({'candidate':ci,'config':cfg,'validation':stats,
                   'selection_objective_mae_kwh':stats['mean_origin_late505_mae_kwh'],'prediction_csv':str(csv)})
        frames[ci]=f
    chosen=min(records,key=lambda r:r['selection_objective_mae_kwh']);cfg=chosen['config']
    lock={'model':'chronos2','scenario':resolution,'selected_candidate':chosen['candidate'],'selected_config':cfg,
          'candidate_results':records,'selection_objective':'Mean three validation-origin Late505 MAE only','validation_origins':list(VALID),
          'latest_validation_label':str(max(VALID)+pd.Timedelta(hours=744)),'test_labels_used_for_selection':False,
          'May_labels_read':False,'weights_frozen':True,'checkpoint_revision':provenance['revision'],
          'mean_blend_fit_scope':'Each origin historical native series mean only; no later-origin joint-series attention',
          'alpha0_warning':'Exactly mean baseline if alpha0; no learned foundation-model improvement',
          'daily_mapping_locked':'Uniform repeat24 crop745','locked_before_internal_test':True}
    assert pd.Timestamp(lock['latest_validation_label'])<TEST
    write(dest/'selection_lock.json',lock);write(dest/'candidate_results.json',{'candidate_results':records})
    frames[chosen['candidate']].to_csv(dest/'validation_predictions.csv',index=False,encoding='utf-8-sig')
    fits={};replays=[];test_stats=None
    for split,origin in [('internal_test',TEST),('future_unlabelled',FINAL)]:
        h=prefix(y,origin);raw=cached_raw(pipe,h,resolution,cfg['context'],dest/f'raw_{split}.csv')
        native=calibrated(raw,h,resolution,cfg['mean_blend_alpha']);f=pred_frame('chronos2',resolution,split,0,origin,y,native)
        csv=dest/('test_predictions.csv' if split=='internal_test' else 'future_path.csv');f.to_csv(csv,index=False,encoding='utf-8-sig')
        native_csv(dest/f'{split}_native_predictions.csv',native,origin,resolution)
        s=native_series(h,resolution)
        meta={'family':'chronos2','resolution':resolution,'selected_config':cfg,'checkpoint_snapshot_read_only':provenance['local_snapshot'],
              'revision':provenance['revision'],'files_sha256':provenance['files'],'history_end':str(h.index[-1]),
              'history_hour_count':len(h),'native_history_times':[str(t) for t in s.index],'native_history_values':s.to_list(),
              'mean_blend_level':float(s.mean()),'input_context_count':len(s) if cfg['context']=='full' else min(cfg['context'],len(s)),
              'trainable_weights':False,'point_semantics':'Official0.5quantile median','mapping_profile':'uniform',
              'raw_native_prediction_path':str(dest/f'raw_{split}.csv'),'raw_native_prediction_sha256':sha(dest/f'raw_{split}.csv'),
              'stored_native_forecast':native.tolist(),'forecast_origin':str(origin),'May_labels_read':False,
              'unit':'Provisional guide kWh/h; meter hierarchy unresolved'}
        modelpath=dest/'saved_models'/f'{split}.json';write(modelpath,meta)
        again=cfg['mean_blend_alpha']*pd.read_csv(meta['raw_native_prediction_path'],float_precision='round_trip').forecast_rate_kwh_per_hour.to_numpy(float)+(1-cfg['mean_blend_alpha'])*meta['mean_blend_level']
        assert np.array_equal(again,native)
        poisoned=y.copy();poisoned.loc[poisoned.index>=origin]=1e12
        assert native_series(prefix(poisoned,origin),resolution).equals(s)
        fits[split]={'saved_model':str(modelpath),'prediction_csv':str(csv),'native_prediction_csv':str(dest/f'{split}_native_predictions.csv'),
                     'history_end':str(h.index[-1]),'native_history_count':len(s),'mean_blend_level':float(s.mean()),'trainable_weights':False}
        replays.append({'split':split,'cached_raw_plus_locked_blend_replay_max_error':float(np.abs(again-native).max()),
                        'future_poison_native_input_unchanged':True,'raw_csv_sha256':meta['raw_native_prediction_sha256'],
                        'full_frozen_checkpoint_forward_replay':'Not rerun here; reproducible CLI provided. Cached raw SHA and locked blend exact replay verified.'})
        if split=='internal_test':test_stats=summary(f)
    result={'family':'chronos2','scenario':resolution,'selected_candidate':chosen['candidate'],'selected_config':cfg,
            'validation':chosen['validation'],'internal_test':test_stats,'fits':fits,'replay_verification':replays,
            'selection_lock':str(dest/'selection_lock.json'),'candidate_results_json':str(dest/'candidate_results.json'),
            'validation_prediction_csv':str(dest/'validation_predictions.csv'),'may_labels_read':False,
            'alpha0_baseline_equivalent':cfg['mean_blend_alpha']==0,'weight_finetuning':False,
            'input_scope':'Aggregate hourly target only' if resolution=='hourly' else 'Completed-day hourly-average aggregate energy rate',
            'target_mode':'Frozen official median forecast with CV-selected train-only historical-mean blend',
            'daily_mapping':'32 daily rates uniformly repeat24 crop745' if resolution=='daily' else None,
            'provenance':provenance,'runtime_seconds':time.perf_counter()-started,
            'claim_limits':['Pinned official120M pretrained checkpoint, no current-SOTA claim','Alpha0 is simple mean baseline',
                            'No calibrated prediction interval claim for blended point path']}
    write(dest/'results.json',result)
    print(json.dumps({'done':resolution+'/chronos2','selected':cfg,'cv505':chosen['selection_objective_mae_kwh'],
                      'test505':test_stats['late505']['mae_kwh'],'seconds':result['runtime_seconds']}),flush=True)

def predict(metadata,output,recompute=False,pipe=None):
    m=read(metadata);assert sha(m['raw_native_prediction_path'])==m['raw_native_prediction_sha256']
    if recompute:
        s=pd.Series(m['native_history_values'],index=pd.to_datetime(m['native_history_times']))
        # Daily saved native series is already completed rate; invoke official API directly.
        cfg=m['selected_config'];count=len(s) if cfg['context']=='full' else min(cfg['context'],len(s));s=s.iloc[-count:]
        pipe=pipe or pipeline()[0];frame=pd.DataFrame({'id':'aggregate','timestamp':s.index,'target':s.to_numpy()})
        r=pipe.predict_df(frame,id_column='id',timestamp_column='timestamp',target='target',prediction_length=745 if m['resolution']=='hourly' else 32,
                         quantile_levels=[.1,.5,.9],batch_size=1,context_length=8192 if cfg['context']=='full' else int(cfg['context']),
                         freq='h' if m['resolution']=='hourly' else 'D').sort_values('timestamp')
        raw=r.predictions.to_numpy(float)
        reference=pd.read_csv(m['raw_native_prediction_path'],float_precision='round_trip').forecast_rate_kwh_per_hour.to_numpy(float)
        assert np.array_equal(raw,reference),'Official frozen raw forward does not exactly replay cached median path'
    else:raw=pd.read_csv(m['raw_native_prediction_path'],float_precision='round_trip').forecast_rate_kwh_per_hour.to_numpy(float)
    native=m['selected_config']['mean_blend_alpha']*raw+(1-m['selected_config']['mean_blend_alpha'])*m['mean_blend_level']
    idx=pd.date_range(m['forecast_origin'],periods=745,freq='h');p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'timestamp':idx,'predicted_energy_kwh':hourly_map(native,m['resolution'])}).to_csv(p,index=False,encoding='utf-8-sig')
    return float(np.abs(native-np.asarray(m['stored_native_forecast'])).max())

def main():
    p=argparse.ArgumentParser();p.add_argument('--resolution',choices=['hourly','daily','both'],default='both')
    p.add_argument('--predict',action='store_true');p.add_argument('--metadata');p.add_argument('--output');p.add_argument('--recompute',action='store_true');a=p.parse_args()
    if a.predict:
        if not a.metadata or not a.output:p.error('--metadata and --output required')
        err=predict(a.metadata,a.output,a.recompute);print(f'Replay max native difference={err:.12g}');return
    y=load_hourly_target();pipe,provenance=pipeline()
    for resolution in ['hourly','daily'] if a.resolution=='both' else [a.resolution]:run(resolution,pipe,provenance,y)
    update_summary()
if __name__=='__main__':main()
