"""Local frozen official Chronos2; validate context and past-only level blend."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import os,hashlib,time
import numpy as np
import pandas as pd
from common import *
os.environ['HF_HOME']=str(OUT/'runtime_cache');os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
def pipeline():
    import torch
    torch.set_num_threads(THREADS)
    from chronos import Chronos2Pipeline
    p=read(ROOT/'output/modeling/forecast/chronos2/provenance.json');snapshot=Path(p['local_snapshot'])
    for file,expected in p['files'].items():
        with (snapshot/file).open('rb') as h:assert hashlib.file_digest(h,'sha256').hexdigest()==expected
    return Chronos2Pipeline.from_pretrained(str(snapshot),device_map='cpu',local_files_only=True),p
def raw_predict(pipe,h,res,context):
    s=native(h,res);s=s if context=='full' else s.iloc[-int(context):]
    f=pd.DataFrame({'id':'aggregate','timestamp':s.index,'target':s.to_numpy()});points=745 if res=='hourly' else 32
    out=pipe.predict_df(f,id_column='id',timestamp_column='timestamp',target='target',prediction_length=points,
                       quantile_levels=[.1,.5,.9],batch_size=1,context_length=8192 if context=='full' else int(context),freq='h' if res=='hourly' else 'D').sort_values('timestamp')
    origin=h.index[-1]+pd.Timedelta(hours=1);assert pd.DatetimeIndex(out.timestamp).equals(pd.date_range(origin,periods=points,freq='h' if res=='hourly' else 'D'))
    return out.predictions.to_numpy(float)
def blend(raw,h,cfg):
    s=native(h,cfg['resolution']);level=s.mean() if cfg['level']=='mean' else s.median()
    return cfg['alpha']*np.asarray(raw)+(1-cfg['alpha'])*level
def main():
    assert_training_permitted()
    y=load_y();pipe,provenance=pipeline();dest=OUT/'chronos2';dest.mkdir(parents=True,exist_ok=True);records=[];paths={};start=time.perf_counter()
    for res in ['hourly','daily']:
        contexts=['full',168,720] if res=='hourly' else ['full',7,28];raws={}
        for context in contexts:
            for fold,origin in enumerate(VALID,1):
                h=prefix(y,origin);raw=raw_predict(pipe,h,res,context);raws[(context,origin)]=raw
                pd.DataFrame({'raw_forecast_rate':raw}).to_csv(dest/f'raw_{res}_context{context}_fold{fold}.csv',index=False)
        configs=[{'resolution':res,'context':'full','alpha':0.,'level':level} for level in ['mean','median']]
        configs +=[{'resolution':res,'context':context,'alpha':alpha,'level':level} for context in contexts for alpha in [.05,.1,.25,.5,.75,1.] for level in ['mean','median']]
        for cfg in configs:
            ci=len(records);f=pd.concat([frame('chronos2','validation',fold,origin,y,hourly_map(blend(raws[(cfg['context'],origin)],prefix(y,origin),cfg),res))
                                        for fold,origin in enumerate(VALID,1)],ignore_index=True)
            stats=summary(f);records.append({'candidate':ci,'config':cfg,'validation':stats,'objective_mae_kwh':stats['mean_origin_late505_mae_kwh']});paths[ci]=f
    chosen=min(records,key=lambda r:r['objective_mae_kwh']);cfg=chosen['config']
    write(dest/'selection_lock.json',{'chosen':chosen,'candidates':records,'weights_frozen':True,'selection_has_test_or_May':False,'locked_before_test':True,
       'alpha0_is_level_baseline':cfg['alpha']==0,'point':'Official0.5quantile median','checkpoint_provenance':provenance})
    paths[chosen['candidate']].to_csv(dest/'validation_predictions.csv',index=False,encoding='utf-8-sig');fits={};test_stats=None
    for split,origin in [('internal_test',TEST),('future_unlabelled',FINAL)]:
        h=prefix(y,origin);raw=raw_predict(pipe,h,cfg['resolution'],cfg['context']);native_p=blend(raw,h,cfg);p=hourly_map(native_p,cfg['resolution'])
        meta={'config':cfg,'provenance':provenance,'history_values':h.to_list(),'history_times':[str(t) for t in h.index],
              'history_end':str(h.index[-1]),'stored_raw':raw.tolist(),'stored_native':native_p.tolist(),'forecast_origin':str(origin),
              'weights_frozen':True,'unit':'provisional guide kWh','no_May_labels':True}
        path=dest/f'{split}.json';write(path,meta);f=frame('chronos2',split,0,origin,y,p);csv=dest/('test_predictions.csv' if split=='internal_test' else 'future_path.csv');f.to_csv(csv,index=False,encoding='utf-8-sig')
        again=raw_predict(pipe,h,cfg['resolution'],cfg['context']);error=float(abs(again-raw).max());assert error==0
        fits[split]={'saved_model':str(path),'prediction_csv':str(csv),'history_end':str(h.index[-1]),'official_forward_raw_replay_max_error':error}
        if split=='internal_test':test_stats=summary(f)
    result={'family':'chronos2','selected_config':cfg,'validation':chosen['validation'],'internal_test':test_stats,'fits':fits,
             'candidate_count':len(records),'selection_lock':str(dest/'selection_lock.json'),'validation_prediction_csv':str(dest/'validation_predictions.csv'),
             'weights_frozen':True,'alpha0_baseline_equivalent':cfg['alpha']==0,'May_read':False,'provenance':provenance,'seconds':time.perf_counter()-start}
    write(dest/'results.json',result);print({'Chronos':cfg,'cv505':chosen['objective_mae_kwh'],'test505':test_stats['late505']['mae_kwh']},flush=True)
if __name__=='__main__':main()
