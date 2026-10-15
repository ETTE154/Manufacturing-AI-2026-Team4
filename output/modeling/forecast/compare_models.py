"""Freeze deployment on validation, then score saved fixed forecast paths."""
from pathlib import Path
import json, argparse, hashlib
import numpy as np
import pandas as pd
from common import OUT, write_json, score, may_labels_after_lock

DISPLAY={'arima':'ARIMA','xgboost':'XGBoost','patchtst':'PatchTST','itransformer':'iTransformer',
         'chronos2':'Chronos-2','baseline_mean':'평균 기준선','baseline_median':'중앙값 기준선',
         'baseline_weekday_hour_mean':'요일·시간 기준선'}
ALIASES={'iTransformer':'itransformer','Chronos-2':'chronos2','PatchTST':'patchtst'}

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def inputs():
    initial=read_json(OUT/'arima_xgb_results.json')
    models={key:dict(value) for key,value in initial['models'].items() if key in DISPLAY}
    patch=read_json(OUT/'patchtst/results.json')
    models['patchtst']={key:patch[key] for key in ['validation','internal_test']}
    models['patchtst']['config']=patch['frozen_config'];models['patchtst']['epochs']=patch['frozen_epochs']
    for folder in ['itransformer','chronos2']:
        result=read_json(OUT/folder/'results.json')
        models[folder]={s:result['splits'][s]['metrics'] for s in ['validation','internal_test']}
        models[folder]['config']=read_json(OUT/folder/'selection_lock.json')
    for key,value in models.items():
        value['display_name']=DISPLAY[key]
        value['input_scope']='과거 설비별 13개 시간 전력량' if key=='itransformer' else '과거 합계 시간 전력량; XGB는 알려진 달력 포함'
        value['training_type']={'chronos2':'공식 120M 사전학습 가중치, zero-shot','itransformer':'공식 구조 축소, 현 데이터 재학습','patchtst':'공식 구조 축소, 현 데이터 재학습'}.get(key,'현재 데이터로 학습')
    csvs=[OUT/'internal_predictions.csv']+[OUT/f/'internal_predictions.csv' for f in ['patchtst','itransformer','chronos2']]
    frames=[]
    for path in csvs:
        df=pd.read_csv(path,parse_dates=['id','origin']);df['model']=df['model'].replace(ALIASES)
        frames.append(df[df['model'].isin(DISPLAY)])
    all_internal=pd.concat(frames,ignore_index=True)
    for model in models:
        for split in ['validation','internal_test']:
            rows=all_internal[(all_internal.model==model)&(all_internal.split==split)].sort_values('id')
            if len(rows)!=745:raise ValueError(f'{model} {split} count={len(rows)}')
            existing=models[model][split]['P745']; calculated=score(rows.actual_kwh,rows.predicted_kwh)
            np.testing.assert_allclose(existing['mae_kwh'],calculated['mae_kwh'],rtol=0,atol=1e-7)
            tail=rows[rows.horizon_offset_hours>=240]
            models[model][split]['LateP505']=score(tail.actual_kwh,tail.predicted_kwh)
    return models,all_internal

def freeze():
    models,frames=inputs()
    selected=min(models,key=lambda m:models[m]['validation']['P745']['mae_kwh'])
    files=[OUT/'arima_xgb_selection_lock.json']+[OUT/f/'selection_lock.json' for f in ['patchtst','itransformer','chronos2']]
    lock={'selected_model':selected,'selected_model_id':selected,
          'selection_metric':'validation.P745.mae_kwh',
          'validation_metric':models[selected]['validation']['P745']['mae_kwh'],
          'selection_uses_may_labels':False,'selection_uses_internal_test_labels':False,
          'candidate_validation_mae_kwh':{m:models[m]['validation']['P745']['mae_kwh'] for m in models},
          'config':models[selected].get('config',models[selected].get('frozen_config',{})),
          'model_lock_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
          'baseline_fallback':'baseline_mean',
          'practical_note':'Tiny validation improvement is not evidence of superiority; deployment choice follows predeclared lowest validation MAE, mean remains a simple fallback.',
          'holdout_disclosure':'May labels were previously inspected in descriptive EDA; no model or parameter selection here uses them.'}
    path=OUT/'deployment_selection_lock.json'
    if path.exists():
        old=read_json(path)
        if old!=lock:raise ValueError('Attempt to replace frozen deployment choice')
    else:write_json(path,lock)
    write_json(OUT/'comparison_results.json',{'status':'internal_locked_awaiting_may','models':models,'deployment':lock})
    print('Deployment frozen:',selected,'validation MAE',lock['validation_metric'],flush=True)

def paired_block_interval(rows,baseline,seed=2026):
    joined=rows[['id','absolute_error_kwh']].merge(baseline[['id','absolute_error_kwh']],on='id',suffixes=('_model','_mean'),validate='one_to_one').sort_values('id')
    delta=(joined.absolute_error_kwh_model-joined.absolute_error_kwh_mean).to_numpy()
    n=len(delta);rng=np.random.default_rng(seed); samples=[]
    # Circular 24-hour blocks. Conditional on this single origin, not population confidence.
    for _ in range(2000):
        starts=rng.integers(0,n,size=int(np.ceil(n/24)))
        indexes=((starts[:,None]+np.arange(24))%n).ravel()[:n]
        samples.append(float(delta[indexes].mean()))
    return {'count':n,'delta_mae_model_minus_mean_kwh':float(delta.mean()),
            'conditional_24h_block_bootstrap_95_interval_kwh':np.quantile(samples,[.025,.975]).tolist(),
            'resamples':2000,'seed':seed,'limitation':'Within one fixed-origin path; does not quantify variation across independent origins or real plants'}

def complete():
    lock=read_json(OUT/'deployment_selection_lock.json')
    for name,digest in lock['model_lock_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Model selection changed after lock')
    models,internal=inputs();labels=may_labels_after_lock();mayframes=[];paths=[]
    baseline_data=pd.read_csv(OUT/'may_labelled_predictions.csv',parse_dates=['id','origin'])
    baseline_data=baseline_data[baseline_data.model.isin(DISPLAY)]
    mayframes.append(baseline_data)
    for folder,filename in [('patchtst','may_full_bridge.csv'),('itransformer','may_full_path.csv'),('chronos2','may_full_path.csv')]:
        path=OUT/folder/filename;df=pd.read_csv(path,parse_dates=['id','origin']);df['model']=folder
        if len(df)!=745 or (df.horizon_offset_hours.to_numpy()!=np.arange(745)).any():raise ValueError('Bridge range mismatch')
        if df.actual_kwh.notna().any():raise ValueError('May labels contaminated pre-scoring forecast path')
        paths.append(df)
        matched=df[df.id.isin(labels.index)].copy().sort_values('id')
        if len(matched)!=505:raise ValueError('May labels did not align')
        matched['actual_kwh']=labels.reindex(matched.id).to_numpy()
        matched['absolute_error_kwh']=np.abs(matched.actual_kwh-matched.predicted_kwh)
        matched['split']='may_labelled';mayframes.append(matched)
    may=pd.concat(mayframes,ignore_index=True)
    for model in models:
        rows=may[may.model==model].sort_values('id')
        if len(rows)!=505:raise ValueError(f'May rows missing for {model}: {len(rows)}')
        metrics=score(rows.actual_kwh,rows.predicted_kwh)
        variance=float(np.mean((rows.actual_kwh-rows.actual_kwh.mean())**2))
        metrics['r2']=1-float(np.mean((rows.actual_kwh-rows.predicted_kwh)**2))/variance
        metrics['cost_mae_krw_project_rate']=metrics['mae_kwh']*180
        metrics['carbon_mae_kgco2_project_factor']=metrics['mae_kwh']*.424
        models[model]['may']={'P505':metrics}
        for split,frame in [('internal_test',internal),('may_labelled',may)]:
            part=frame[frame.split==split]
            analysis=paired_block_interval(part[part.model==model],part[part.model=='baseline_mean'])
            analysis['horizon']='P745 offsets0..744' if split=='internal_test' else 'P505 offsets240..744'
            models[model].setdefault('paired_error_analysis',{})[split]=analysis
        if model=='chronos2' and 'q01' in rows and 'q09' in rows:
            models[model]['may']['interval']={'nominal_coverage':0.8,'empirical_coverage':float(((rows.actual_kwh>=rows.q01)&(rows.actual_kwh<=rows.q09)).mean()),'mean_width_kwh':float((rows.q09-rows.q01).mean()),'calibrated':False}
    combined=pd.concat([internal,may],ignore_index=True).sort_values(['model','split','id'])
    if combined.duplicated(['model','split','id']).any():raise ValueError('Duplicate model/split/timestamp')
    combined.to_csv(OUT/'all_predictions.csv',index=False,encoding='utf-8-sig')
    selected=lock['selected_model']; best_may=min(models,key=lambda m:models[m]['may']['P505']['mae_kwh'])
    result={'status':'completed','models':models,'deployment':lock,'may_descriptive_best_not_selected':best_may,
      'evaluation_note':'Deployment fixed on validation P745. May P505 and internal LateP505 are offsets240..744; model selection not rerun.',
      'protocol_file':str(OUT/'protocol.json'),'main_forecast_model_count':5,'baseline_count':3}
    write_json(OUT/'comparison_results.json',result)
    print(json.dumps({m:{'valid':v['validation']['P745']['mae_kwh'],'test':v['internal_test']['P745']['mae_kwh'],'may':v['may']['P505']['mae_kwh']} for m,v in models.items()},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['lock','score'],required=True)
    args=p.parse_args();freeze() if args.stage=='lock' else complete()
