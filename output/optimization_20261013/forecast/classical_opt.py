"""Budgeted ARIMA/XGBoost/baseline search in two distinct forecast scenarios.

No May labels are opened. All artifacts are confined to this new directory.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import argparse
import copy
import json
import time
import warnings
import numpy as np
import pandas as pd
from opt_common import (OUT,SEED,THREADS,DAY_FOLD_STARTS,APRIL_TEST_ORIGINS,
                        LONG_VALID_ORIGIN,LONG_TEST_ORIGIN,load_target,history,
                        future,actual,metrics,prediction_frame,day_summary,
                        write_json,require_source_approval)
from threadpoolctl import threadpool_limits
threadpool_limits(limits=THREADS)

ARIMA_CONFIGS = [{'order':list(order)} for order in [
    (0,0,0),(1,0,0),(2,0,0),(0,0,1),(0,0,2),(1,0,1),(2,0,1),(1,0,2),
    (0,1,0),(1,1,0),(0,1,1),(1,1,1),(2,1,0)]]

def xgb_configs():
    rng = np.random.default_rng(SEED)
    result = [dict(n_estimators=50,max_depth=1,learning_rate=.03,min_child_weight=50,
                   reg_lambda=100.,subsample=1.,colsample_bytree=1.,feature_set='history_calendar',residual_shrinkage=1.),
              dict(n_estimators=150,max_depth=3,learning_rate=.03,min_child_weight=20,
                   reg_lambda=10.,subsample=1.,colsample_bytree=1.,feature_set='history_calendar',residual_shrinkage=1.)]
    while len(result)<30:
        cfg = dict(n_estimators=int(rng.choice([50,100,200,350])),max_depth=int(rng.choice([1,2,3,4])),
                   learning_rate=float(rng.choice([.01,.03,.07])),min_child_weight=int(rng.choice([5,20,50,100])),
                   reg_lambda=float(rng.choice([1,10,100])),subsample=float(rng.choice([.8,1.])),
                   colsample_bytree=float(rng.choice([.8,1.])),feature_set=str(rng.choice(['history','history_calendar'])),
                   residual_shrinkage=float(rng.choice([.25,.5,1.])))
        if cfg not in result: result.append(cfg)
    return result

XGB_CONFIGS = xgb_configs()
BASELINE_CONFIGS = [{'name':'train_mean'},{'name':'train_median'},{'name':'expanding_mean'}]+[
    {'name':name,'window':window} for name in ['recent_mean','ema'] for window in [24,72,168,336,720]]+[
    {'name':'calendar_ridge','alpha':alpha} for alpha in [.1,1.,10.,100.]]

def calendar(times):
    idx = pd.DatetimeIndex(times)
    hour = idx.hour.to_numpy();dow=idx.dayofweek.to_numpy();month=idx.month.to_numpy()
    return np.column_stack([np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),
                            np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7),
                            np.sin(2*np.pi*(month-1)/12),np.cos(2*np.pi*(month-1)/12),dow>=5]).astype(float)

FEATURE_NAMES = ['lag1','lag24','lag168','mean24','mean168','mean720','std24','std168',
                 'origin_hour_sin','origin_hour_cos','lead_days']
CALENDAR_NAMES = ['target_hour_sin','target_hour_cos','target_weekday_sin','target_weekday_cos',
                  'target_month_sin','target_month_cos','target_weekend']

def origin_features(values,pos,time):
    a=np.asarray(values,dtype=float)
    w24=a[max(0,pos-24):pos];w168=a[max(0,pos-168):pos];w720=a[max(0,pos-720):pos]
    return [a[pos-1],a[pos-24],a[pos-168],float(w24.mean()),float(w168.mean()),float(w720.mean()),
            float(w24.std()),float(w168.std()),np.sin(2*np.pi*time.hour/24),np.cos(2*np.pi*time.hour/24)]

def build_train(history_y,points,feature_set):
    values=history_y.to_numpy(dtype=float)
    lead_grid=list(range(24)) if points==24 else sorted(set(range(25))|set(range(48,745,24)))
    rows,targets=[],[]
    for pos in range(168,len(values),6):
        ori=history_y.index[pos];base=origin_features(values,pos,ori)
        for lead in lead_grid:
            if pos+lead>=len(values):continue
            row=base+[lead/24.]
            if feature_set=='history_calendar':row+=calendar([history_y.index[pos+lead]])[0].tolist()
            rows.append(row);targets.append(values[pos+lead])
    if not rows:raise ValueError('No admissible direct training examples')
    return np.asarray(rows,dtype=np.float32),np.asarray(targets,dtype=float)

def build_future(history_y,idx,feature_set):
    values=history_y.to_numpy(dtype=float);base=origin_features(values,len(values),idx[0])
    rows=[]
    for lead in range(len(idx)):
        row=base+[lead/24.]
        if feature_set=='history_calendar':row+=calendar([idx[lead]])[0].tolist()
        rows.append(row)
    return np.asarray(rows,dtype=np.float32)

def fit_model(name,h,points,cfg,cache=None):
    if name=='arima':
        from statsmodels.tsa.arima.model import ARIMA
        order=tuple(cfg['order'])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fit=ARIMA(h.to_numpy(),order=order,trend='c' if order[1]==0 else 'n',
                      enforce_stationarity=True,enforce_invertibility=True).fit(method_kwargs={'maxiter':200,'disp':0})
        if not fit.mle_retvals.get('converged',True):raise ValueError('ARIMA did not converge')
        return {'name':name,'object':fit,'observed_count':len(h),'config':cfg,
                'fit_metadata':{'aic_training_only':float(fit.aic),'warnings':[str(x.message) for x in caught]}}
    if name=='xgboost':
        from xgboost import XGBRegressor
        key=(len(h),points,cfg['feature_set'])
        if cache is not None and key in cache:X,y=cache[key]
        else:
            X,y=build_train(h,points,cfg['feature_set'])
            if cache is not None:cache[key]=(X,y)
        center=float(y.mean())
        params={k:v for k,v in cfg.items() if k not in ['feature_set','residual_shrinkage','objective']}
        fit=XGBRegressor(objective=cfg.get('objective','reg:squarederror'),tree_method='hist',random_state=SEED,n_jobs=THREADS,**params)
        fit.fit(X,y-center)
        return {'name':name,'object':fit,'center':center,'config':cfg,
                'fit_metadata':{'direct_examples':len(y),'target_center':center,'feature_set':cfg['feature_set'],'objective':cfg.get('objective','reg:squarederror')}}
    if name=='baselines':
        fit={'name':name,'config':cfg,'mean':float(h.mean()),'median':float(h.median())}
        if cfg['name']=='calendar_ridge':
            from sklearn.linear_model import Ridge
            fit['object']=Ridge(alpha=cfg['alpha']).fit(calendar(h.index),h.to_numpy()-fit['mean'])
        return fit
    raise ValueError(name)

def predict(model,h,idx):
    name=model['name'];cfg=model['config']
    if name=='arima':
        if len(h)<model['observed_count']:raise ValueError('ARIMA state moved backwards')
        if len(h)>model['observed_count']:
            model['object']=model['object'].extend(h.to_numpy()[model['observed_count']:])
            model['observed_count']=len(h)
        return np.asarray(model['object'].forecast(steps=len(idx)),dtype=float)
    if name=='xgboost':
        X=build_future(h,idx,cfg['feature_set'])
        return model['center']+cfg['residual_shrinkage']*np.asarray(model['object'].predict(X),dtype=float)
    kind=cfg['name']
    if kind=='train_mean':value=model['mean']
    elif kind=='train_median':value=model['median']
    elif kind=='expanding_mean':value=float(h.mean())
    elif kind=='recent_mean':value=float(h.iloc[-cfg['window']:].mean())
    elif kind=='ema':value=float(h.ewm(span=cfg['window'],adjust=False).mean().iloc[-1])
    elif kind=='calendar_ridge':return model['mean']+model['object'].predict(calendar(idx))
    else:raise ValueError(kind)
    return np.repeat(value,len(idx))

def save_model(model,h,points,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    name=model['name'];cfg=model['config']
    info={'model':name,'config':cfg,'history_end':str(h.index[-1]),'history_count':len(h),
          'forecast_points':points,'past_tail_kwh':h.iloc[-720:].to_list(),
          'history_mean':float(h.mean()),'history_median':float(h.median()),'fit_metadata':model.get('fit_metadata',{})}
    if name=='arima':
        path=destination/'model.pkl';model['object'].save(str(path))
    elif name=='xgboost':
        path=destination/'model.json';model['object'].save_model(str(path));info['target_center']=model['center']
        info['feature_names']=FEATURE_NAMES+(CALENDAR_NAMES if cfg['feature_set']=='history_calendar' else [])
    elif cfg['name']=='calendar_ridge':
        info['ridge_coef']=model['object'].coef_.tolist();info['ridge_intercept']=float(model['object'].intercept_)
        path=destination/'model_metadata.json'
    else:path=destination/'model_metadata.json'
    if name=='baselines' and cfg['name']=='ema':
        info['ema_last']=float(h.ewm(span=cfg['window'],adjust=False).mean().iloc[-1])
    info['model_path']=str(path)
    write_json(destination/'model_metadata.json',info)
    return info

def evaluate_day(name,cfg,y,origins,split,fold,fit_origin,cache=None):
    fitted=fit_model(name,history(y,fit_origin),24,cfg,cache)
    frames=[]
    for origin in origins:
        idx=future(origin,24);a=actual(y,idx);p=predict(fitted,history(y,origin),idx)
        frames.append(prediction_frame(name,'dayahead24',split,fold,origin,idx,a,p))
    frame=pd.concat(frames,ignore_index=True)
    return frame,day_summary(frame)

def run_search(name,scenario):
    require_source_approval();y=load_target();started=time.monotonic()
    destination=OUT/scenario/name;destination.mkdir(parents=True,exist_ok=True)
    configs={'arima':ARIMA_CONFIGS,'xgboost':XGB_CONFIGS,'baselines':BASELINE_CONFIGS}[name]
    records=[];candidate_frames=[];cache={}
    for candidate,cfg in enumerate(configs):
        try:
            if scenario=='dayahead24':
                frames=[];fold_stats=[]
                for fold,start in enumerate(DAY_FOLD_STARTS,1):
                    origins=pd.date_range(start,periods=14,freq='D')
                    frame,stats=evaluate_day(name,cfg,y,origins,'validation',fold,start,cache)
                    frames.append(frame);fold_stats.append(stats)
                frame=pd.concat(frames,ignore_index=True);stats=day_summary(frame)
                objective=stats['mean_origin_path_mae_kwh']
            else:
                h=history(y,LONG_VALID_ORIGIN);idx=future(LONG_VALID_ORIGIN,745)
                fit=fit_model(name,h,745,cfg,cache);p=predict(fit,h,idx);a=actual(y,idx)
                frame=prediction_frame(name,'long745','validation',0,LONG_VALID_ORIGIN,idx,a,p)
                stats={'path_P745':metrics(a,p),'late_P505':metrics(a[240:745],p[240:745])};objective=stats['path_P745']['mae_kwh'];fold_stats=[]
            record={'candidate':candidate,'config':cfg,'objective_mae_kwh':objective,'validation':stats,'fold_validation':fold_stats}
            candidate_frames.append((candidate,frame));records.append(record)
        except Exception as exc:
            records.append({'candidate':candidate,'config':cfg,'error':repr(exc)})
        if candidate%5==0:print(f'{name}/{scenario} candidate{candidate+1}/{len(configs)} elapsed{time.monotonic()-started:.1f}s',flush=True)
        write_json(destination/'search_progress.json',{'records':records,'candidate_count':len(configs),'runtime_seconds':time.monotonic()-started})
    success=[r for r in records if 'objective_mae_kwh' in r]
    if not success:raise RuntimeError('All candidates failed')
    chosen=min(success,key=lambda r:r['objective_mae_kwh']);cfg=chosen['config']
    locked={'model':name,'scenario':scenario,'selected_config':cfg,'objective':chosen['objective_mae_kwh'],
            'selected_candidate':chosen['candidate'],'selection_uses_April_test':False,'selection_uses_May_labels':False,
            'selection_basis':'70 day-origin mean path MAE' if scenario=='dayahead24' else 'Single Feb25 P745 blocked validation MAE',
            'search_records':records,'seed':SEED,'configuration_count':len(configs),
            'optimization_limit':'Budgeted search; no global optimum guarantee; post-EDA rather than pristine unseen test'}
    write_json(destination/'selection_lock.json',locked)
    valid_frame=next(frame for candidate,frame in candidate_frames if candidate==chosen['candidate'])
    valid_frame.to_csv(destination/'validation_predictions.csv',index=False,encoding='utf-8-sig')
    if scenario=='dayahead24':
        test_frame,test_stats=evaluate_day(name,cfg,y,APRIL_TEST_ORIGINS,'April_test',0,pd.Timestamp('2025-04-01'),cache)
        final_fit=fit_model(name,y,24,cfg,cache);unlabelled=future(y.index[-1]+pd.Timedelta(hours=1),24)
    else:
        test_h=history(y,LONG_TEST_ORIGIN);idx=future(LONG_TEST_ORIGIN,745);test_fit=fit_model(name,test_h,745,cfg,cache)
        p=predict(test_fit,test_h,idx);a=actual(y,idx);test_frame=prediction_frame(name,scenario,'internal_test',0,LONG_TEST_ORIGIN,idx,a,p)
        test_stats={'path_P745':metrics(a,p),'late_P505':metrics(a[240:745],p[240:745])}
        final_fit=fit_model(name,y,745,cfg,cache);unlabelled=future(y.index[-1]+pd.Timedelta(hours=1),745)
    test_frame.to_csv(destination/'test_predictions.csv',index=False,encoding='utf-8-sig')
    final_path=predict(final_fit,y,unlabelled)
    model_info=save_model(final_fit,y,len(unlabelled),destination/'saved_model')
    prediction_frame(name,scenario,'future_unlabelled',0,unlabelled[0],unlabelled,np.repeat(np.nan,len(unlabelled)),final_path).to_csv(destination/'future_path.csv',index=False,encoding='utf-8-sig')
    result={'model':name,'scenario':scenario,'selected_config':cfg,'validation':chosen['validation'],
            'test':test_stats,'saved_model':model_info,'runtime_seconds':time.monotonic()-started,'May_labels_read':False,
            'selection_lock':str(destination/'selection_lock.json'),'input_scope':'Past aggregate target; XGB/Ridge may also use known calendar',
            'path_endpoint_convention':'Day path contains24 values lead1..24; lead24 endpoint is separately reported. Long path contains745 values lead1..745.'}
    write_json(destination/'results.json',result)
    print(json.dumps({'model':name,'scenario':scenario,'selected':cfg,'validation':chosen['validation'],'test':test_stats},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['arima','xgboost','baselines'],required=True)
    p.add_argument('--scenario',choices=['dayahead24','long745'],required=True);p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    if args.prepare_only:
        write_json(OUT/'classical_search_plan.json',{'ARIMA':ARIMA_CONFIGS,'XGBoost':XGB_CONFIGS,'baselines':BASELINE_CONFIGS})
        print('Classical search prepared; no fitting or May-label access')
    else:run_search(args.model,args.scenario)
