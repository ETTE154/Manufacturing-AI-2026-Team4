"""Hourly fixed-origin models with strictly completed H/D/W/M context.

All40 candidates are selected on mean late505 MAE at three historical origins.
May labels are never opened by this training or inference module.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import argparse,calendar as cal,hashlib,json,time,warnings
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
OUT=Path(__file__).resolve().parent;RES=OUT.parent;ROOT=RES.parents[1]
sys.path.insert(0,str(RES))
from resample_common import load_hourly_target,periods
sys.path.insert(0,str(ROOT/'output/optimization_20261013/forecast'))
from opt_common import metrics,prediction_frame,write_json
from classical_opt import origin_features,calendar,FEATURE_NAMES,CALENDAR_NAMES
SEED=20261013;THREADS=4;P=745
threadpool_limits(limits=THREADS)
VALID=pd.to_datetime(['2025-02-15','2025-02-22','2025-02-25'])
TEST=pd.Timestamp('2025-03-29');GROUPS=['H','HD','HDW','HDWM']
D_NAMES=['D_mean7_rate','D_mean28_rate','D_last_minus_mean7_rate','D_support28_fraction','D_missing_flag']
W_NAMES=['W_last_rate','W_mean4_rate','W_support4_fraction','W_missing_flag']
M_NAMES=['M_last_rate','M_mean3_rate','M_last_energy','M_completed_count','M_missing_flag','M_support_lt3_flag','known_partial_month_observed_hours']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def target_sha(y):
    return hashlib.sha256(np.asarray(y.index.to_numpy(dtype='datetime64[ns]')).view(np.int64).tobytes()+y.to_numpy(dtype=np.float64).tobytes()).hexdigest()
def prefix(y,origin):
    h=y.loc[y.index<pd.Timestamp(origin)]
    if len(h)<168 or h.index[-1]+pd.Timedelta(hours=1)!=pd.Timestamp(origin):raise ValueError('Invalid hourly prefix')
    return h
def future(h):return pd.date_range(h.index[-1]+pd.Timedelta(hours=1),periods=P,freq='h')
def stats(a,p):return {'late505':metrics(np.asarray(a)[240:],np.asarray(p)[240:]),'full745':metrics(a,p)}

def completed(h,freq,origin):
    table=periods(h,freq,complete_only=True)
    if len(table):
        if not table.complete.all() or (pd.to_datetime(table.available_at)>origin).any():raise ValueError('Future/incomplete period admitted')
        if not (table.observed_hours==table.expected_hours).all():raise ValueError('Partial period admitted')
    return table

def origin_context(h):
    """Resample only this actual fitting/origin prefix, never the full target."""
    origin=h.index[-1]+pd.Timedelta(hours=1);a=h.to_numpy(dtype=float)
    H=origin_features(a,len(a),origin);fallback=float(a.mean())
    d=completed(h,'D',origin);w=completed(h,'W-SUN',origin);m=completed(h,'MS',origin)
    dr=d.rate_kwh_per_hour.to_numpy(dtype=float);wr=w.rate_kwh_per_hour.to_numpy(dtype=float);mr=m.rate_kwh_per_hour.to_numpy(dtype=float)
    d7=float(dr[-7:].mean()) if len(dr) else fallback;d28=float(dr[-28:].mean()) if len(dr) else fallback
    D=[d7,d28,float(dr[-1]-d7) if len(dr) else 0.,min(len(dr),28)/28.,float(len(dr)==0)]
    W=[float(wr[-1]) if len(wr) else fallback,float(wr[-4:].mean()) if len(wr) else fallback,min(len(wr),4)/4.,float(len(wr)==0)]
    prevmonth=origin.to_period('M')-1;prevhours=prevmonth.days_in_month*24
    partial=int(((h.index.year==origin.year)&(h.index.month==origin.month)).sum())
    M=[float(mr[-1]) if len(mr) else fallback,float(mr[-3:].mean()) if len(mr) else fallback,
       float(m.energy_kwh.iloc[-1]) if len(m) else fallback*prevhours,float(len(m)),float(len(m)==0),float(len(m)<3),float(partial)]
    audit={'origin':str(origin),'history_end':str(h.index[-1]),'history_hours':len(h),'D_complete_count':len(d),'W_complete_count':len(w),
           'M_complete_count':len(m),'monthly_fallback_used':len(m)==0,'monthly_latest_available_at':str(m.available_at.iloc[-1]) if len(m) else None,
           'fallback':'Mean of this strict historical prefix; missing/support flags retained. No global/future imputation.'}
    return H,D,W,M,audit

def feature_names(group):
    return FEATURE_NAMES.copy()+(CALENDAR_NAMES if group!='H_previous_history_only' else [])+(D_NAMES if group in ['HD','HDW','HDWM'] else [])+(W_NAMES if group in ['HDW','HDWM'] else [])+(M_NAMES if group=='HDWM' else [])

def matrix_from_context(context,origin,offsets,group):
    H,D,W,M,_=context;offsets=np.asarray(offsets,dtype=int)
    idx=pd.DatetimeIndex([pd.Timestamp(origin)+pd.Timedelta(hours=int(o)) for o in offsets])
    rows=np.tile(np.asarray(H,dtype=float),(len(offsets),1))
    rows=np.column_stack([rows,offsets/24.])
    if group!='H_previous_history_only':rows=np.column_stack([rows,calendar(idx)])
    if group in ['HD','HDW','HDWM']:rows=np.column_stack([rows,np.tile(D,(len(offsets),1))])
    if group in ['HDW','HDWM']:rows=np.column_stack([rows,np.tile(W,(len(offsets),1))])
    if group=='HDWM':rows=np.column_stack([rows,np.tile(M,(len(offsets),1))])
    rows=rows.astype(np.float64)
    assert rows.shape==(len(offsets),len(feature_names(group))) and np.isfinite(rows).all()
    return rows

def build_train(h,group,context_cache=None):
    offsets=np.array(sorted(set(range(25))|set(range(48,745,24))))
    xs=[];ys=[];a=h.to_numpy(dtype=float)
    for pos in range(168,len(h),6):
        allowed=offsets[pos+offsets<len(h)]
        if not len(allowed):continue
        key=(str(h.index[0]),pos)
        if context_cache is not None and key in context_cache:context=context_cache[key]
        else:
            context=origin_context(h.iloc[:pos])
            if context_cache is not None:context_cache[key]=context
        xs.append(matrix_from_context(context,h.index[pos],allowed,group));ys.append(a[pos+allowed])
    return np.concatenate(xs),np.concatenate(ys)

def configs():
    base=[dict(n_estimators=50,max_depth=2,learning_rate=.01,min_child_weight=100,reg_lambda=100.,
               subsample=.8,colsample_bytree=.8,objective='reg:absoluteerror',residual_shrinkage=.5),
          dict(n_estimators=50,max_depth=1,learning_rate=.01,min_child_weight=5,reg_lambda=10.,
               subsample=1.,colsample_bytree=1.,objective='reg:squarederror',residual_shrinkage=.25),
          dict(n_estimators=200,max_depth=4,learning_rate=.03,min_child_weight=50,reg_lambda=100.,
               subsample=.8,colsample_bytree=.8,objective='reg:squarederror',residual_shrinkage=1.)]
    trials=[dict(model='xgboost',group=g,**b) for g in GROUPS for b in base]
    trials.append(dict(model='xgboost',group='H_previous_history_only',**base[0]))
    trials += [dict(model='ridge',group=g,alpha=1e5) for g in GROUPS]
    trials += [dict(model='arima',order=list(order)) for order in [(0,0,0),(1,0,0),(2,0,0),(0,0,1),(0,0,2),(1,0,1),
                 (2,0,1),(1,0,2),(0,1,0),(1,1,0),(0,1,1),(1,1,1),(2,1,0)]]
    trials += [dict(model='baseline',kind=k) for k in ['train_mean','train_median']]
    trials += [dict(model='baseline',kind=k,window=w) for k in ['recent_mean','ema'] for w in [168,720,1440]]
    trials += [dict(model='baseline',kind='calendar_ridge',alpha=1e5),dict(model='baseline',kind='weekday_hour_shrink',shrinkage=.5)]
    assert len(trials)==40
    return trials

def fit_model(h,cfg,cache=None,contexts=None):
    model=cfg['model'];bundle={'config':cfg,'history_end':str(h.index[-1]),'history_count':len(h),
                              'forecast_context':origin_context(h),'forecast_points':P}
    if model in ['xgboost','ridge']:
        key=(len(h),cfg['group'])
        if cache is not None and key in cache:X,y=cache[key]
        else:
            X,y=build_train(h,cfg['group'],contexts)
            if cache is not None:cache[key]=(X,y)
        center=float(y.mean());bundle['target_center']=center
        if model=='xgboost':
            from xgboost import XGBRegressor
            params={k:v for k,v in cfg.items() if k not in ['model','group','residual_shrinkage']}
            obj=XGBRegressor(**params,tree_method='hist',n_jobs=THREADS,random_state=SEED).fit(X,y-center)
        else:
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import Ridge
            obj=make_pipeline(StandardScaler(),Ridge(alpha=cfg['alpha'],solver='lsqr',tol=1e-8)).fit(X,y-center)
        bundle.update(object=obj,feature_names=feature_names(cfg['group']),training_examples=len(y),
                      train_target_last=str(h.index[-1]),future_sensor_measurements_used=False)
    elif model=='arima':
        from statsmodels.tsa.arima.model import ARIMA
        order=tuple(cfg['order']);center=float(h.mean())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            obj=ARIMA(h.to_numpy(dtype=float)-center,order=order,trend='c' if order[1]==0 else 'n',
                      enforce_stationarity=True,enforce_invertibility=True).fit(method_kwargs={'maxiter':200,'disp':0})
        if not obj.mle_retvals.get('converged',True):raise ValueError('ARIMA did not converge')
        bundle.update(object=obj,target_center=center,aic_training_only=float(obj.aic),warnings=[str(w.message) for w in caught])
    else:
        bundle.update(mean=float(h.mean()),median=float(h.median()))
        if cfg['kind']=='calendar_ridge':
            from sklearn.linear_model import Ridge
            bundle['object']=Ridge(alpha=cfg['alpha']).fit(calendar(h.index),h.to_numpy(dtype=float)-bundle['mean'])
        elif cfg['kind']=='weekday_hour_shrink':
            table=pd.DataFrame({'value':h.to_numpy(dtype=float),'dow':h.index.dayofweek,'hour':h.index.hour}).groupby(['dow','hour']).value.mean()
            bundle['weekday_hour_means']={f'{dow}:{hour}':float(v) for (dow,hour),v in table.items()}
        elif cfg['kind']=='ema':bundle['frozen_level']=float(h.ewm(span=cfg['window'],adjust=False).mean().iloc[-1])
        elif cfg['kind']=='recent_mean':bundle['frozen_level']=float(h.iloc[-cfg['window']:].mean())
    return bundle

def predict(bundle):
    cfg=bundle['config'];idx=pd.date_range(pd.Timestamp(bundle['history_end'])+pd.Timedelta(hours=1),periods=P,freq='h')
    if cfg['model'] in ['xgboost','ridge']:
        X=matrix_from_context(bundle['forecast_context'],idx[0],np.arange(P),cfg['group'])
        return idx,bundle['target_center']+cfg.get('residual_shrinkage',1.)*np.asarray(bundle['object'].predict(X),dtype=np.float64)
    if cfg['model']=='arima':return idx,bundle['target_center']+np.asarray(bundle['object'].forecast(P),dtype=np.float64)
    kind=cfg['kind']
    if kind=='calendar_ridge':p=bundle['mean']+np.asarray(bundle['object'].predict(calendar(idx)),dtype=np.float64)
    elif kind=='weekday_hour_shrink':
        raw=np.array([bundle['weekday_hour_means'].get(f'{t.dayofweek}:{t.hour}',bundle['mean']) for t in idx])
        p=bundle['mean']+cfg['shrinkage']*(raw-bundle['mean'])
    else:p=np.repeat(bundle['mean'] if kind=='train_mean' else bundle['median'] if kind=='train_median' else bundle['frozen_level'],P)
    return idx,np.asarray(p,dtype=np.float64)

def save_model(bundle,dest):
    dest.mkdir(parents=True,exist_ok=True);idx,p=predict(bundle)
    joblib.dump(bundle,dest/'model.joblib',compress=3);loaded=joblib.load(dest/'model.joblib');_,again=predict(loaded)
    error=float(np.max(np.abs(p-again)));assert error<=1e-8
    kind=bundle['config']['model'];native_error=None
    if kind=='xgboost':
        from xgboost import XGBRegressor
        bundle['object'].save_model(dest/'model_native.json');native=XGBRegressor();native.load_model(dest/'model_native.json');loaded['object']=native
        native_error=float(np.max(np.abs(p-predict(loaded)[1])));assert native_error<=1e-8
    elif kind=='arima':bundle['object'].save(dest/'model_statsmodels.pkl')
    info={'model_path':str(dest/'model.joblib'),'model_sha256':sha(dest/'model.joblib'),'config':bundle['config'],
          'history_end':bundle['history_end'],'history_count':bundle['history_count'],'forecast_points':P,
          'feature_names':bundle.get('feature_names'),'target_center':bundle.get('target_center'),'training_examples':bundle.get('training_examples'),
          'forecast_context_audit':bundle['forecast_context'][-1],'reload_max_absdiff_kwh':error,'native_reload_max_absdiff_kwh':native_error,
          'scaler':'Ridge StandardScaler fit on historical direct examples only; XGB/ARIMA none.',
          'standalone_inference':'Saved feature context/state forecasts exactly the frozen Apr30 origin; no target file or May read is needed.'}
    write_json(dest/'metadata.json',info)
    return idx,p,info

def audit_temporal(y):
    origin=pd.Timestamp('2025-02-15');h=prefix(y,origin);context=origin_context(h)
    poisoned=y.copy();poisoned.loc[poisoned.index>=origin]=-999999.
    same=origin_context(prefix(poisoned,origin))
    checks=[]
    for group in GROUPS+['H_previous_history_only']:
        X=matrix_from_context(context,origin,np.arange(P),group);Xp=matrix_from_context(same,origin,np.arange(P),group)
        assert np.array_equal(X,Xp)
        checks.append({'group':group,'future_poison_max_absdiff':0.,'features':len(feature_names(group))})
    for t in pd.to_datetime(['2024-12-08','2025-01-01','2025-02-01']):
        c=origin_context(prefix(y,t));assert c[-1]['monthly_fallback_used']==(t<pd.Timestamp('2025-01-01'))
    write_json(OUT/'temporal_audit.json',{'status':'passed','checks':checks,'source_step_seconds':5,'model_step_hours':1,
       'available_at_rule':'Completed hour/day/week/month has endexclusive<=forecast origin; partial boundary periods are excluded.',
       'training_label_rule':'Historical direct origins use only pos+offset<len(fitting_prefix); no forecast label crosses prefix end.',
       'fallback':'Early missing complete month/week replaced with this prefix mean and explicitmissing/support flags.',
       'global_resampling_used_for_predictor_fit':False,'future_period_measurements_used':False})

def search():
    OUT.mkdir(parents=True,exist_ok=True);y=load_hourly_target();audit_temporal(y);trials=configs();started=time.monotonic()
    assert VALID.max()+pd.Timedelta(hours=P-1)<TEST
    write_json(OUT/'search_plan.json',{'candidate_count':40,'candidates':trials,'validation_origins':[str(t) for t in VALID],
        'criterion':'Mean of three origin late505 MAE values, offsets240..744; full745 is ancillary.',
        'overlap_limit':'The three validation target spans overlap;1515 scored pairs do not represent1515 independent targets or independent origins.',
        'group_ablation':GROUPS,'monthly_context_limit':'Only two complete months at each validation origin and four at Apr30; weak support, no monthly model learned.',
        'future_period_policy':'No future D/W/M actuals. All resolution context is frozen at hourly origin.',
        'prior_candidate':'Existing validation-selected long XGB exact history-only configuration included; no May-based choice.',
        'source_policy':'Preserve all prior artifacts; new files only in this hourly folder.'})
    records=[];best={};cache={};contexts={}
    for candidate,cfg in enumerate(trials,1):
        try:
            frames=[];scores=[]
            for origin_index,origin in enumerate(VALID,1):
                h=prefix(y,origin);bundle=fit_model(h,cfg,cache,contexts);idx,p=predict(bundle);a=y.reindex(idx).to_numpy(dtype=float)
                if not np.isfinite(a).all():raise ValueError('Validation label beyond source')
                frames.append(prediction_frame(cfg['model'],'hourly_multires745','validation',origin_index,origin,idx,a,p));scores.append({'origin':str(origin),**stats(a,p)})
            objective=float(np.mean([s['late505']['mae_kwh'] for s in scores]));full=float(np.mean([s['full745']['mae_kwh'] for s in scores]))
            record={'candidate':candidate,'config':cfg,'validation_late505_mean_mae_kwh':objective,'validation_full745_mean_mae_kwh':full,'by_origin':scores}
            family=cfg['model'];f=pd.concat(frames,ignore_index=True)
            if family not in best or objective<best[family]['validation_late505_mean_mae_kwh']:
                best[family]=record;dest=OUT/family;dest.mkdir(parents=True,exist_ok=True)
                f.to_csv(dest/'validation_predictions.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
        except Exception as exc:record={'candidate':candidate,'config':cfg,'error':repr(exc)}
        records.append(record);write_json(OUT/'search_progress.json',{'records':records,'best_by_model':best,'elapsed_seconds':time.monotonic()-started})
        print(f'hourly candidate{candidate}/40 {cfg["model"]}/{cfg.get("group",cfg.get("order",cfg.get("kind")))} {record.get("validation_late505_mean_mae_kwh",record.get("error"))} elapsed{time.monotonic()-started:.1f}s',flush=True)
    if len(best)!=4:raise ValueError('Model family failed all candidates')
    selected=min(best,key=lambda k:best[k]['validation_late505_mean_mae_kwh'])
    lock={'status':'frozen_before_new_internal_test_and_May','selected_model':selected,'models':best,'records':records,
          'criterion':'Three-origin average late505 MAE only','May_used_for_selection':False,'internal_test_used_for_selection':False,
          'common_sha256':sha(RES/'resample_common.py'),'code_sha256':sha(__file__),'seed':SEED,
          'hourly_target_fingerprint_sha256':target_sha(y),
          'data_hours':len(y),'data_start':str(y.index[0]),'data_end':str(y.index[-1]),
          'validation_overlap_declared':True,'post_EDA':'Previous work has seen test/May; this is locked post-EDA experimentation.'}
    write_json(OUT/'selection_lock.json',lock)
    print('Hourly40 candidates complete; late505 selection frozen. No May access.',flush=True)

def evaluate():
    y=load_hourly_target();lock=json.loads((OUT/'selection_lock.json').read_text(encoding='utf-8'))
    if lock['common_sha256']!=sha(RES/'resample_common.py'):raise ValueError('Common API changed after selection')
    if lock['hourly_target_fingerprint_sha256']!=target_sha(y):raise ValueError('Hourly target changed after selection')
    models={};cache={};contexts={};ablations=[]
    for r in lock['records']:
        ablations.append({'candidate':r['candidate'],'model':r['config']['model'],'group':r['config'].get('group'),
                          'validation_late505_mae_kwh':r.get('validation_late505_mean_mae_kwh'),
                          'validation_full745_mae_kwh':r.get('validation_full745_mean_mae_kwh'),'error':r.get('error')})
    pd.DataFrame(ablations).to_csv(OUT/'ablation.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    for name,chosen in lock['models'].items():
        cfg=chosen['config'];dest=OUT/name;h=prefix(y,TEST);bundle=fit_model(h,cfg,cache,contexts);idx,p=predict(bundle);a=y.reindex(idx).to_numpy(dtype=float)
        prediction_frame(name,'hourly_multires745','internal_test',0,TEST,idx,a,p).to_csv(dest/'test_predictions.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
        internal=stats(a,p);final=fit_model(y,cfg,cache,contexts);fidx,fp,info=save_model(final,dest/'saved_model')
        prediction_frame(name,'hourly_multires745','future_unlabelled',0,fidx[0],fidx,np.full(P,np.nan),fp).to_csv(dest/'future_path.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
        model={'selected_config':cfg,'selected_candidate':chosen['candidate'],'trials_count':sum(r['config']['model']==name for r in lock['records']),
               'validation_late505':{'mean_origin_mae_kwh':chosen['validation_late505_mean_mae_kwh'],'by_origin':[{'origin':s['origin'],**s['late505']} for s in chosen['by_origin']]},
               'validation_full745':{'mean_origin_mae_kwh':chosen['validation_full745_mean_mae_kwh'],'by_origin':[{'origin':s['origin'],**s['full745']} for s in chosen['by_origin']]},
               'internal_test':internal,'saved_model':info,'paths':{'validation_csv':str(dest/'validation_predictions.csv'),'test_csv':str(dest/'test_predictions.csv'),'future_csv':str(dest/'future_path.csv')},
               'input_scope':'Past hourly target, deterministic future calendar/horizon; optionally completed D/W/M rates. No physical sensor or future period actuals.',
               'May_labels_read':False,'monthly_small_sample_limit':'Only2 complete monthly observations per validation origin,4 before final origin; M features are contextual candidates, no monthly model fit.'}
        if name in ['ridge','xgboost']:
            table=pd.DataFrame({'feature':final['feature_names']})
            if name=='ridge':table['coefficient_per_training_std']=final['object'][-1].coef_;table['coefficient_per_raw_unit']=final['object'][-1].coef_/final['object'][0].scale_
            else:table['training_gain']=final['object'].feature_importances_
            table.to_csv(dest/'final_feature_importance.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
            model['training_importance_csv']=str(dest/'final_feature_importance.csv')
        write_json(dest/'results.json',model);models[name]=model
        print(f'{name} validlate{chosen["validation_late505_mean_mae_kwh"]:.6f} internallate{internal["late505"]["mae_kwh"]:.6f}',flush=True)
    result={'models':models,'selected_model':lock['selected_model'],'selection_scope':'These40 new hourly candidates; parent compares all resampling families after global lock.',
            'selection_lock':str(OUT/'selection_lock.json'),'selection_lock_sha256':sha(OUT/'selection_lock.json'),
            'selection_criterion':'Mean late505 MAE at Feb15/22/25 origins; complete745 is ancillary.','validation_origin_count':3,
            'validation_scored_pairs':1515,'validation_overlap':True,'internal_test_origin':str(TEST),'final_origin':'2025-04-30 00:00:00',
            'future_points':P,'May_labels_read':False,'May_score_status':'awaiting_parent_global_lock',
            'ablation_csv':str(OUT/'ablation.csv'),'temporal_audit':str(OUT/'temporal_audit.json'),
            'limitations':['Overlapping validation periods limit independent evidence.','Completed-month context has only2~4 samples.',
              'Forecast origin stays fixed; no actual target or period update inside745future hours.','13-module energy sum remains provisional with unknown hierarchy/units.',
              'Post-EDA experiment; previous evaluation already saw April/May.','Additional resampling features can harm generalization.']}
    write_json(OUT/'results.json',result)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['search','evaluate'],required=True);args=parser.parse_args()
    search() if args.stage=='search' else evaluate()
