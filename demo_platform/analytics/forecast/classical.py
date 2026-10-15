"""Native-hourly505 CV search of residual-normalized statistical/ML forecasts."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import argparse,time,warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from common import *
threadpool_limits(limits=THREADS)

def configs():
    cs=[]
    for kind in ['mean','median','trimmed_mean','ema','recent_mean']:
        for window in ([168,720] if kind in ['ema','recent_mean'] else ['full']):
            cs.append({'family':'baseline','kind':kind,'window':window,'resolution':'hourly'})
    for resolution in ['hourly','daily']:
        for scaler in ['zscore','robust','minmax']:
            for order in [(0,0,0),(1,0,0),(2,0,0),(1,0,1),(0,1,0),(0,1,1)]:
                cs.append({'family':'arima','resolution':resolution,'scaler':scaler,'order':list(order)})
            for alpha in [1.,100.,10000.,1000000.]:
                for intercept in [False,True]:cs.append({'family':'ridge','resolution':resolution,'scaler':scaler,'alpha':alpha,'intercept':intercept})
            for depth in [1,2]:cs.append({'family':'xgboost','resolution':resolution,'scaler':scaler,'depth':depth,'max_estimators_protection':10000})
    return cs

def feature_rows(s,pos,points,n,resolution):
    v=transform(s.iloc[:pos].to_numpy(),n)-zero_prior(n)[0]
    short,long=(24,168) if resolution=='hourly' else (1,7)
    a=v[-long:];base=[v[-1],v[-short],v[-long],v[-short:].mean(),a.mean(),a.std(),(a[-1]-a[0])/max(len(a)-1,1)]
    times=pd.date_range(s.index[pos-1]+pd.Timedelta(hours=1) if resolution=='hourly' else s.index[pos-1]+pd.Timedelta(days=1),
                         periods=points,freq='h' if resolution=='hourly' else 'D')
    X=np.asarray([base+[lead/max(points-1,1),np.sin(2*np.pi*t.hour/24),np.cos(2*np.pi*t.hour/24),
                       np.sin(2*np.pi*t.dayofweek/7),np.cos(2*np.pi*t.dayofweek/7)] for lead,t in enumerate(times)],float)
    return X
def train_xy(s,cfg,n):
    context=168 if cfg['resolution']=='hourly' else 7;points=745 if cfg['resolution']=='hourly' else 32
    last=len(s)-points;positions=list(range(context,last+1,24 if cfg['resolution']=='hourly' else 1))
    if last>=context and (not positions or positions[-1]!=last):positions.append(last)
    if not positions:raise ValueError('Insufficient past labels for supervised path')
    X=np.vstack([feature_rows(s,pos,points,n,cfg['resolution']) for pos in positions])
    target=np.concatenate([transform(s.iloc[pos:pos+points].to_numpy(),n)-zero_prior(n)[0] for pos in positions])
    return X,target,len(positions)
def fit(h,cfg,iterations=None):
    s=native(h,cfg['resolution']);bundle={'config':cfg,'history_end':str(h.index[-1]),'history_values':h.to_list(),'history_times':[str(t) for t in h.index]}
    family=cfg['family']
    if family=='baseline':return bundle
    n=norm_fit(s.values,cfg['scaler']);n['fit_end']=str(s.index[-1]);bundle['normalizer']=n
    if family=='arima':
        from statsmodels.tsa.arima.model import ARIMA
        order=tuple(cfg['order']);z=transform(s.values,n)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always');m=ARIMA(z,order=order,trend='c' if order[1]==0 else 'n').fit(method_kwargs={'maxiter':1000})
        if not m.mle_retvals.get('converged',True):raise ValueError('ARIMA optimizer did not converge')
        bundle['model']=m;bundle['warnings']=[str(w.message) for w in ws]
    elif family=='ridge':
        X,target,windows=train_xy(s,cfg,n);bundle['model']=Ridge(alpha=cfg['alpha'],fit_intercept=cfg['intercept']).fit(X,target)
        bundle['training_windows']=windows;bundle['supervised_examples']=len(target)
    elif family=='xgboost':
        from xgboost import XGBRegressor
        X,target,windows=train_xy(s,cfg,n)
        m=XGBRegressor(n_estimators=int(iterations),max_depth=cfg['depth'],learning_rate=.03,min_child_weight=30,
                       reg_lambda=100.,objective='reg:squarederror',tree_method='hist',n_jobs=THREADS,random_state=SEED)
        bundle['model']=m.fit(X,target);bundle['training_windows']=windows;bundle['supervised_examples']=len(target);bundle['iterations']=iterations
    return bundle
def predict(bundle,h=None):
    cfg=bundle['config'];h=h if h is not None else pd.Series(bundle['history_values'],index=pd.to_datetime(bundle['history_times']))
    s=native(h,cfg['resolution']);points=745 if cfg['resolution']=='hourly' else 32
    if cfg['family']=='baseline':
        a=h.to_numpy();window=cfg['window'];v=a if window=='full' else a[-int(window):]
        if cfg['kind']=='mean':level=a.mean()
        elif cfg['kind']=='median':level=np.median(a)
        elif cfg['kind']=='trimmed_mean':
            q=np.quantile(a,[.05,.95]);level=a[(a>=q[0])&(a<=q[1])].mean()
        elif cfg['kind']=='ema':level=h.ewm(span=int(window),adjust=False).mean().iloc[-1]
        else:level=v.mean()
        return np.repeat(float(level),745)
    n=bundle['normalizer']
    if cfg['family']=='arima':z=np.asarray(bundle['model'].forecast(steps=points),float)
    else:z=bundle['model'].predict(feature_rows(s,len(s),points,n,cfg['resolution']))+zero_prior(n)[0]
    return hourly_map(inverse(z,n),cfg['resolution'])
def inner_iterations(h,cfg):
    from xgboost import XGBRegressor
    s=native(h,cfg['resolution']);hold=745 if cfg['resolution']=='hourly' else 32;fit_s=s.iloc[:-hold];tail=s.iloc[-hold:]
    n=norm_fit(fit_s.values,cfg['scaler']);X,target,windows=train_xy(fit_s,cfg,n)
    nativeX=feature_rows(fit_s,len(fit_s),hold,n,cfg['resolution'])
    evalX=nativeX[240:] if cfg['resolution']=='hourly' else np.repeat(nativeX,24,axis=0)[240:745]
    actual=h.reindex(pd.date_range(tail.index[0],periods=745,freq='h')).to_numpy(float)[240:]
    assert np.isfinite(actual).all();evalY=transform(actual,n)-zero_prior(n)[0]
    m=XGBRegressor(n_estimators=10000,max_depth=cfg['depth'],learning_rate=.03,min_child_weight=30,reg_lambda=100.,
        objective='reg:squarederror',eval_metric='mae',tree_method='hist',n_jobs=THREADS,random_state=SEED,early_stopping_rounds=75)
    m.fit(X,target,eval_set=[(evalX,evalY)],verbose=False)
    return m.best_iteration+1,{'fit_native_rows':len(fit_s),'fit_end':str(fit_s.index[-1]),'holdout_periods':hold,
            'best_iteration':int(m.best_iteration+1),'epochs_evaluated':len(m.evals_result()['validation_0']['mae']),
            'early_stopping_objective':'Native-hourly Late505 MAE, affine-standardized positive scale, same ordering',
            'normalizer':n,'training_windows':windows,'test_or_May_used':False}

def main():
    assert_training_permitted()
    y=load_y();write(OUT/'normalization_audit.json',audit_units());cs=configs()
    write(OUT/'classical_search_plan.json',{'configs':cs,'seed':SEED,'selection':'Three origin mean Late505 only','validation_origins':VALID,
          'unit_rescaling':False,'native_units_retained':True})
    records=[];paths={};start=time.perf_counter()
    for ci,cfg in enumerate(cs):
        fs=[];audits=[]
        try:
            for fold,origin in enumerate(VALID,1):
                h=prefix(y,origin)
                iterations,early=inner_iterations(h,cfg) if cfg['family']=='xgboost' else (None,None)
                b=fit(h,cfg,iterations);p=predict(b);fs.append(frame(cfg['family'],'validation',fold,origin,y,p))
                audits.append({'origin':str(origin),'history_end':str(h.index[-1]),'early_stopping':early,'normalizer':b.get('normalizer')})
            f=pd.concat(fs,ignore_index=True);stats=summary(f);records.append({'candidate':ci,'config':cfg,'validation':stats,
                 'objective_mae_kwh':stats['mean_origin_late505_mae_kwh'],'origin_fit_audits':audits,'status':'completed'})
            paths[ci]=f
        except Exception as e:records.append({'candidate':ci,'config':cfg,'status':'failed','reason':str(e)})
        write(OUT/'classical_search_progress.json',{'completed_candidates':ci+1,'total_candidates':len(cs),'last':records[-1],'seconds':time.perf_counter()-start})
        if ci%10==0:print(f'classical {ci+1}/{len(cs)} {cfg["family"]} {time.perf_counter()-start:.1f}s',flush=True)
    models={}
    for family in ['baseline','arima','ridge','xgboost']:
        done=[r for r in records if r['config']['family']==family and r['status']=='completed'];chosen=min(done,key=lambda r:r['objective_mae_kwh'])
        cfg=chosen['config'];dest=OUT/'classical'/family;dest.mkdir(parents=True,exist_ok=True)
        write(dest/'selection_lock.json',{'chosen':chosen,'selection_uses_May':False,'selection_uses_internal_test':False,'locked_before_test':True})
        paths[chosen['candidate']].to_csv(dest/'validation_predictions.csv',index=False,encoding='utf-8-sig')
        fitted={};replays=[];test_stats=None
        final_iterations=int(np.median([a['early_stopping']['best_iteration'] for a in chosen['origin_fit_audits']])) if family=='xgboost' else None
        for split,origin in [('internal_test',TEST),('future_unlabelled',FINAL)]:
            h=prefix(y,origin);iterations,early=inner_iterations(h,cfg) if family=='xgboost' and split=='internal_test' else (final_iterations,None)
            b=fit(h,cfg,iterations);p=predict(b);path=dest/f'{split}.joblib';joblib.dump(b,path);again=predict(joblib.load(path))
            error=float(abs(p-again).max());assert error==0
            f=frame(family,split,0,origin,y,p);csv=dest/('test_predictions.csv' if split=='internal_test' else 'future_path.csv');f.to_csv(csv,index=False,encoding='utf-8-sig')
            fitted[split]={'saved_model':str(path),'prediction_csv':str(csv),'history_end':str(h.index[-1]),'inner_early_stopping':early}
            replays.append({'split':split,'max_abs_reload_error':error,'history_only_before_origin':True})
            if split=='internal_test':test_stats=summary(f)
        models[family]={'family':family,'selected_config':cfg,'validation':chosen['validation'],'internal_test':test_stats,'fits':fitted,
                         'reload':replays,'selection_lock':str(dest/'selection_lock.json'),'validation_prediction_csv':str(dest/'validation_predictions.csv'),
                         'candidate_count':len([r for r in records if r['config']['family']==family]),'May_read':False}
    write(OUT/'classical_results.json',{'status':'complete','models':models,'candidates':records,'seconds':time.perf_counter()-start,'May_read':False})
    print({k:(r['validation']['mean_origin_late505_mae_kwh'],r['internal_test']['late505']['mae_kwh']) for k,r in models.items()},flush=True)
if __name__=='__main__':main()
