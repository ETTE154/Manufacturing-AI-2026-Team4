"""Daily/weekly/monthly forecast candidates mapped to the guide hourly path.

All fitting/profile statistics are from the forecast origin's prefix. No May
labels are read here. Monthly models are explicitly sparse-data baselines.
"""
from pathlib import Path
import sys,hashlib,json,time,warnings
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
OUT=Path(__file__).resolve().parent;EXP=OUT.parent;ROOT=EXP.parents[1]
SEED=20261013;threadpool_limits(limits=4)
VALID=pd.to_datetime(['2025-02-15','2025-02-22','2025-02-25'])
TEST=pd.Timestamp('2025-03-29');FINAL=pd.Timestamp('2025-04-30')
FREQ={'daily':'D','weekly':'W-SUN','monthly':'MS'}
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(Path,pd.Timestamp)):return str(x)
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.bool_,)):return bool(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x
def write(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_y():
    f=pd.read_csv(ROOT/'output/analysis/baseline_audit/complete_hour_module_sum.csv',float_precision='round_trip')
    y=pd.Series(f.iloc[:,1].to_numpy(float),index=pd.to_datetime(f.iloc[:,0]))
    assert len(y)==3600 and y.index.equals(pd.date_range('2024-12-01','2025-04-29 23:00',freq='h'))
    return y
def prefix(y,origin):
    h=y[y.index<pd.Timestamp(origin)];assert h.index[-1]+pd.Timedelta(hours=1)==pd.Timestamp(origin)
    return h
def periods(h,resolution):
    g=h.resample(FREQ[resolution]).agg(['sum','mean','count'])
    start=g.index if resolution!='weekly' else g.index-pd.Timedelta(days=6)
    end=start+pd.Timedelta(days=1 if resolution=='daily' else 7) if resolution!='monthly' else start+pd.offsets.MonthBegin(1)
    expected=((end-start)/pd.Timedelta(hours=1)).to_numpy(int)
    good=(g['count'].to_numpy()==expected)&(end<=h.index[-1]+pd.Timedelta(hours=1))
    r=pd.Series(g.loc[good,'mean'].to_numpy(float),index=g.index[good])
    assert len(r)>0
    return r,{'complete_periods':len(r),'last_period_label':str(r.index[-1]),'last_period_available_at':str(end[good][-1]),
        'period_target':'Hourly-average energy rate; period sum uses known calendar duration',
        'history_end':str(h.index[-1]),'partial_periods_excluded':int((~good).sum())}
def labels(index,resolution):
    index=pd.DatetimeIndex(index)
    if resolution=='daily':return index.normalize()
    if resolution=='weekly':return index.normalize()+pd.to_timedelta(6-index.dayofweek,unit='D')
    return index.to_period('M').to_timestamp()
def profiles(h):
    # Per-day relative hourly shares. Only fully observed days are admitted.
    df=pd.DataFrame({'y':h.values,'day':h.index.normalize(),'hour':h.index.hour,'dow':h.index.dayofweek},index=h.index)
    df.index.name='observed_time'
    size=df.groupby('day').size();df=df[df.day.isin(size[size==24].index)].copy()
    daily_total=df.groupby('day').y.transform('sum');df['relative']=24*df.y/daily_total
    hour=df.groupby('hour').relative.mean().reindex(range(24)).to_numpy(dtype=float,copy=True);hour/=hour.mean()
    dow=np.tile(hour,(7,1))
    for d in range(7):
        a=df[df.dow==d].groupby('hour').relative.mean().reindex(range(24)).to_numpy(float)
        if np.isfinite(a).all():dow[d]=a/a.mean()
    assert np.isfinite(hour).all() and np.isfinite(dow).all()
    return {'hour':hour,'weekday_hour':dow,'fit_end':str(h.index[-1]),'complete_days':df.day.nunique()}
def multiplier(bundle,index,cfg):
    idx=pd.DatetimeIndex(index)
    if cfg['profile']=='uniform':return np.ones(len(idx))
    base=bundle['profile']['hour'][idx.hour] if cfg['profile']=='hour' else bundle['profile']['weekday_hour'][idx.dayofweek,idx.hour]
    return 1+cfg.get('profile_alpha',.5)*(base-1)
def metric(a,p):
    a,p=np.asarray(a,float),np.asarray(p,float);e=p-a;den=((a-a.mean())**2).sum()
    return {'count':len(a),'mae_kwh':float(np.abs(e).mean()),'rmse_kwh':float(np.sqrt((e*e).mean())),
        'mape_percent':float((np.abs(e)/np.abs(a)).mean()*100),'r2':float(1-(e*e).sum()/den) if den>0 else None,
        'bias_pred_minus_actual_kwh':float(e.mean())}
def frame(name,split,fold,origin,y,p):
    idx=pd.date_range(origin,periods=745,freq='h');a=y.reindex(idx).to_numpy(float)
    return pd.DataFrame({'id':idx,'model':name,'split':split,'fold':fold,'origin':origin,'horizon_offset_hours':np.arange(745),
        'lead_hours':np.arange(1,746),'actual_kwh':a,'predicted_kwh':p,'absolute_error_kwh':np.abs(a-p)})
def summary(f):
    late=f[f.horizon_offset_hours>=240]
    return {'full745':metric(f.actual_kwh,f.predicted_kwh),'late505':metric(late.actual_kwh,late.predicted_kwh),
        'origin_count':f.origin.nunique(),'mean_origin_late505_mae_kwh':float(late.groupby('origin').absolute_error_kwh.mean().mean())}
def train_matrix(s,cfg):
    vals=s.to_numpy(float);rows=[];ys=[]
    for pos in range(28,len(vals)):
        b=[vals[pos-1],vals[pos-7],vals[pos-7:pos].mean(),vals[pos-28:pos].mean(),vals[pos-7:pos].std(),
           (vals[pos-7:pos][-1]-vals[pos-7:pos][0])/6]
        for lead in range(min(32,len(vals)-pos)):
            t=s.index[pos+lead]
            rows.append(b+[lead,np.sin(2*np.pi*t.dayofweek/7),np.cos(2*np.pi*t.dayofweek/7),t.month])
            ys.append(vals[pos+lead])
    return np.asarray(rows,float),np.asarray(ys,float)
def fit(h,cfg):
    s,info=periods(h,cfg['resolution']);bundle={'config':cfg,'period_series':s,'profile':profiles(h),'period_metadata':info,
        'history_end':str(h.index[-1]),'history_mean':float(h.mean()),'center':float(s.mean())}
    kind=cfg['kind']
    if kind=='arima':
        from statsmodels.tsa.arima.model import ARIMA
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always');m=ARIMA(s.to_numpy(),order=tuple(cfg['order']),trend='c' if cfg['order'][1]==0 else 'n').fit(method_kwargs={'maxiter':150})
        if not m.mle_retvals.get('converged',True):raise ValueError('ARIMA convergence failed')
        bundle['model']=m;bundle['training_warnings']=[str(w.message) for w in ws]
    elif kind in ['xgboost','ridge']:
        X,y=train_matrix(s,cfg)
        if len(y)==0:raise ValueError('No daily supervised examples')
        if kind=='xgboost':
            from xgboost import XGBRegressor
            m=XGBRegressor(n_estimators=cfg['trees'],max_depth=cfg['depth'],learning_rate=.03,min_child_weight=20,
                reg_lambda=cfg['lambda'],objective=cfg.get('loss','reg:squarederror'),tree_method='hist',n_jobs=4,random_state=SEED).fit(X,y-bundle['center'])
        else:
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import Ridge
            m=make_pipeline(StandardScaler(),Ridge(alpha=cfg['alpha'])).fit(X,y-bundle['center'])
        bundle['model']=m;bundle['training_examples']=len(y);bundle['unique_observed_periods']=len(s)
    return bundle
def forecast(bundle,points=745):
    cfg=bundle['config'];resolution=cfg['resolution'];origin=pd.Timestamp(bundle['history_end'])+pd.Timedelta(hours=1)
    idx=pd.date_range(origin,periods=points,freq='h');lab=labels(idx,resolution);wanted=pd.DatetimeIndex(lab.unique())
    s=bundle['period_series'];last=s.index[-1]
    future_index=pd.date_range(last,periods=1+len(wanted)+2,freq=FREQ[resolution])[1:]
    if not wanted.isin(future_index).all():raise ValueError('Future period mapping exceeds available rate forecast')
    n=int(future_index.get_indexer(wanted).max()+1);fi=future_index[:n]
    kind=cfg['kind'];a=s.to_numpy(float)
    if kind=='arima':rate=np.asarray(bundle['model'].forecast(steps=n),float)
    elif kind in ['xgboost','ridge']:
        b=[a[-1],a[-7],a[-7:].mean(),a[-28:].mean(),a[-7:].std(),(a[-1]-a[-7])/6]
        X=np.asarray([b+[lead,np.sin(2*np.pi*t.dayofweek/7),np.cos(2*np.pi*t.dayofweek/7),t.month] for lead,t in enumerate(fi)],float)
        rate=bundle['center']+cfg.get('shrink',1)*np.asarray(bundle['model'].predict(X),float)
    elif kind=='mean':rate=np.repeat(a.mean(),n)
    elif kind=='median':rate=np.repeat(np.median(a),n)
    elif kind=='recent':rate=np.repeat(a[-cfg['window']:].mean(),n)
    elif kind=='prefix_mean':rate=np.repeat(bundle['history_mean'],n)
    else:raise ValueError(kind)
    hourly=pd.Series(rate,index=fi).reindex(lab).to_numpy(float)*multiplier(bundle,idx,cfg)
    assert len(hourly)==points and np.isfinite(hourly).all()
    return idx,hourly,pd.DataFrame({'period_label':fi,'forecast_rate_kwh_per_hour':rate})
def configs():
    result=[]
    for resolution in ['daily','weekly','monthly']:
        basics=[{'kind':'mean'},{'kind':'median'},{'kind':'prefix_mean'}]
        basics += [{'kind':'recent','window':w} for w in ([1,7,28,56] if resolution=='daily' else [1,4] if resolution=='weekly' else [1])]
        for base in basics:
            for profile in ['uniform','hour','weekday_hour']:
                result.append({'family':resolution+'_level','resolution':resolution,'profile':profile,'profile_alpha':.5,**base})
        if resolution!='monthly':
            for order in [(0,0,0),(1,0,0),(0,1,0),(0,1,1)]:
                for profile in ['uniform','hour']:
                    result.append({'family':resolution+'_arima','resolution':resolution,'kind':'arima','order':list(order),'profile':profile,'profile_alpha':.5})
    for depth,trees,lam,loss in [(1,50,100.,'reg:squarederror'),(2,100,100.,'reg:squarederror'),(1,100,100.,'reg:absoluteerror')]:
        for profile in ['uniform','hour']:
            result.append({'family':'daily_xgboost','resolution':'daily','kind':'xgboost','depth':depth,'trees':trees,'lambda':lam,'loss':loss,'shrink':.5,'profile':profile,'profile_alpha':.5})
    for alpha in [100.,10000.,1000000.]:
        for profile in ['uniform','hour']:result.append({'family':'daily_ridge','resolution':'daily','kind':'ridge','alpha':alpha,'profile':profile,'profile_alpha':.5})
    return result
def main():
    OUT.mkdir(parents=True,exist_ok=True);y=load_y();records=[];frames={};started=time.perf_counter()
    candidates=configs();write(OUT/'search_plan.json',{'configs':candidates,'selection':'Mean validation505 MAE only','validation_origins':list(VALID),'monthly_warning':'Only2 completed months per validation fitting prefix; level baselines only'})
    for ci,cfg in enumerate(candidates):
        try:
            fs=[];metadata=[]
            for fi,origin in enumerate(VALID,1):
                h=prefix(y,origin);bundle=fit(h,cfg);idx,p,_=forecast(bundle)
                fs.append(frame(cfg['family'],'validation',fi,origin,y,p));metadata.append(bundle['period_metadata'])
            f=pd.concat(fs,ignore_index=True);score=summary(f);records.append({'candidate':ci,'config':cfg,'validation':score,'period_metadata_by_fold':metadata,'objective':score['mean_origin_late505_mae_kwh']});frames[ci]=f
        except Exception as e:records.append({'candidate':ci,'config':cfg,'error':repr(e)})
        if ci%10==0:print('COARSE_CANDIDATE',ci+1,'/',len(candidates),flush=True)
        write(OUT/'search_progress.json',{'records':records,'elapsed_seconds':time.perf_counter()-started})
    winners={}
    for record in records:
        if 'objective' not in record:continue
        family=record['config']['family']
        if family not in winners or record['objective']<winners[family]['objective']:winners[family]=record
    assert len(winners)==7
    lock={'selection':'Three validation origins,505 late-hour path mean MAE','models':winners,'May_labels_read':False,'test_used_to_select':False,
        'code_sha256':sha(__file__),'target_sha256':sha(ROOT/'output/analysis/baseline_audit/complete_hour_module_sum.csv'),'validation_target_overlap':True}
    write(OUT/'selection_lock.json',lock);results={}
    for family,chosen in winners.items():
        d=OUT/family;d.mkdir(exist_ok=True);cfg=chosen['config'];v=frames[chosen['candidate']];v.to_csv(d/'validation_predictions.csv',index=False,float_format='%.17g')
        h=prefix(y,TEST);bundle=fit(h,cfg);_,p,_=forecast(bundle);test=frame(family,'internal_test',0,TEST,y,p);test.to_csv(d/'test_predictions.csv',index=False,float_format='%.17g')
        final=fit(y,cfg);idx,p,rates=forecast(final);future=frame(family,'future_unlabelled',0,FINAL,y,p)
        assert future.actual_kwh.isna().all();future.to_csv(d/'future_path.csv',index=False,float_format='%.17g');rates.to_csv(d/'future_period_rates.csv',index=False,float_format='%.17g')
        joblib.dump(final,d/'saved_model.joblib');restored=joblib.load(d/'saved_model.joblib');_,q,_=forecast(restored)
        assert np.allclose(q,p,rtol=0,atol=1e-8)
        card={'family':family,'selected_config':cfg,'validation':chosen['validation'],'test':summary(test),'validation_period_metadata':chosen['period_metadata_by_fold'],
            'final_period_metadata':final['period_metadata'],'saved_model':str(d/'saved_model.joblib'),'reload_max_abs_diff':float(np.abs(q-p).max()),
            'prediction_csv':{'validation':str(d/'validation_predictions.csv'),'test':str(d/'test_predictions.csv'),'future':str(d/'future_path.csv')},
            'allocation':'Historical-only hourly multipliers; each learned day shape has sum24. Weekly/monthly rate mapping is not joint hierarchical reconciliation of already-known partial observations.',
            'May_labels_read':False,'monthly_limit':family=='monthly_level'}
        write(d/'results.json',card);results[family]=card
    write(OUT/'results.json',{'status':'complete','models':results,'selection_lock':str(OUT/'selection_lock.json'),'selection_lock_sha256':sha(OUT/'selection_lock.json'),'candidate_count':len(candidates),'runtime_seconds':time.perf_counter()-started,'May_labels_read':False})
    print(json.dumps({k:v['validation']['mean_origin_late505_mae_kwh'] for k,v in results.items()},ensure_ascii=True),flush=True)
if __name__=='__main__':main()
