"""Train-prefix feature screening and direct hourly forecasting; no May reads.

New files stay in this classical directory. Previous experiments are read-only.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import argparse,hashlib,json,time,warnings
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
OUT=Path(__file__).resolve().parent
REV=OUT.parent
ROOT=REV.parents[1]
sys.path.insert(0,str(REV))
from feature_common import load_hourly,feature_columns,select_sensor_features,TARGET,SEED,THREADS
sys.path.insert(0,str(ROOT/'output/optimization_20261013/forecast'))
from opt_common import metrics,prediction_frame,write_json,DAY_FOLD_STARTS,APRIL_TEST_ORIGINS
threadpool_limits(limits=THREADS)
GROUPS=['target_fixed_calendar','target_aligned_calendar','module_aligned_calendar',
        'selected_sensor_calendar','all_sensor_calendar']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def prefix(frame,origin):
    h=frame.loc[frame.index<pd.Timestamp(origin)]
    if len(h)<168 or h.index[-1]+pd.Timedelta(hours=1)!=pd.Timestamp(origin):raise ValueError('Bad prefix')
    return h
def idx_future(h,p):return pd.date_range(h.index[-1]+pd.Timedelta(hours=1),periods=p,freq='h')
def summary(f):
    out={'path_24_values':metrics(f.actual_kwh,f.predicted_kwh),'origin_count':int(f.origin.nunique())}
    for lead in [1,6,24]:
        sub=f.loc[f.lead_hours.eq(lead)];out[f'lead_{lead}']=metrics(sub.actual_kwh,sub.predicted_kwh)
    return out
def calendar(idx):
    idx=pd.DatetimeIndex(idx);hour=idx.hour.to_numpy();dow=idx.dayofweek.to_numpy()
    return np.column_stack([np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),
                           np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7),dow>=5])
CAL_NAMES=['target_hour_sin','target_hour_cos','target_dow_sin','target_dow_cos','target_weekend']

def feature_schema(group,selected):
    base=[('target_history','lag1'),('target_history','lag24'),('target_history','lag168'),
          ('target_history','mean24'),('target_history','mean168'),('target_history','mean720'),
          ('target_history','std24'),('target_history','std168')]
    if group!='target_fixed_calendar':base += [('target_history','target_aligned_daily'),('target_history','target_aligned_weekly')]
    base += [('calendar','lead_fraction')]+[('calendar',x) for x in CAL_NAMES]
    raw=[]
    if group in GROUPS[2:]:
        for col in feature_columns('module_energy'):
            for lag in ['origin_lag1','origin_mean24','target_aligned_daily','target_aligned_weekly']:
                raw.append(('module_energy13_history',f'{col}::{lag}'))
    sensors=selected if group=='selected_sensor_calendar' else (feature_columns('all_sensors') if group=='all_sensor_calendar' else [])
    for col in sensors:
        for lag in ['origin_lag1','target_aligned_daily']:
            raw.append(('selected_sensor_lags' if group=='selected_sensor_calendar' else 'all_sensor_lags',f'{col}::{lag}'))
    return base+raw

def features_at(frame,pos,offsets,group,selected):
    """At origin position pos, every source index is < pos, even long horizons.

    offsets0..23 use real target-hour daily/weekly lags. Beyond that, phase-wrap
    into the last observed day/week: explicit historical analogues, not future lags.
    """
    offsets=np.asarray(offsets,dtype=int);y=frame[TARGET].to_numpy(dtype=float)
    daily=pos-24+offsets%24;weekly=pos-168+offsets%168
    assert np.all(daily<pos) and np.all(weekly<pos) and daily.min()>=0 and weekly.min()>=0
    vals=[y[pos-1],y[pos-24],y[pos-168],y[pos-24:pos].mean(),y[pos-168:pos].mean(),
          y[max(0,pos-720):pos].mean(),y[pos-24:pos].std(),y[pos-168:pos].std()]
    columns=[np.repeat(v,len(offsets)) for v in vals]
    if group!='target_fixed_calendar':columns += [y[daily],y[weekly]]
    origin=frame.index[0]+pd.Timedelta(hours=pos)
    idx=pd.DatetimeIndex([origin+pd.Timedelta(hours=int(o)) for o in offsets])
    columns += [offsets/24.]+[calendar(idx)[:,j] for j in range(len(CAL_NAMES))]
    if group in GROUPS[2:]:
        for col in feature_columns('module_energy'):
            a=frame[col].to_numpy(dtype=float)
            columns += [np.repeat(a[pos-1],len(offsets)),np.repeat(a[pos-24:pos].mean(),len(offsets)),a[daily],a[weekly]]
    sensors=selected if group=='selected_sensor_calendar' else (feature_columns('all_sensors') if group=='all_sensor_calendar' else [])
    for col in sensors:
        a=frame[col].to_numpy(dtype=float);columns += [np.repeat(a[pos-1],len(offsets)),a[daily]]
    x=np.column_stack(columns).astype(np.float64)
    assert x.shape[1]==len(feature_schema(group,selected)) and np.isfinite(x).all()
    return x

def train_matrix(h,points,group,selected):
    leads=np.arange(24) if points==24 else np.array(sorted(set(range(25))|set(range(48,745,24))))
    xs,ys=[],[]
    for pos in range(168,len(h),6):
        offsets=leads[pos+leads<len(h)]
        if not len(offsets):continue
        xs.append(features_at(h,pos,offsets,group,selected));ys.append(h[TARGET].to_numpy(dtype=float)[pos+offsets])
    return np.concatenate(xs),np.concatenate(ys)

def arimax_exog(h,positions,cfg,selected):
    positions=np.asarray(positions,dtype=int)
    idx=pd.DatetimeIndex([h.index[0]+pd.Timedelta(hours=int(i)) for i in positions])
    columns=[calendar(idx)[:,j] for j in range(len(CAL_NAMES))];names=CAL_NAMES.copy()
    if cfg['exog']!='calendar':
        a=h[TARGET].to_numpy(dtype=float)
        columns += [a[positions-24],a[positions-168]];names += ['target_lag24','target_lag168']
    if cfg['exog']=='selected':
        for col in selected:
            columns.append(h[col].to_numpy(dtype=float)[positions-24]);names.append(col+'::lag24')
    return np.column_stack(columns),names

def configs():
    out=[]
    for group in GROUPS:
        out.append({'model':'xgboost','group':group,'objective':'reg:squarederror','max_depth':1,'n_estimators':80,
                    'learning_rate':.02,'min_child_weight':50,'reg_lambda':100.,'residual_shrinkage':.5})
        out.append({'model':'xgboost','group':group,'objective':'reg:absoluteerror','max_depth':2,'n_estimators':100,
                    'learning_rate':.02,'min_child_weight':100,'reg_lambda':100.,'residual_shrinkage':.5})
    for group,alpha in [('target_aligned_calendar',1e3),('target_aligned_calendar',1e5),
                        ('module_aligned_calendar',1e4),('selected_sensor_calendar',1e4),
                        ('all_sensor_calendar',1e4),('all_sensor_calendar',1e6)]:
        out.append({'model':'ridge','group':group,'alpha':alpha})
    for group in ['selected_sensor_calendar','all_sensor_calendar']:
        out.append({'model':'elasticnet','group':group,'alpha':1.,'l1_ratio':.2})
    for order,exog in [([0,0,0],'calendar'),([0,0,0],'target_history'),([1,0,0],'selected')]:
        out.append({'model':'arimax','order':order,'exog':exog})
    for kind in ['train_mean','train_median','ema720']:out.append({'model':'baseline','kind':kind})
    assert len(out)==24
    return out

def fit_model(h,points,cfg,selected,cache=None):
    model=cfg['model'];bundle={'config':cfg,'selected_sensor_columns':selected,'observed_count':len(h),
                              'history_end':str(h.index[-1]),'history_count':len(h)}
    if model in ['xgboost','ridge','elasticnet']:
        key=(len(h),points,cfg['group'],tuple(selected))
        if cache is not None and key in cache:X,y=cache[key]
        else:
            X,y=train_matrix(h,points,cfg['group'],selected)
            if cache is not None:cache[key]=(X,y)
        center=float(y.mean());bundle['target_center']=center
        if model=='xgboost':
            from xgboost import XGBRegressor
            params={k:v for k,v in cfg.items() if k not in ['model','group','residual_shrinkage']}
            fitted=XGBRegressor(**params,tree_method='hist',n_jobs=THREADS,random_state=SEED).fit(X,y-center)
        else:
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import Ridge,ElasticNet
            estimator=Ridge(alpha=cfg['alpha'],solver='lsqr',tol=1e-7) if model=='ridge' else ElasticNet(alpha=cfg['alpha'],l1_ratio=cfg['l1_ratio'],max_iter=5000,tol=1e-6,random_state=SEED)
            fitted=make_pipeline(StandardScaler(),estimator).fit(X,y-center)
        bundle.update(object=fitted,feature_schema=feature_schema(cfg['group'],selected),training_examples=len(y),training_scaler_only=model!='xgboost')
    elif model=='arimax':
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        X,names=arimax_exog(h,np.arange(168,len(h)),cfg,selected)
        mean=X.mean(axis=0);scale=X.std(axis=0);scale[scale<1e-12]=1
        center=float(h[TARGET].iloc[168:].mean())
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            fitted=SARIMAX(h[TARGET].to_numpy(dtype=float)[168:]-center,exog=(X-mean)/scale,
                          order=cfg['order'],trend='c',enforce_stationarity=True,enforce_invertibility=True).fit(disp=False,maxiter=150)
        if not fitted.mle_retvals.get('converged',True):raise ValueError('ARIMAX did not converge')
        bundle.update(object=fitted,target_center=center,exog_mean=mean,exog_scale=scale,exog_names=names,
                      aic_training_only=float(fitted.aic),warnings=[str(w.message) for w in ws])
    else:bundle.update(mean=float(h[TARGET].mean()),median=float(h[TARGET].median()))
    return bundle

def predict(bundle,h,points,override_x=None):
    cfg=bundle['config'];kind=cfg['model'];selected=bundle['selected_sensor_columns']
    if kind in ['xgboost','ridge','elasticnet']:
        X=features_at(h,len(h),np.arange(points),cfg['group'],selected) if override_x is None else override_x
        residual=np.asarray(bundle['object'].predict(X),dtype=np.float64)
        return bundle['target_center']+cfg.get('residual_shrinkage',1.)*residual
    if kind=='arimax':
        seen=bundle['observed_count']
        if len(h)>seen:
            X,_=arimax_exog(h,np.arange(seen,len(h)),cfg,selected)
            bundle['object']=bundle['object'].extend(h[TARGET].to_numpy(dtype=float)[seen:]-bundle['target_center'],exog=(X-bundle['exog_mean'])/bundle['exog_scale'])
            bundle['observed_count']=len(h)
        X,_=arimax_exog(h,np.arange(len(h),len(h)+points),cfg,selected)
        return bundle['target_center']+np.asarray(bundle['object'].forecast(points,exog=(X-bundle['exog_mean'])/bundle['exog_scale']),dtype=np.float64)
    if cfg['kind']=='train_mean':v=bundle['mean']
    elif cfg['kind']=='train_median':v=bundle['median']
    elif len(h)<bundle['history_count'] and str(h.index[-1])==bundle['history_end'] and 'frozen_ema_value' in bundle:
        v=bundle['frozen_ema_value']
    else:v=float(h[TARGET].ewm(span=720,adjust=False).mean().iloc[-1])
    return np.repeat(v,points).astype(np.float64)

def evaluate(frame,cfg,selected,fit_origin,origins,split,fold,cache=None):
    fitted=fit_model(prefix(frame,fit_origin),24,cfg,selected,cache);frames=[]
    for origin in origins:
        h=prefix(frame,origin);idx=idx_future(h,24);p=predict(fitted,h,24)
        frames.append(prediction_frame(cfg['model'],'dayahead24',split,fold,origin,idx,frame.loc[idx,TARGET],p))
    return pd.concat(frames,ignore_index=True)

def selections(frame):
    result={}
    for label,origin in [(f'fold{i}',t) for i,t in enumerate(DAY_FOLD_STARTS,1)]+[('pre_April',pd.Timestamp('2025-04-01'))]:
        selected,audit=select_sensor_features(prefix(frame,origin),max_features=12)
        path=OUT/'feature_selection'/f'{label}.json';write_json(path,audit)
        result[label]={'columns':selected,'audit_path':str(path),'sha256':sha(path),'fit_origin':str(origin)}
    return result

def search():
    started=time.monotonic();frame=load_hourly();sels=selections(frame);trials=configs()
    write_json(OUT/'search_plan.json',{'predeclared_candidates':trials,'count':24,'groups':GROUPS,
              'target_aligned_past_values':'For offset0..23, y/module/sensor[origin+offset-24] and y/module[origin+offset-168] are strictly pre-origin.',
              'all_sensor_limit':'403 completed-hour sensor summaries; origin-lag1 and target-hour-aligned daily analogues. No raw target/future-hour measurements.',
              'selection':'Lowest70-origin mean24path MAE; separate family winners and strongest baseline; no April/May tuning.',
              'data_sha256':sha(REV/'data/hourly_features.csv'),'selector_sha256':sha(REV/'feature_common.py')})
    cache={};records=[];best={}
    for candidate,cfg in enumerate(trials,1):
        try:
            folds=[]
            for fold,start in enumerate(DAY_FOLD_STARTS,1):
                selected=sels[f'fold{fold}']['columns']
                folds.append(evaluate(frame,cfg,selected,start,pd.date_range(start,periods=14,freq='D'),'validation',fold,cache))
            f=pd.concat(folds,ignore_index=True);stats=summary(f);value=stats['path_24_values']['mae_kwh']
            record={'candidate':candidate,'config':cfg,'validation_metrics':stats,'objective_mae_kwh':value}
            family=cfg['model']
            if family not in best or value<best[family]['objective_mae_kwh']:
                best[family]=record
                dest=OUT/family;dest.mkdir(parents=True,exist_ok=True)
                f.to_csv(dest/'validation_predictions.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
        except Exception as exc:record={'candidate':candidate,'config':cfg,'error':repr(exc)}
        records.append(record);write_json(OUT/'search_progress.json',{'records':records,'best_by_family':best,'elapsed_seconds':time.monotonic()-started})
        print(f'candidate {candidate}/24 {cfg["model"]}/{cfg.get("group",cfg.get("exog",cfg.get("kind")))}: {record.get("objective_mae_kwh",record.get("error"))} elapsed {time.monotonic()-started:.1f}s',flush=True)
    if len(best)!=5:raise RuntimeError('A model family has no admissible candidate')
    selected=min(best,key=lambda k:best[k]['objective_mae_kwh'])
    lock={'scenario':'dayahead24','selected_model':selected,'models':best,'feature_selection':sels,'records':records,
          'selected_using_April':False,'selected_using_May':False,'data_sha256':sha(REV/'data/hourly_features.csv'),
          'code_sha256':sha(__file__),'selector_sha256':sha(REV/'feature_common.py'),'status':'frozen_before_new_April_evaluation',
          'post_EDA_limit':'April and May were inspected in previous work. New hyperparameters and sensor selection use historical training/CV only; this is not a pristine prospective holdout.'}
    write_json(OUT/'selection_lock.json',lock)
    print('Classical family and global validation selection frozen; April not evaluated.',flush=True)

def audit_new(frame):
    cfg=configs()[6];selected=feature_columns('physical_sensors')[:2];pos=960
    for group in GROUPS:
        x=features_at(frame,pos,np.arange(24),group,selected)
        changed=frame.copy();changed.iloc[pos:]=-999999.
        assert np.array_equal(x,features_at(changed,pos,np.arange(24),group,selected))
        assert np.array_equal(x,features_at(frame.iloc[:pos],pos,np.arange(24),group,selected))
        long=features_at(frame,pos,np.arange(745),group,selected)
        assert np.array_equal(long,features_at(changed,pos,np.arange(745),group,selected))
    write_json(OUT/'new_feature_alignment_audit.json',{'status':'passed','groups_checked':GROUPS,'future_mutation_changes_features':False,
         'full_frame_vs_prefix_features_max_absdiff':0.,'long_future_mutation_max_absdiff':0.,
         'daily_lag_indices_offset0_23':[pos-24,pos-1],'weekly_lag_indices_offset0_23':[pos-168,pos-145],
         'long_phase_wrap':'daily=origin-24+(offset%24); weekly=origin-168+(offset%168); always strictly earlier than origin',
         'train_matrix_boundaries':'Offsets are filtered with origin_position+offset<len(fitting_prefix); no target outside prefix admitted.',
         'known_future_covariates':['calendar','horizon_offset'],'future_sensor_measurements_used':False})

def importance(frame,cfg,selected,family):
    """Nonselection diagnostic: March chronological inner7-day holdout, block permutation.

    Sensor screening is refit on the inner prefix, not the seven diagnostic days.
    """
    if family not in ['xgboost','ridge','elasticnet']:return None
    end=pd.Timestamp('2025-04-01');inner_origin=end-pd.Timedelta(days=7);h=prefix(frame,inner_origin)
    inner_selected,selection=select_sensor_features(h,max_features=12)
    dest=OUT/family;write_json(dest/'importance_inner_selection.json',selection)
    fitted=fit_model(h,24,cfg,inner_selected);xs=[];ys=[]
    for origin in pd.date_range(inner_origin,periods=7,freq='D'):
        past=prefix(frame,origin);xs.append(features_at(past,len(past),np.arange(24),cfg['group'],inner_selected));ys.append(frame.loc[idx_future(past,24),TARGET].to_numpy(dtype=float))
    X=np.concatenate(xs);a=np.concatenate(ys)
    p=predict(fitted,h,len(X),override_x=X);base=metrics(a,p)['mae_kwh'];schema=fitted['feature_schema']
    rng=np.random.default_rng(SEED);rows=[]
    for group in sorted(set(g for g,n in schema)):
        cols=[i for i,(g,n) in enumerate(schema) if g==group];diff=[]
        for repeat in range(5):
            new=X.copy();order=rng.permutation(7)
            new[:,cols]=X.reshape(7,24,-1)[order][:,:,cols].reshape(len(X),len(cols))
            changed=predict(fitted,h,len(X),override_x=new)
            diff.append(metrics(a,changed)['mae_kwh']-base)
        rows.append({'group':group,'columns':len(cols),'baseline_inner_mae_kwh':base,'permuted_delta_mae_kwh_mean':float(np.mean(diff)),
                     'permuted_delta_mae_kwh_std':float(np.std(diff,ddof=1)),'repeats':5})
    pd.DataFrame(rows).to_csv(dest/'chronological_group_permutation.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    if family in ['ridge','elasticnet']:
        coeff=fitted['object'][-1].coef_
        raw=fitted['object'][0].scale_
        pd.DataFrame({'group':[g for g,n in schema],'feature':[n for g,n in schema],
                      'coefficient_per_training_std':coeff,'coefficient_raw_unit':coeff/raw}).to_csv(dest/'coefficients_inner.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    else:
        pd.DataFrame({'group':[g for g,n in schema],'feature':[n for g,n in schema],
                      'training_importance_gain':fitted['object'].feature_importances_}).to_csv(dest/'gain_inner.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    result={'inner_fit_end':str(h.index[-1]),'diagnostic_target_start':str(inner_origin),'diagnostic_target_end':str(end-pd.Timedelta(hours=1)),
            'not_used_for_config_selection':True,'sensor_selection':str(dest/'importance_inner_selection.json'),
            'permutation':'Shuffle whole24-hour feature blocks across seven chronological holdout days, preserving hour phase; five repeats per group.',
            'base_mae_kwh':base,'groups':rows,'caution':'Correlated groups can substitute; negative differences indicate noise/harm in this diagnostic. This is association and conditional predictive reliance, not causality.'}
    write_json(dest/'importance.json',result);return result

def save_and_check(bundle,h,points,dest):
    dest.mkdir(parents=True,exist_ok=True);bundle['forecast_context']=h.iloc[-720:].copy();bundle['forecast_points']=points
    if bundle['config']['model']=='baseline' and bundle['config']['kind']=='ema720':
        bundle['frozen_ema_value']=float(h[TARGET].ewm(span=720,adjust=False).mean().iloc[-1])
    p=predict(bundle,h,points)
    joblib.dump(bundle,dest/'model.joblib',compress=3)
    loaded=joblib.load(dest/'model.joblib');again=predict(loaded,h,points)
    error=float(np.max(np.abs(p-again)));assert error<=1e-8
    cfg=bundle['config'];native_error=None
    if cfg['model']=='xgboost':
        from xgboost import XGBRegressor
        bundle['object'].save_model(dest/'model_native.json')
        native=XGBRegressor();native.load_model(dest/'model_native.json')
        loaded['object']=native;native_error=float(np.max(np.abs(p-predict(loaded,h,points))));assert native_error<=1e-8
    elif cfg['model']=='arimax':bundle['object'].save(dest/'model_statsmodels.pkl')
    metadata={'model_path':str(dest/'model.joblib'),'sha256':sha(dest/'model.joblib'),'config':cfg,
              'selected_sensor_columns':bundle['selected_sensor_columns'],'feature_schema':bundle.get('feature_schema',bundle.get('exog_names')),
              'history_end':str(h.index[-1]),'history_count':len(h),'target_center':bundle.get('target_center'),
              'forecast_points':points,'reload_max_absdiff_kwh':error,'native_reload_max_absdiff_kwh':native_error,
              'scaler_fit':'Training prefix only for Ridge/ElasticNet and ARIMAX exog; no global scaling for XGB.',
              'context_storage':'Last720 completed hourly rows saved for standalone fixed-origin prediction; operational new-history input is not implemented.'}
    write_json(dest/'model_metadata.json',metadata);return p,metadata

def test_and_finalize():
    frame=load_hourly();audit_new(frame)
    lock=json.loads((OUT/'selection_lock.json').read_text(encoding='utf-8'))
    if lock['data_sha256']!=sha(REV/'data/hourly_features.csv') or lock['selector_sha256']!=sha(REV/'feature_common.py'):raise ValueError('Frozen data/selector changed')
    selected=lock['feature_selection']['pre_April']['columns'];models={};cache={}
    ablation=[{'candidate':r['candidate'],'model':r['config']['model'],'group':r['config'].get('group',r['config'].get('exog',r['config'].get('kind'))),
               'validation_mae_kwh':r.get('objective_mae_kwh'),'error':r.get('error')} for r in lock['records']]
    pd.DataFrame(ablation).to_csv(OUT/'feature_group_ablation.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    for family,chosen in lock['models'].items():
        cfg=chosen['config'];dest=OUT/family
        diagnostic=importance(frame,cfg,selected,family)
        test=evaluate(frame,cfg,selected,pd.Timestamp('2025-04-01'),APRIL_TEST_ORIGINS,'April_test',0,cache)
        test.to_csv(dest/'test_predictions.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
        fit=fit_model(frame,24,cfg,selected,cache);p,model_info=save_and_check(fit,frame,24,dest/'saved_model')
        idx=idx_future(frame,24)
        prediction_frame(family,'dayahead24','future_unlabelled',0,idx[0],idx,np.full(24,np.nan),p).to_csv(dest/'future_path.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
        model={'selected_config':cfg,'selected_candidate':chosen['candidate'],'trials_count':sum(r['config']['model']==family for r in lock['records']),
               'validation_metrics':chosen['validation_metrics'],'test_metrics':summary(test),
               'feature_selection':lock['feature_selection'],'frozen_test_and_final_sensor_columns':selected,
               'final_feature_schema':model_info['feature_schema'],'saved_model':model_info,
               'paths':{'validation_csv':str(dest/'validation_predictions.csv'),'test_csv':str(dest/'test_predictions.csv'),'future_csv':str(dest/'future_path.csv')},
               'importance':diagnostic,'input_scope':'Past aggregate, known calendar, optional module/sensor history according to selected group. Future sensor readings unavailable.'}
        write_json(dest/'results.json',model);models[family]=model
        print(f'{family}: validation {chosen["validation_metrics"]["path_24_values"]["mae_kwh"]:.6f}, April {model["test_metrics"]["path_24_values"]["mae_kwh"]:.6f}',flush=True)
    write_json(OUT/'results.json',{'scenario':'dayahead24','models':models,'selected_model':lock['selected_model'],
               'selection_lock':str(OUT/'selection_lock.json'),'selection_lock_sha256':sha(OUT/'selection_lock.json'),
               'validation_origin_count':70,'test_origin_count':29,'validation_points':1680,'test_points':696,
               'feature_group_ablation_csv':str(OUT/'feature_group_ablation.csv'),'previous_temporal_audit':str(OUT/'previous_feature_audit.json'),
               'new_temporal_audit':str(OUT/'new_feature_alignment_audit.json'),'May_labels_read':False,
               'limitations':['Post-EDA experiment; earlier work already saw April and May.','Sensor screening may select none; no guaranteed improvement.',
                              '24h lead24 endpoint is origin+23h,24h after the preceding hour key.','Provisional13-module energy sum; unknown meter hierarchy/units.',
                              'Future raw sensor measurements cannot be used for745h prediction.','Input-rich new models differ from prior univariate models.'],
               'standalone_inference':'python predict_classical.py --model selected|xgboost|ridge|elasticnet|arimax|baseline --output FILE.csv'})

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['search','test'],required=True);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='search':search()
    else:test_and_finalize()
if __name__=='__main__':main()
