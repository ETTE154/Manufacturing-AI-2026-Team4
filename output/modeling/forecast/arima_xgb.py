"""ARIMA and pooled-direct XGBoost without future sensor measurements."""
import warnings
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from common import score,write_json

ARIMA_ORDERS = [(0,0,0),(1,0,0),(0,0,1),(1,0,1),(2,0,0),
                (0,1,0),(1,1,0),(0,1,1),(1,1,1)]
XGB_GRID = [
    {'n_estimators':150,'max_depth':2,'learning_rate':0.03,'min_child_weight':20},
    {'n_estimators':300,'max_depth':2,'learning_rate':0.03,'min_child_weight':20},
    {'n_estimators':150,'max_depth':3,'learning_rate':0.03,'min_child_weight':20},
    {'n_estimators':300,'max_depth':3,'learning_rate':0.03,'min_child_weight':20},
]
# Pooled-direct training includes all near-term clock offsets and long leads.
DIRECT_LEADS = sorted(set(range(25))|set(range(48,745,24))|{96,192,336,720,744})
FEATURE_NAMES = ['lag_1','lag_24','lag_168','mean_24','mean_168','mean_720',
                 'std_24','std_168','origin_hour_sin','origin_hour_cos',
                 'target_hour_sin','target_hour_cos','target_weekday_sin','target_weekday_cos',
                 'target_month_sin','target_month_cos','target_weekend','lead_days']

def _arima_fit(history,order):
    from statsmodels.tsa.arima.model import ARIMA
    order = tuple(int(x) for x in order)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fitted = ARIMA(history.to_numpy(dtype=float),order=order,
                       trend='c' if order[1]==0 else 'n',
                       enforce_stationarity=True,enforce_invertibility=True).fit(
                           method_kwargs={'maxiter':200,'disp':0})
    meta = {'order':list(order),'aic_training_only':float(fitted.aic),
            'converged':bool(fitted.mle_retvals.get('converged',True)),
            'warnings':[str(w.message) for w in caught]}
    return fitted,meta

def select_arima(history,future,actual):
    fitted_candidates = []
    records = []
    for order in ARIMA_ORDERS:
        try:
            fit,meta = _arima_fit(history,order)
            records.append(meta)
            if np.isfinite(fit.aic) and meta['converged']:
                fitted_candidates.append((float(fit.aic),order,fit,meta))
        except Exception as exc:
            records.append({'order':list(order),'error':repr(exc)})
    if not fitted_candidates:
        raise RuntimeError('No converged ARIMA candidate')
    # AIC sees only history. Validation compares only the three AIC-shortlisted orders.
    shortlist = sorted(fitted_candidates,key=lambda x:x[0])[:3]
    validation = []
    for _,order,fit,meta in shortlist:
        pred = np.asarray(fit.forecast(steps=len(future)),dtype=float)
        validation.append({'order':list(order),'scores':score(actual,pred),
                           'aic_training_only':meta['aic_training_only'],
                           'prediction':pred})
    winner = min(validation,key=lambda x:x['scores']['mae_kwh'])
    config = {'order':winner['order'],'trend':'c' if winner['order'][1]==0 else 'n'}
    evidence = {'candidate_fits':records,'aic_shortlist_validation':[
        {k:v for k,v in item.items() if k!='prediction'} for item in validation],
        'selection':'Converged training-only AIC top 3, then lowest 745-row blocked validation MAE',
        'selected_config':config}
    return config,evidence,winner['prediction']

def predict_saved_arima(model_path,metadata_path,future):
    """Infer from this task's trusted local statsmodels pickle artifact."""
    from statsmodels.tsa.arima.model import ARIMAResults
    metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    if hashlib.sha256(Path(model_path).read_bytes()).hexdigest()!=metadata['model_sha256']:
        raise ValueError('Saved ARIMA artifact checksum changed')
    if pd.Timestamp(metadata['history_end'])+pd.Timedelta(hours=1)!=future[0]:
        raise ValueError('Saved ARIMA forecast must start after the saved history')
    fitted = ARIMAResults.load(str(model_path))
    return np.asarray(fitted.forecast(steps=len(future)),dtype=float)

def predict_arima(history,future,config,save_path=None,metadata_path=None):
    fit,meta = _arima_fit(history,config['order'])
    if not meta['converged']:
        raise RuntimeError(f'Frozen ARIMA order failed to converge on refit: {meta}')
    predicted = np.asarray(fit.forecast(steps=len(future)),dtype=float)
    if save_path is not None:
        save_path = Path(save_path)
        metadata_path = Path(metadata_path) if metadata_path is not None else save_path.with_name(save_path.stem+'_metadata.json')
        save_path.parent.mkdir(parents=True,exist_ok=True)
        fit.save(str(save_path))
        saved = {'model':'arima','format':'statsmodels ARIMAResults trusted local pickle',
                 'frozen_config':config,'history_start':str(history.index[0]),'history_end':str(history.index[-1]),
                 'history_count':len(history),'feature_names':['past_provisional_sum_kwh'],
                 'target_center':None,'forecast_origin':str(future[0]),
                 'model_path':str(save_path),'model_sha256':hashlib.sha256(save_path.read_bytes()).hexdigest()}
        write_json(metadata_path,saved)
        restored = predict_saved_arima(save_path,metadata_path,future)
        if not np.allclose(predicted,restored,rtol=0,atol=1e-9):
            raise ValueError('Saved ARIMA reload changed forecast')
        saved['reload_forecast_max_abs_diff_kwh'] = float(np.max(np.abs(predicted-restored)))
        saved['reload_forecast_rows'] = len(restored)
        write_json(metadata_path,saved)
        meta['saved_model'] = {**saved,'metadata_path':str(metadata_path)}
    return predicted,meta

def _features(values,origin_pos,origin_time,target_time,lead):
    # origin_pos is the first unobserved index; every target lag is before origin.
    a = np.asarray(values,dtype=float)
    tail24 = a[max(0,origin_pos-24):origin_pos]
    tail168 = a[max(0,origin_pos-168):origin_pos]
    tail720 = a[max(0,origin_pos-720):origin_pos]
    oh = origin_time.hour
    th = target_time.hour
    wd = target_time.dayofweek
    mo = target_time.month
    return [a[origin_pos-1],a[origin_pos-24],a[origin_pos-168],
            np.mean(tail24),np.mean(tail168),np.mean(tail720),np.std(tail24),np.std(tail168),
            np.sin(2*np.pi*oh/24),np.cos(2*np.pi*oh/24),
            np.sin(2*np.pi*th/24),np.cos(2*np.pi*th/24),
            np.sin(2*np.pi*wd/7),np.cos(2*np.pi*wd/7),
            np.sin(2*np.pi*(mo-1)/12),np.cos(2*np.pi*(mo-1)/12),int(wd>=5),lead/24.0]

def direct_training_examples(history):
    values = history.to_numpy(dtype=float)
    rows,targets,origin_times,target_times,leads = [],[],[],[],[]
    # Six-hour origins diversify clock phase; labels always remain inside history.
    for pos in range(168,len(history),6):
        origin_time = history.index[pos]
        for lead in DIRECT_LEADS:
            target_pos = pos+lead
            if target_pos>=len(history):
                continue
            target_time = history.index[target_pos]
            rows.append(_features(values,pos,origin_time,target_time,lead))
            targets.append(values[target_pos])
            origin_times.append(origin_time)
            target_times.append(target_time)
            leads.append(lead)
    if not rows:
        raise ValueError('Insufficient direct XGBoost training examples')
    metadata = {'examples':len(rows),'features':FEATURE_NAMES,'history_count':len(history),
                'example_origin_start':str(min(origin_times)),'example_origin_end':str(max(origin_times)),
                'example_target_start':str(min(target_times)),'example_target_end':str(max(target_times)),
                'max_example_lead_hours':max(leads),'origin_stride_hours':6,
                'lead_training_grid_hours':DIRECT_LEADS,
                'protocol':'Pooled direct model: features freeze at each historical origin; all example labels occur before the current training cutoff. Forecast all 745 future hours in one batch.'}
    return np.asarray(rows,dtype=np.float32),np.asarray(targets,dtype=np.float64),metadata

def direct_future_features(history,future):
    origin = future[0]
    expected_origin = history.index[-1]+pd.Timedelta(hours=1)
    if origin!=expected_origin:
        raise ValueError('XGBoost bridge must start at first hour after observed history')
    values = history.to_numpy(dtype=float)
    return np.asarray([_features(values,len(history),origin,t,lead) for lead,t in enumerate(future)],dtype=np.float32)

def _xgb_fit(X,y,config,n_jobs):
    from xgboost import XGBRegressor
    # Residual centering avoids float32 absolute-target precision impairing tiny variation.
    center = float(np.mean(y))
    estimator = XGBRegressor(objective='reg:squarederror',tree_method='hist',
                             random_state=42,n_jobs=n_jobs,subsample=1.0,colsample_bytree=1.0,
                             reg_lambda=10.0,**config)
    estimator.fit(X,y-center)
    return estimator,center

def select_xgb(history,future,actual,n_jobs=2):
    X,y,training_meta = direct_training_examples(history)
    future_X = direct_future_features(history,future)
    records = []
    predictions = []
    for config in XGB_GRID:
        model,center = _xgb_fit(X,y,config,n_jobs)
        pred = np.asarray(model.predict(future_X),dtype=float)+center
        records.append({'config':dict(config),'scores':score(actual,pred),
                        'target_center':center})
        predictions.append(pred)
    chosen_index = min(range(len(records)),key=lambda i:records[i]['scores']['mae_kwh'])
    config = records[chosen_index]['config']
    evidence = {'training_examples':training_meta,'validation_candidates':records,
                'selected_config':config,'selection':'Lowest fixed-origin 745-row blocked validation MAE; no early stopping uses test or May targets'}
    return config,evidence,predictions[chosen_index]

def predict_saved_xgb(model_path,metadata_path,future,n_jobs=4):
    """Infer from native XGBoost JSON plus the frozen past-only 720-hour tail."""
    from xgboost import XGBRegressor
    metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    if hashlib.sha256(Path(model_path).read_bytes()).hexdigest()!=metadata['model_sha256']:
        raise ValueError('Saved XGBoost artifact checksum changed')
    if metadata['feature_names']!=FEATURE_NAMES:
        raise ValueError('Saved XGBoost feature schema differs')
    last = pd.Timestamp(metadata['history_end'])
    if last+pd.Timedelta(hours=1)!=future[0]:
        raise ValueError('Saved XGBoost forecast must start after saved history')
    values = np.asarray(metadata['history_tail_kwh'],dtype=float)
    past = pd.Series(values,index=pd.date_range(end=last,periods=len(values),freq='h'))
    X = direct_future_features(past,future)
    model = XGBRegressor()
    model.load_model(str(model_path))
    model.set_params(n_jobs=n_jobs)
    return np.asarray(model.predict(X),dtype=float)+float(metadata['target_center'])

def predict_xgb(history,future,config,n_jobs=2,save_path=None,metadata_path=None):
    X,y,meta = direct_training_examples(history)
    future_X = direct_future_features(history,future)
    model,center = _xgb_fit(X,y,config,n_jobs)
    prediction = np.asarray(model.predict(future_X),dtype=float)+center
    meta.update({'target_center':center,'feature_importance_gain':dict(zip(FEATURE_NAMES,model.feature_importances_.tolist()))})
    if save_path is not None:
        save_path = Path(save_path)
        metadata_path = Path(metadata_path) if metadata_path is not None else save_path.with_name(save_path.stem+'_metadata.json')
        save_path.parent.mkdir(parents=True,exist_ok=True)
        model.save_model(str(save_path))
        saved = {'model':'xgboost','format':'Native XGBoost JSON',
                 'frozen_config':config,'target_center':center,'feature_names':FEATURE_NAMES,
                 'history_start':str(history.index[0]),'history_end':str(history.index[-1]),'history_count':len(history),
                 'history_tail_kwh':history.to_numpy(dtype=float)[-720:].tolist(),
                 'history_tail_start':str(history.index[max(0,len(history)-720)]),
                 'forecast_origin':str(future[0]),'protocol':meta['protocol'],
                 'model_path':str(save_path),'model_sha256':hashlib.sha256(save_path.read_bytes()).hexdigest()}
        write_json(metadata_path,saved)
        restored = predict_saved_xgb(save_path,metadata_path,future,n_jobs=n_jobs)
        if not np.allclose(prediction,restored,rtol=0,atol=1e-9):
            raise ValueError('Saved XGBoost reload changed forecast')
        saved['reload_forecast_max_abs_diff_kwh'] = float(np.max(np.abs(prediction-restored)))
        saved['reload_forecast_rows'] = len(restored)
        write_json(metadata_path,saved)
        meta['saved_model'] = {**{k:v for k,v in saved.items() if k!='history_tail_kwh'},'metadata_path':str(metadata_path)}
    return prediction,meta
