"""Replay-safe prefix-only forecasts. This API never loads an April checkpoint.

Guide745 model comparison is a separate revised historical experiment. The
online demo uses an explicitly named independently fitted historical baseline,
with recent past replay metrics and no invented calibrated interval.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import json
OUT=Path(__file__).resolve().parent

def _metric(a,p):
    a,p=np.asarray(a,float),np.asarray(p,float);e=p-a;den=np.sum((a-a.mean())**2)
    return {'count':len(a),'mae_kwh':float(np.mean(abs(e))),'rmse_kwh':float(np.sqrt(np.mean(e**2))),
            'r2':float(1-np.sum(e**2)/den) if den>0 else None,'bias_pred_minus_actual_kwh':float(e.mean())}

def serving_forecast(series,origin=None,points=24,module_id=None):
    if not isinstance(series,pd.Series) or not isinstance(series.index,pd.DatetimeIndex):
        raise ValueError('A pandas Series indexed by recorded naive hourly timestamps is required')
    if series.index.tz is not None:raise ValueError('Timezone-aware keys require an explicit source-clock mapping outside this API')
    if not series.index.is_unique or not series.index.is_monotonic_increasing:raise ValueError('Duplicate/unordered hourly timestamps')
    points=int(points)
    if not 1<=points<=745:raise ValueError('Forecast points must be1..745')
    requested_origin=pd.Timestamp(origin) if origin is not None else series.index[-1]+pd.Timedelta(hours=1)
    if requested_origin.tzinfo is not None:raise ValueError('Origin must use the same naive source clock as history')
    origin=requested_origin.floor('h')
    h=series.loc[series.index<origin].astype(float).copy()
    if h.empty or h.index[-1]+pd.Timedelta(hours=1)!=origin:raise ValueError('Need the last completed hour immediately before forecast origin')
    if not h.index.equals(pd.date_range(h.index[0],periods=len(h),freq='h')):raise ValueError('Historical grid is not contiguous hourly data')
    if not np.isfinite(h.to_numpy()).all() or (h<0).any():raise ValueError('Historical energy must be finite and nonnegative')
    level=float(h.mean());future=pd.date_range(origin,periods=points,freq='h')
    cold=len(h)<168;metrics=None
    # Seven observed24h paths, fitted independently at each earlier origin.
    # Actuals for these paths are already known at this requested origin.
    replay_origins=[origin-pd.Timedelta(days=days) for days in range(7,0,-1)]
    actual=[];pred=[];valid_origins=[]
    for o in replay_origins:
        fit=h.loc[h.index<o];idx=pd.date_range(o,periods=24,freq='h')
        if len(fit)>=24 and idx.isin(h.index).all():
            actual.extend(h.reindex(idx).to_list());pred.extend([float(fit.mean())]*24);valid_origins.append(str(o))
    if actual:
        metrics={**_metric(actual,pred),'scope':'Recent already-observed24h replay, prefix-only refit per origin',
                 'origin_count':len(valid_origins),'origins':valid_origins,'forecast_horizon_hours':24,
                 'uses_current_future_labels':False,'guarantees_next_path_accuracy':False}
    selected=None
    if (OUT/'results.json').exists():selected=json.loads((OUT/'results.json').read_text(encoding='utf8')).get('selected_model')
    return {'status':'cold_start_baseline' if cold else 'ready','model_name':'historical_prefix_mean_baseline',
      'model_display_name':'해당 시점까지의 과거 평균 기준선','forecast_quality':'cold_start_unknown' if cold else 'recent_past_replay_only',
      'quality_note':'각 호출의 완료된 과거 구간으로 독립 재적합합니다. 향후 정확도와 모듈별 예측 성능은 보장되지 않습니다.',
      'deployment_role':'Replay-safe online baseline; separate from guide745 selected experimental model',
      'guide_selected_experiment_model':selected,'origin':str(origin),'requested_origin':str(requested_origin),
      'points':points,'unit':'provisional_kWh','clock':'Recorded source naive hourly clock; no implicit timezone conversion',
      'history_end':str(h.index[-1]),'history_count':len(h),'training_cutoff':str(h.index[-1]),
      'uses_only_origin_history':True,'module_id':str(module_id) if module_id is not None else 'all',
      'scope':'independently_fitted_single_module' if module_id is not None and str(module_id) not in ['all','aggregate'] else 'aggregate_13_module_sum_hierarchy_unconfirmed',
      'module_model_is_total_proportional_allocation':False,'physical_unit_rescaling':False,'metrics':metrics,
      'normalizer':{'center_kwh':level,'fit_count':len(h),'fit_end':str(h.index[-1]),'kind':'prefix mean, no target rescale'},
      'interval':None,'interval_note':'검증된 미래 예측구간을 제공하지 않습니다.',
      'forecast':[{'timestamp':str(t),'predicted_kwh':level,'lower_kwh':None,'upper_kwh':None} for t in future]}

def guide_forecast():
    if not (OUT/'results.json').exists():return {'status':'training_in_progress'}
    r=json.loads((OUT/'results.json').read_text(encoding='utf8'));selected=r['models'][r['selected_model']]
    f=pd.read_csv(selected['fits']['future_unlabelled']['prediction_csv'],float_precision='round_trip')
    return {'status':'ready','model_name':r['selected_model'],'role':'Fixed guide experiment, source cutoffApril29 23',
            'unit':'provisional_kWh','origin':'2025-04-30 00:00:00','selection':'Three CV origin Late505 MAE only',
            'forecast':[{'timestamp':str(t),'predicted_kwh':float(p)} for t,p in zip(f.id,f.predicted_kwh)],
            'quality_note':r.get('interpretation_limits',[])}
