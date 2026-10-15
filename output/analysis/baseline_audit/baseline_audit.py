"""Fixed-origin baselines selected on April validation before reading May targets.

Only the small derived hourly file is read. Raw multi-GB sensor CSV is untouched.
The module sum is provisional: parent/child meter hierarchy is not confirmed.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(r'E:\AI work\AWS_프로젝트')
BASE = ROOT / 'output' / 'analysis'
DEST = BASE / 'baseline_audit'
DEST.mkdir(parents=True, exist_ok=True)
HOURLY = BASE / 'hourly_by_equipment.csv'
PROFILE = BASE / 'training_profile.json'
TIME_AUDIT = BASE / 'time_anomaly_audit.json'
MAY = ROOT / 'may-dataset' / 'rtu_ground_truth_may.csv'

def forecast(name, history, future, origin):
    """All forecasts use only observations strictly before origin."""
    if not (history.index < origin).all():
        raise ValueError('History contains observations at or after forecast origin')
    mean = float(history.mean())
    if name == 'historical_mean':
        return np.repeat(mean, len(future))
    if name == 'historical_median':
        return np.repeat(float(history.median()), len(future))
    if name == 'last_7_days_mean':
        recent = history[history.index >= origin-pd.Timedelta(days=7)]
        return np.repeat(float(recent.mean()), len(future))
    if name == 'hour_of_day_mean':
        mapping = history.groupby(history.index.hour).mean()
        return np.array([mapping.get(t.hour, mean) for t in future],dtype=float)
    if name == 'weekday_hour_mean':
        mapping = history.groupby([history.index.dayofweek,history.index.hour]).mean()
        return np.array([mapping.get((t.dayofweek,t.hour), mean) for t in future],dtype=float)
    if name in ['seasonal_naive_24h_fixed','seasonal_naive_168h_fixed']:
        lag = 24 if name == 'seasonal_naive_24h_fixed' else 168
        # Repeat the last pre-origin day/week recursively; no future actuals.
        hour_offsets = ((future-origin).total_seconds()/3600).astype(int)
        references = [origin-pd.Timedelta(hours=lag)+pd.Timedelta(hours=int(x%lag)) for x in hour_offsets]
        return history.reindex(pd.DatetimeIndex(references)).to_numpy(dtype=float)
    raise ValueError(name)

def score(actual, predicted):
    actual = np.asarray(actual,dtype=float)
    predicted = np.asarray(predicted,dtype=float)
    if np.isfinite(predicted).all():
        error = actual-predicted
        return {'mae':float(np.abs(error).mean()),'rmse':float(np.sqrt(np.mean(error**2))),
                'mape_percent':float(np.mean(np.abs(error)/np.abs(actual))*100),
                'prediction_count':int(len(predicted)), 'complete':True}
    return {'mae':None,'rmse':None,'mape_percent':None,
            'prediction_count':int(np.isfinite(predicted).sum()),'complete':False}

if not HOURLY.exists() or not PROFILE.exists() or not TIME_AUDIT.exists():
    raise SystemExit('Requires hourly_by_equipment.csv, training_profile.json, and exhaustive time_anomaly_audit.json')
profile = json.loads(PROFILE.read_text(encoding='utf-8'))
time_audit = json.loads(TIME_AUDIT.read_text(encoding='utf-8'))
hourly = pd.read_csv(HOURLY)
hourly['hour'] = pd.to_datetime(hourly['hour'])
equipment_col = 'module(equipment)'
expected_equipment = sorted(profile['profiles'].keys())
if set(hourly[equipment_col].astype(str)) != set(expected_equipment):
    raise ValueError('Hourly equipment set disagrees with training profile')
if hourly.duplicated([equipment_col,'hour']).any():
    raise ValueError('Duplicate equipment-hour rows')
# The independent exhaustive audit confirms localtime is a unique, continuous
# five-second grid, despite anomalies in the supplied Unix timestamp key.
sampling_flags = {m:{k:p.get(k) for k in ['zero_diffs','negative_time_diffs','offgrid']} for m,p in profile['profiles'].items()}
local_checks = time_audit['time_key_checks']
if set(local_checks) != set(expected_equipment):
    raise ValueError('Exhaustive localtime audit equipment set disagrees with hourly aggregation')
if int(time_audit['rows']) != int(profile['records']):
    raise ValueError('Exhaustive localtime audit row count disagrees with training profile')
for module,checks in local_checks.items():
    for key in ['local_duplicate_rows','local_negative_diffs','local_non_5s_diffs']:
        if checks[key] != 0:
            raise ValueError(f'Localtime audit failed for {module}: {key}={checks[key]}')
if len({checks['local_origin'] for checks in local_checks.values()}) != 1 or len({checks['local_last'] for checks in local_checks.values()}) != 1:
    raise ValueError('Modules do not share identical localtime observation windows')
hourly['recomputed_energy_kwh'] = hourly['power_sum']*5.0/3600.0
energy_conversion_max_abs_diff = float((hourly['recomputed_energy_kwh']-hourly['energy_kwh']).abs().max())
complete = hourly.loc[hourly['records'].eq(720)]
module_counts = complete.groupby('hour')[equipment_col].nunique()
eligible_hours = module_counts.index[module_counts.eq(len(expected_equipment))]
eligible = complete.loc[complete['hour'].isin(eligible_hours)]
series = eligible.groupby('hour')['recomputed_energy_kwh'].sum().sort_index()
series.name = 'provisional_sum_energy_kwh'
series.to_csv(DEST/'complete_hour_module_sum.csv',encoding='utf-8-sig')
if not series.index.is_unique:
    raise ValueError('Duplicate provisional sum times')
april_series = series[series.index < pd.Timestamp('2025-05-01 00:00:00')]
april_end_exclusive = april_series.index.max()+pd.Timedelta(hours=1)
validation_origin = april_end_exclusive-pd.Timedelta(days=14)
history = series[series.index < validation_origin]
validation = series[(series.index >= validation_origin)&(series.index < april_end_exclusive)]
models = ['historical_mean','historical_median','last_7_days_mean','hour_of_day_mean',
          'weekday_hour_mean','seasonal_naive_24h_fixed','seasonal_naive_168h_fixed']
if history.empty or validation.empty:
    raise ValueError('Insufficient pre-April-validation history or validation targets')
scores = {}
val_predictions = pd.DataFrame({'id':validation.index,'actual_provisional_sum_kwh':validation.to_numpy()})
for model in models:
    pred = forecast(model,history,validation.index,validation_origin)
    val_predictions[model] = pred
    scores[model] = score(validation.to_numpy(),pred)
winner = min((m for m in models if scores[m]['complete']),key=lambda m:scores[m]['mae'])
if winner != 'weekday_hour_mean':
    raise ValueError('April validation selection changed from the previously established weekday_hour_mean winner')
val_predictions.to_csv(DEST/'april_validation_predictions.csv',index=False,encoding='utf-8-sig')

# Winner is frozen above; May target values have not been read before selection.
may = pd.read_csv(MAY)
may['id'] = pd.to_datetime(may['id'])
may = may.sort_values('id')
if len(may) != 505:
    raise ValueError('May evaluation horizon is not the expected 505 hours')
all_training = series[series.index < april_end_exclusive]
# The fixed forecast origin is the first hour after the complete training data.
forecast_origin = april_end_exclusive
future = pd.DatetimeIndex(may['id'])
chosen_forecast = forecast(winner,all_training,future,forecast_origin)
may_predictions = pd.DataFrame({'id':future,'actual_hourly_pow':may['hourly_pow'].to_numpy(),
                                'selected_model':winner,'forecast_provisional_sum_kwh':chosen_forecast})
may_predictions['absolute_error'] = np.abs(may_predictions['actual_hourly_pow']-may_predictions['forecast_provisional_sum_kwh'])
may_predictions.to_csv(DEST/'may_selected_baseline_predictions.csv',index=False,encoding='utf-8-sig')
mean_power = float(all_training.mean())
result = {
    'source_hourly':str(HOURLY), 'source_may':str(MAY),
    'aggregation':'sum over modules of sum(activePower kW)*5/3600 kWh; only module hours with exactly 720 records, and only hours where every listed module is present',
    'hierarchy_limitation':'Meter hierarchy is unconfirmed; module sums may double count parent and child equipment. Training aggregation and May target scope may differ.',
    'target_unit_limitation':'Guide says activePower is kW; training hourly energy is derived in kWh. May hourly_pow column itself has no explicit unit in CSV; comparison presumes same energy scale and scope.',
    'expected_equipment':expected_equipment,'expected_equipment_count':len(expected_equipment),
    'raw_timestamp_sampling_flags':sampling_flags,
    'source_time_audit':str(TIME_AUDIT),
    'localtime_grid_audit_status':'Verified by exhaustive raw-row audit: every module has zero localtime duplicate rows, negative differences, and non-five-second differences; all modules share identical localtime start and end',
    'localtime_grid_audit_rows':int(time_audit['rows']),
    'localtime_grid_audit_checks':local_checks,
    'time_key_limitation':'Forecasting uses recorded localtime. Unix timestamps contain 720 duplicate rows per module around the March 9 offset transition from local-minus-Unix -8 hours to -7 hours, inconsistent with the guide describing Korean time.',
    'energy_conversion_max_abs_diff':energy_conversion_max_abs_diff,
    'equipment_hour_rows':int(len(hourly)),'complete_equipment_hour_rows':int(len(complete)),
    'eligible_all_module_hours':int(len(series)),
    'excluded_partial_or_missing_module_hours':int(hourly['hour'].nunique()-len(series)),
    'training_start':str(all_training.index.min()),'training_end':str(all_training.index.max()),
    'all_training_count':int(len(all_training)),
    'all_training_mean_provisional_kwh':mean_power,
    'missing_all_module_training_hours':[str(t) for t in pd.date_range(all_training.index.min(),all_training.index.max(),freq='h').difference(all_training.index)],
    'selection':'Lowest MAE on fixed-origin validation consisting of the last 14 complete days available in April; May actual values are read only after selecting winner',
    'validation':{'forecast_origin':str(validation_origin),'history_start':str(history.index.min()),'history_end':str(history.index.max()),
                  'history_count':int(len(history)),'target_start':str(validation.index.min()),'target_end':str(validation.index.max()),
                  'target_count':int(len(validation)),'expected_target_count':336,
                  'missing_validation_hours':[str(t) for t in pd.date_range(validation_origin,april_end_exclusive-pd.Timedelta(hours=1),freq='h').difference(validation.index)],
                  'scores':scores,'selected_model':winner},
    'may_selected_baseline':{'forecast_origin':str(forecast_origin),'selected_model':winner,'evaluation_start':str(future.min()),
                             'evaluation_end':str(future.max()),'evaluation_count':len(future),
                             'first_horizon_hours':float((future.min()-forecast_origin).total_seconds()/3600),
                             'last_horizon_hours':float((future.max()-forecast_origin).total_seconds()/3600),
                             'scores':score(may['hourly_pow'].to_numpy(),chosen_forecast),
                             'forecast_mean':float(np.nanmean(chosen_forecast)),
                             'may_actual_mean':float(may['hourly_pow'].mean()),
                             'mean_scope_ratio_actual_to_forecast':float(may['hourly_pow'].mean()/np.nanmean(chosen_forecast))},
    'forecast_protocol':'Every future prediction is fixed from pre-origin training. Seasonal baselines repeat the last training day/week and never consume future actual observations.',
}
(DEST/'baseline_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['expected_equipment','eligible_all_module_hours','excluded_partial_or_missing_module_hours','validation','may_selected_baseline']},ensure_ascii=False,indent=2))
