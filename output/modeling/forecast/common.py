"""Common hourly target, chronological splits, forecasts, and scoring protocol.

No future sensor measurements or May labels are used to build model features.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'output' / 'modeling' / 'forecast'
SOURCE = ROOT / 'output' / 'analysis' / 'baseline_audit' / 'complete_hour_module_sum.csv'
AUDIT = ROOT / 'output' / 'analysis' / 'time_anomaly_audit.json'
MAY = ROOT / 'may-dataset' / 'rtu_ground_truth_may.csv'
HORIZON_POINTS = {'H96':96,'H192':192,'H336':336,'H720':720,'P745':745}
MAX_HORIZON = 744
VALIDATION_ORIGIN = pd.Timestamp('2025-02-25 00:00:00')
TEST_ORIGIN = pd.Timestamp('2025-03-29 00:00:00')
MIN_HISTORY_HOURS = 40*24
PREDICTION_COLUMNS = ['model','split','origin','id','horizon_offset_hours','lead_hours','actual_kwh','predicted_kwh','absolute_error_kwh']

def json_safe(obj):
    if isinstance(obj,dict):
        return {str(k):json_safe(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj,(np.integer,)):
        return int(obj)
    if isinstance(obj,(np.floating,float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj,(pd.Timestamp,Path)):
        return str(obj)
    if isinstance(obj,np.ndarray):
        return json_safe(obj.tolist())
    return obj

def write_json(path,obj):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(json_safe(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def load_training():
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    for equipment,check in audit['time_key_checks'].items():
        for key in ['local_duplicate_rows','local_negative_diffs','local_non_5s_diffs']:
            if check[key] != 0:
                raise ValueError(f'Localtime completeness audit failed: {equipment} {key}')
    raw = pd.read_csv(SOURCE)
    # The analysis artifact contains one timestamp key and one energy target.
    target_col = 'provisional_sum_energy_kwh'
    time_col = raw.columns[0]
    series = pd.Series(raw[target_col].to_numpy(dtype=float),index=pd.to_datetime(raw[time_col]),name=target_col).sort_index()
    if not series.index.is_unique or not np.isfinite(series.to_numpy()).all():
        raise ValueError('Nonunique timestamps or nonfinite energy targets')
    expected = pd.date_range(series.index.min(),series.index.max(),freq='h')
    if not series.index.equals(expected):
        raise ValueError('Complete aggregate series contains an hourly gap')
    if len(series) != 3600:
        raise ValueError(f'Expected 3600 complete all-module hours, got {len(series)}')
    if series.index.max() != pd.Timestamp('2025-04-29 23:00:00'):
        raise ValueError('Training end changed; update protocol explicitly before running')
    return series

def future_index(origin,horizon=MAX_HORIZON):
    # h=0 is the first future unobserved hour; h=H is included.
    return pd.date_range(pd.Timestamp(origin),periods=int(horizon)+1,freq='h')

def history_before(series,origin):
    history = series.loc[series.index < pd.Timestamp(origin)].copy()
    if len(history)<MIN_HISTORY_HOURS:
        raise ValueError('Expanding history is shorter than 40 days')
    if history.index.max()+pd.Timedelta(hours=1) != pd.Timestamp(origin):
        raise ValueError('Forecast origin does not immediately follow the observed history')
    return history

def actual_at(series,future):
    target = series.reindex(future)
    if target.isna().any():
        raise ValueError('Internal validation/test horizon includes unavailable labels')
    return target

def score(actual,predicted):
    actual = np.asarray(actual,dtype=float)
    predicted = np.asarray(predicted,dtype=float)
    if actual.shape != predicted.shape or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Unaligned or nonfinite evaluation arrays')
    err = actual-predicted
    return {'count':len(actual),'mae_kwh':float(np.abs(err).mean()),
            'rmse_kwh':float(np.sqrt(np.mean(err**2))),
            'mape_percent':float(np.mean(np.abs(err)/np.abs(actual))*100),
            'bias_pred_minus_actual_kwh':float((predicted-actual).mean())}

def score_prefixes(actual,predicted):
    return {name:score(np.asarray(actual)[:points],np.asarray(predicted)[:points]) for name,points in HORIZON_POINTS.items()}

def baseline_predict(name,history,future):
    origin = future[0]
    if history.index.max() >= origin:
        raise ValueError('Baseline history includes future observations')
    mean = float(history.mean())
    if name == 'mean':
        return np.repeat(mean,len(future))
    if name == 'median':
        return np.repeat(float(history.median()),len(future))
    if name == 'last_week_mean':
        return np.repeat(float(history.iloc[-168:].mean()),len(future))
    if name == 'hour_mean':
        mapping = history.groupby(history.index.hour).mean()
        return np.array([mapping.get(t.hour,mean) for t in future],dtype=float)
    if name == 'weekday_hour_mean':
        mapping = history.groupby([history.index.dayofweek,history.index.hour]).mean()
        return np.array([mapping.get((t.dayofweek,t.hour),mean) for t in future],dtype=float)
    if name in ['seasonal_24','seasonal_168']:
        lag = 24 if name == 'seasonal_24' else 168
        reference = [origin-pd.Timedelta(hours=lag)+pd.Timedelta(hours=h%lag) for h in range(len(future))]
        predicted = history.reindex(reference).to_numpy(dtype=float)
        if not np.isfinite(predicted).all():
            raise ValueError('Last observed seasonal template has a missing hour')
        return predicted
    raise ValueError(name)

def prediction_frame(model,split,origin,future,actual,predicted):
    offsets = ((pd.DatetimeIndex(future)-pd.Timestamp(origin)).total_seconds()/3600).astype(int)
    return pd.DataFrame({'model':model,'split':split,'origin':origin,'id':future,
                         'horizon_offset_hours':offsets,'lead_hours':offsets+1,
                         'actual_kwh':np.asarray(actual,dtype=float),
                         'predicted_kwh':np.asarray(predicted,dtype=float),
                         'absolute_error_kwh':np.abs(np.asarray(actual,dtype=float)-np.asarray(predicted,dtype=float))})

def protocol(series):
    valid = future_index(VALIDATION_ORIGIN)
    test = future_index(TEST_ORIGIN)
    if valid[-1] >= test[0]:
        raise ValueError('Validation and internal test target periods overlap')
    for origin in [VALIDATION_ORIGIN,TEST_ORIGIN]:
        history_before(series,origin)
        actual_at(series,future_index(origin))
    return {
        'target':'Provisional simultaneous 13-module aggregate hourly energy, kWh',
        'target_source':str(SOURCE),'target_source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'target_formula':'sum of activePower(kW)*5/3600 over 720 unique five-second samples per module, then sum over all 13 modules',
        'meter_hierarchy_limitation':'Unknown parent/child meter hierarchy can double count energy; not certified whole-facility consumption',
        'time_key':'Recorded localtime with exhaustive unique continuous five-second grid audit; supplied Unix timestamps have a March 9 offset transition inconsistent with Korean-time guide',
        'training_start':str(series.index.min()),'training_end':str(series.index.max()),'training_complete_hours':len(series),
        'minimum_evaluation_history_hours':MIN_HISTORY_HOURS,
        'validation_origin':str(VALIDATION_ORIGIN),'validation_history_hours':len(history_before(series,VALIDATION_ORIGIN)),
        'validation_target_start':str(valid[0]),'validation_target_end':str(valid[-1]),
        'test_origin':str(TEST_ORIGIN),'test_history_hours':len(history_before(series,TEST_ORIGIN)),
        'test_target_start':str(test[0]),'test_target_end':str(test[-1]),
        'target_periods_overlap':False,
        'horizon_convention':'Origin offset h=0 is the first unobserved hour, equivalent to lead=1 relative to the last training observation. Conventional H96/H192/H336/H720 evaluate exactly H values at offsets 0..H-1; full task P745 evaluates offsets 0..744.',
        'evaluated_horizon_points':HORIZON_POINTS,'fixed_forecast_path_rows':MAX_HORIZON+1,
        'prediction_csv_columns':PREDICTION_COLUMNS,
        'features':'Observed target history and deterministically known calendar only; no future current, voltage, power factor, operation, power, or energy measurements',
        'selection':'Hyperparameters are selected only on blocked chronological validation. The internal test and May holdout are evaluated after frozen selection; no tuning uses those labels.',
        'honest_holdout_disclosure':'May labels were inspected during earlier descriptive EDA. This experiment is therefore a predeclared post-EDA lock, not a never-seen prospective holdout. No May-label-based hyperparameter tuning is allowed.',
        'may_origin':str(series.index.max()+pd.Timedelta(hours=1)),
        'may_full_path':'Apr30 00 h=0 through May31 00 h=744: 745 hourly predictions including the unlabelled bridge',
        'may_scoring':'Score only the supplied May10 00 through May31 00 labelled 505 rows, after selection and full-history refit',
        'development_origins_not_used_for_selection':[str(series.index[960]),str(series.index[1128]),str(series.index[1296])],
        'reserved_unscored_internal_tail_hours':int((series.index>test[-1]).sum()),
        'scale_note':'May hourly_pow CSV lacks an explicit unit; comparison assumes its scale and aggregate scope match derived hourly kWh and reports that assumption',
    }

def may_labels_after_lock():
    frame = pd.read_csv(MAY,usecols=['id','hourly_pow'])
    frame['id'] = pd.to_datetime(frame['id'])
    frame = frame.sort_values('id')
    if len(frame)!=505 or frame['id'].duplicated().any():
        raise ValueError('May labelled target shape changed')
    return frame.set_index('id')['hourly_pow']

if __name__ == '__main__':
    training = load_training()
    manifest = protocol(training)
    write_json(OUT/'protocol.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
