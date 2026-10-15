"""Origin-safe H/D/W/MS resampling and common hourly scoring. No May reads."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
TARGET_SOURCE=ROOT/'output/analysis/baseline_audit/complete_hour_module_sum.csv'
TARGET_SHA256='729b149a3e3165dfb339f0386249bfbda0326657f9e722d86da8138db36117de'
SEED=20261013
THREADS=4
POINTS=745
VALID_ORIGINS=pd.to_datetime(['2025-02-15','2025-02-22','2025-02-25'])
INTERNAL_TEST=pd.Timestamp('2025-03-29')
FINAL_ORIGIN=pd.Timestamp('2025-04-30')
FREQUENCIES=['H','D','W-SUN','MS']

def clean_json(value):
    if isinstance(value,dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(v) for v in value]
    if isinstance(value,np.ndarray):return clean_json(value.tolist())
    if isinstance(value,(pd.Timestamp,Path)):return str(value)
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.bool_):return bool(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean_json(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def canonical_frequency(freq):
    aliases={'h':'H','hourly':'H','H':'H','D':'D','daily':'D','W':'W-SUN','weekly':'W-SUN',
             'W-SUN':'W-SUN','M':'MS','ME':'MS','MS':'MS','monthly':'MS'}
    if freq not in aliases:raise ValueError(f'Unsupported frequency {freq}')
    return aliases[freq]

def _check_hourly(y):
    if not isinstance(y,pd.Series) or y.empty:raise ValueError('Nonempty hourly Series required')
    if not isinstance(y.index,pd.DatetimeIndex) or y.index.tz is not None:raise ValueError('Recorded naive localtime index required')
    if not y.index.is_unique or not y.index.is_monotonic_increasing:raise ValueError('Hourly keys must be unique and ordered')
    if len(y)>1 and not y.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all():raise ValueError('Hourly grid has gaps')
    if not y.index.equals(y.index.floor('h')) or not np.isfinite(y.to_numpy(dtype=float)).all():raise ValueError('Invalid hourly timestamps/values')

def load_hourly_target():
    if hashlib.sha256(TARGET_SOURCE.read_bytes()).hexdigest()!=TARGET_SHA256:raise ValueError('Audited target SHA changed')
    frame=pd.read_csv(TARGET_SOURCE,float_precision='round_trip')
    y=pd.Series(frame.provisional_sum_energy_kwh.to_numpy(dtype=np.float64),index=pd.to_datetime(frame.hour),name='energy_kwh')
    y.index.name='hour';_check_hourly(y)
    if not y.index.equals(pd.date_range('2024-12-01','2025-04-29 23:00',freq='h')):raise ValueError('Unexpected3600-hour target')
    y.attrs={'source_sha256':TARGET_SHA256,'unit':'provisional guide kWh per1h bucket','source':'13-module sum; hierarchy unconfirmed'}
    return y

def prefix(y,origin):
    _check_hourly(y);origin=pd.Timestamp(origin)
    if origin.tz is not None:raise ValueError('Origin must use recorded naive localtime')
    hist=y.loc[y.index<origin].copy();_check_hourly(hist)
    if hist.index[-1]+pd.Timedelta(hours=1)!=origin:raise ValueError('Origin must immediately follow observed completed hours')
    hist.attrs=dict(y.attrs);hist.attrs['origin']=str(origin)
    return hist

def _bounds(index,freq):
    index=pd.DatetimeIndex(index);freq=canonical_frequency(freq)
    if freq=='H':start=index.floor('h');end=start+pd.Timedelta(hours=1);label=start
    elif freq=='D':start=index.normalize();end=start+pd.Timedelta(days=1);label=start
    elif freq=='W-SUN':
        start=index.normalize()-pd.to_timedelta(index.dayofweek,unit='D');end=start+pd.Timedelta(days=7);label=start+pd.Timedelta(days=6)
    else:start=index.to_period('M').start_time;end=start+pd.offsets.MonthBegin(1);label=start
    return pd.DatetimeIndex(start),pd.DatetimeIndex(end),pd.DatetimeIndex(label)

def periods(hist,freq,complete_only=True):
    """Reaggregate only the caller's actual past prefix; never reads full-period CSV.

    Index=period_start; weekly display label=Sunday, available_at=next Monday00.
    Rate is energy/observed_hours (kWh/h), never the unequal monthly energy total.
    Complete training rows must have expected count, exact edges and end<=origin.
    """
    _check_hourly(hist);freq=canonical_frequency(freq)
    origin=pd.Timestamp(hist.attrs.get('origin',hist.index[-1]+pd.Timedelta(hours=1)))
    if hist.index[-1]+pd.Timedelta(hours=1)>origin:raise ValueError('Prefix contains observations unavailable at origin')
    start,end,label=_bounds(hist.index,freq)
    values=pd.DataFrame({'period_start':start,'period_end_exclusive':end,'period_label':label,
                         'value':hist.to_numpy(dtype=np.float64),'observed_at':hist.index})
    grouped=values.groupby('period_start',sort=True).agg(period_end_exclusive=('period_end_exclusive','first'),
              period_label=('period_label','first'),energy_kwh=('value','sum'),observed_hours=('value','size'),
              first_observed=('observed_at','min'),last_observed=('observed_at','max'))
    grouped['period_start']=grouped.index
    grouped['expected_hours']=((grouped.period_end_exclusive-grouped.index)/pd.Timedelta(hours=1)).astype(int)
    grouped['rate_kwh_per_hour']=grouped.energy_kwh/grouped.observed_hours
    grouped['coverage']=grouped.observed_hours/grouped.expected_hours
    grouped['available_at']=grouped.period_end_exclusive
    grouped['partial_left']=grouped.first_observed>grouped.index
    grouped['partial_right']=grouped.last_observed+pd.Timedelta(hours=1)<grouped.period_end_exclusive
    grouped['complete']=grouped.observed_hours.eq(grouped.expected_hours)&~grouped.partial_left&~grouped.partial_right
    grouped['available_before_origin']=grouped.available_at.le(origin)
    grouped['frequency']=freq
    # Existing descriptive names remain explicit aliases for interoperability.
    grouped['energy_kwh_provisional']=grouped.energy_kwh
    grouped['mean_hourly_energy_kwh_provisional']=grouped.rate_kwh_per_hour
    grouped['full_period_available_at']=grouped.available_at
    if complete_only:grouped=grouped[grouped.complete&grouped.available_before_origin].copy()
    grouped.attrs={'origin':str(origin),'frequency':freq,'training_filter':'complete and available_at<=origin'}
    return grouped

def future(origin,points=POINTS):return pd.date_range(pd.Timestamp(origin),periods=int(points),freq='h')

def metrics(a,p):
    a,p=np.asarray(a,dtype=np.float64),np.asarray(p,dtype=np.float64)
    if a.shape!=p.shape or a.ndim!=1 or not len(a) or not np.isfinite(a).all() or not np.isfinite(p).all():raise ValueError('Unaligned/nonfinite metrics')
    error=p-a;denom=np.sum((a-a.mean())**2)
    return {'count':len(a),'mae_kwh':float(np.mean(abs(error))),'rmse_kwh':float(np.sqrt(np.mean(error**2))),
            'mape_percent':float(np.mean(abs(error)/abs(a))*100) if np.all(a!=0) else None,
            'bias_pred_minus_actual_kwh':float(error.mean()),'r2':float(1-np.sum(error**2)/denom) if denom>0 else None}

def path_frame(model,frequency,split,fold,origin,idx,actual,pred):
    idx=pd.DatetimeIndex(idx);origin=pd.Timestamp(origin);a=np.asarray(actual,dtype=np.float64);p=np.asarray(pred,dtype=np.float64)
    offsets=((idx-origin)/pd.Timedelta(hours=1)).to_numpy(dtype=int)
    if not idx.equals(future(origin,len(idx))) or a.shape!=p.shape or len(a)!=len(idx) or not np.isfinite(p).all():raise ValueError('Invalid forecast path')
    return pd.DataFrame({'model':model,'frequency':canonical_frequency(frequency),'split':split,'fold':fold,'origin':origin,
                         'id':idx,'horizon_offset_hours':offsets,'lead_hours':offsets+1,
                         'actual_kwh':a,'predicted_kwh':p,'absolute_error_kwh':np.asarray(abs(p-a),dtype=np.float64)})

def score_late505(frame):
    late=frame.loc[frame.horizon_offset_hours.between(240,744)].copy()
    per=[]
    for origin,d in late.groupby('origin',sort=True):
        if len(d)!=505 or not np.array_equal(np.sort(d.horizon_offset_hours),np.arange(240,745)):raise ValueError('Each origin needs505 unique late offsets')
        per.append({'origin':str(origin),**metrics(d.actual_kwh,d.predicted_kwh)})
    if not per:raise ValueError('No labelled Late505 paths')
    return {**metrics(late.actual_kwh,late.predicted_kwh),'mean_origin_mae_kwh':float(np.mean([x['mae_kwh'] for x in per])),
            'origin_count':len(per),'per_origin':per,'unique_label_hours':int(pd.to_datetime(late.id).nunique()),
            'repeated_label_observations':int(len(late)-pd.to_datetime(late.id).nunique())}

def fit_profile(hist,kind='uniform',alpha=1.):
    """Profile of ONLY completed past hourly buckets; positive relative weights.

    Hour/weekday-hour profiles may use past hours inside an unfinished coarse
    period; no complete future coarse statistic or future target is consulted.
    """
    _check_hourly(hist)
    aliases={'global_hour':'hour','hour_of_day':'hour','weekday-hour':'weekday_hour','weekdayhour':'weekday_hour'}
    kind=aliases.get(kind,kind)
    if kind not in ['uniform','hour','weekday_hour'] or not 0<=alpha<=1:raise ValueError('Invalid profile')
    mean=float(hist.mean())
    if mean<=0 or np.any(hist.to_numpy()<=0):raise ValueError('Positive energy required for multiplicative allocation')
    hour=hist.groupby(hist.index.hour).mean().reindex(range(24)).fillna(mean).to_numpy()/mean
    grouping=pd.MultiIndex.from_arrays([hist.index.dayofweek,hist.index.hour],names=['weekday','hour'])
    table=pd.DataFrame({'value':hist.to_numpy()},index=grouping).groupby(level=['weekday','hour']).mean().value
    week=np.asarray([[float(table.get((day,h),hour[h]*mean))/mean for h in range(24)] for day in range(7)])
    origin=pd.Timestamp(hist.attrs.get('origin',hist.index[-1]+pd.Timedelta(hours=1)))
    return {'kind':kind,'alpha':float(alpha),'hour_relative_weights':hour.tolist(),'weekday_hour_relative_weights':week.tolist(),
            'fit_start':str(hist.index[0]),'fit_end':str(hist.index[-1]),'available_at':str(hist.index[-1]+pd.Timedelta(hours=1)),
            'origin':str(origin),'fit_hours':len(hist),'source_mean_kwh':mean,
            'unit':'Dimensionless relative within-period allocation; predicted rate remains kWh/h',
            'no_future_measurements':True}

def profile_weights(idx,profile):
    idx=pd.DatetimeIndex(idx);kind=profile['kind'];alpha=float(profile['alpha'])
    if kind=='uniform':raw=np.ones(len(idx))
    elif kind=='hour':raw=np.asarray(profile['hour_relative_weights'])[idx.hour]
    else:raw=np.asarray(profile['weekday_hour_relative_weights'])[idx.dayofweek,idx.hour]
    weights=(1-alpha)+alpha*raw
    if not np.isfinite(weights).all() or np.any(weights<=0):raise ValueError('Invalid allocation weights')
    return weights.astype(np.float64)

def forecast_period_index(origin,points,freq):
    starts,ends,labels=_bounds(future(origin,points),freq)
    return pd.DataFrame({'period_start':starts,'period_end_exclusive':ends,'period_label':labels}).drop_duplicates('period_start').set_index('period_start',drop=False)

def allocate_rate(rates,idx,freq,profile=None):
    """Map predicted period RATE to requested hours; conserve full-calendar energy.

    Series rates indexed by period_start, scalar, or array in chronological
    unique-period order. Full period energy=rate*expected_hours. An incomplete
    query reports only its intersection, not the full period total. For an
    ongoing period, this does not force known past + forecast remaining energy
    to equal a hypothetical whole-period forecast; no past residual correction.
    """
    idx=pd.DatetimeIndex(idx);starts,ends,_=_bounds(idx,freq);unique=pd.DatetimeIndex(starts.unique()).sort_values()
    if np.isscalar(rates):rate=pd.Series(float(rates),index=unique)
    elif isinstance(rates,pd.Series):rate=rates.reindex(unique)
    else:
        values=np.asarray(rates,dtype=np.float64)
        if len(values)!=len(unique):raise ValueError('Rate count must match intersected calendar periods')
        rate=pd.Series(values,index=unique)
    if rate.isna().any() or not np.isfinite(rate.to_numpy()).all():raise ValueError('Missing period rate')
    prediction=np.empty(len(idx),dtype=np.float64)
    for start in unique:
        mask=starts==start;end=ends[mask][0];all_hours=pd.date_range(start,end-pd.Timedelta(hours=1),freq='h')
        weights=np.ones(len(all_hours)) if profile is None else profile_weights(all_hours,profile)
        all_pred=float(rate.loc[start])*len(all_hours)*weights/weights.sum()
        positions=((idx[mask]-start)/pd.Timedelta(hours=1)).to_numpy(dtype=int)
        prediction[mask]=all_pred[positions]
    return prediction

def protocol_manifest():
    paths=[future(o) for o in VALID_ORIGINS];late=[p[240:] for p in paths]
    flat=pd.DatetimeIndex(np.concatenate([p.values for p in paths]));flat_late=pd.DatetimeIndex(np.concatenate([p.values for p in late]))
    test=future(INTERNAL_TEST);overlap=flat.unique().intersection(test)
    return {'target_source':str(TARGET_SOURCE),'target_sha256':TARGET_SHA256,'hour_count':3600,
            'step_hours':1,'forecast_points':POINTS,'offsets':'0..744','leads_from_last_observation':'1..745',
            'selection_offsets':'240..744','selection_points_per_origin':505,'selection_objective':'Mean3-origin Late505 MAE; no internal-test/May tuning',
            'validation_origins':[str(o) for o in VALID_ORIGINS],'validation_target_end':str(max(p[-1] for p in paths)),
            'validation_path_observations':len(flat),'validation_path_unique_hours':flat.nunique(),
            'validation_late_observations':len(flat_late),'validation_late_unique_hours':flat_late.nunique(),
            'validation_repeated_late_labels':len(flat_late)-flat_late.nunique(),
            'validation_independence_limit':'Three overlapping fixed-origin paths are not three independent samples; repeated future labels must not inflate reliability claims.',
            'internal_origin':str(INTERNAL_TEST),'internal_end':str(test[-1]),'validation_internal_label_overlap_hours':len(overlap),
            'all_selection_labels_available_before_internal_origin':max(p[-1]+pd.Timedelta(hours=1) for p in paths)<=INTERNAL_TEST,
            'final_origin':str(FINAL_ORIGIN),'final_end':str(future(FINAL_ORIGIN)[-1]),
            'final_evaluation_start':str(future(FINAL_ORIGIN)[240]),'final_label_read_policy':'No May CSV is read by common/data/model development; parent evaluates after locks',
            'rejected_design_note':'Initial lastvalidationoriginFeb28 was rejected before scoring because its labels reachedMarch31 and overlappedMar29 internal test49hours; final lastoriginFeb25 fixes label overlap and selection-time availability.',
            'period_training_filter':'Reaggregate actual prefix; complete and available_at<=origin. WeeklySunday label is not its availability, which is Monday00; monthlynextMonth00.',
            'monthly_limit':'Only2 full monthly periods at each validation origin and4 at final; monthly methods are explicit statistical baselines, not trained high-capacity SOTA.',
            'unit_limit':'13-meter provisional guide kWh sum; unknown hierarchy/unit/circuit scope. Rate normalization separates calendar length from load.',
            'post_EDA_limit':'Earlier April/May EDA occurred; this is a controlled subsequent experiment, not a pristine prospective evaluation.'}
