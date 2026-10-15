"""Single read of original five-second CSV; write only this revision directory."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import json, hashlib, time, warnings
import numpy as np
import pandas as pd
warnings.simplefilter('ignore',pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'output/feature_revision_20261013'
RAW = ROOT/'traing-dataset/rtu_data_full.csv'
OLD = ROOT/'output/analysis/hourly_by_equipment.csv'
TARGET = ROOT/'output/analysis/baseline_audit/complete_hour_module_sum.csv'
SENSORS = ['voltageR','voltageS','voltageT','voltageRS','voltageST','voltageTR',
           'currentR','currentS','currentT','powerFactorR','powerFactorS','powerFactorT',
           'reactivePowerLagging','activePower']
DERIVED = ['energy_kwh','std_power_kw','pf_mean','voltage_unbalance_mean',
           'current_unbalance_mean','counter_delta']

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    started=time.perf_counter(); (OUT/'data').mkdir(parents=True,exist_ok=True)
    original_stat=RAW.stat()
    old=pd.read_csv(OLD,float_precision='round_trip')
    old['hour']=pd.to_datetime(old['hour'])
    old['module_id']=old['module(equipment)'].str.extract(r'^(\d+)').astype(int)
    labels=old[['module_id','module(equipment)']].drop_duplicates().sort_values('module_id')
    ids=labels.module_id.tolist(); module_pos={label:i for i,label in enumerate(labels['module(equipment)'])}
    hours=pd.date_range('2024-12-01 00:00','2025-04-29 23:00',freq='h')
    nh=len(hours); groups=len(ids)*nh
    hour_codes=hours.strftime('%Y%m%d%H').astype(np.int64).to_numpy()
    counts=np.zeros(groups,dtype=np.int64)
    sums=np.zeros((len(SENSORS),groups),dtype=np.float64)
    squares=np.zeros_like(sums)
    rows=0; excluded=0; chunks=0
    usecols=['module(equipment)','localtime']+SENSORS
    dtypes={c:'float64' for c in SENSORS}; dtypes.update({'localtime':'int64','module(equipment)':'category'})
    for d in pd.read_csv(RAW,usecols=usecols,dtype=dtypes,chunksize=500_000):
        chunks+=1; rows+=len(d)
        local=d.localtime.to_numpy(); code=local//10000
        hi=np.searchsorted(hour_codes,code)
        keep=hi<nh
        if np.any(keep & (hour_codes[np.minimum(hi,nh-1)]!=code)):
            raise ValueError('Unexpected hour not in the audited complete grid')
        if np.any(local[~keep]!=20250430000000):
            raise ValueError('Only the known Apr30 endpoint may be excluded')
        excluded+=int((~keep).sum())
        mp=d['module(equipment)'].astype(object).map(module_pos).to_numpy()
        if pd.isna(mp).any():raise ValueError('Unexpected module')
        key=mp[keep].astype(np.int64)*nh+hi[keep]
        counts+=np.bincount(key,minlength=groups)
        for j,c in enumerate(SENSORS):
            values=d[c].to_numpy()[keep]
            if not np.isfinite(values).all():raise ValueError(f'Missing/nonfinite sensor: {c}')
            sums[j]+=np.bincount(key,weights=values,minlength=groups)
            squares[j]+=np.bincount(key,weights=values*values,minlength=groups)
        if chunks%10==0:
            print(json.dumps({'stage':'raw_statistics','rows':rows,'elapsed_seconds':round(time.perf_counter()-started,2)}),flush=True)
    if rows!=33_696_013 or excluded!=13 or not np.all(counts==720):
        raise ValueError(f'Unexpected completeness {rows=}, {excluded=}, {counts.min()=}, {counts.max()=}')
    means=sums/counts
    stds=np.sqrt(np.maximum(squares/counts-means*means,0))
    complete=old[old.complete_720.eq(True)].copy()
    if len(complete)!=groups or complete.duplicated(['module_id','hour']).any():
        raise ValueError('Old complete hourly keys disagree')
    complete['counter_delta']=complete.counter_last-complete.counter_first
    feature=pd.DataFrame(index=hours); feature.index.name='hour'
    catalog=[]
    unit={c:('V' if c.startswith('voltage') else 'A' if c.startswith('current') else
             '%' if c.startswith('powerFactor') else 'kW (guide; conditional)' if c=='activePower' else 'kvar (guide; unverified)') for c in SENSORS}
    family=lambda c: 'voltage' if c.startswith('voltage') else 'current' if c.startswith('current') else 'power_factor' if c.startswith('powerFactor') else 'reactive_power' if c=='reactivePowerLagging' else 'active_power'
    max_power_mean_diff=0.; max_power_std_diff=0.
    for i,m in enumerate(ids):
        q=complete[complete.module_id.eq(m)].set_index('hour').reindex(hours)
        if q.records.isna().any():raise ValueError('Missing old module hour')
        for j,c in enumerate(SENSORS):
            for stat,values in [('mean',means[j,i*nh:(i+1)*nh]),('std',stds[j,i*nh:(i+1)*nh])]:
                name=f'm{m}__{c}_{stat}'; feature[name]=values
                catalog.append({'name':name,'module_id':m,'source_column':c,'aggregation':stat,
                                'std_ddof':0 if stat=='std' else None,'unit':unit[c],'family':family(c),
                                'interval':'[hour,hour+1h),720 observations','available_at':'hour+1h',
                                'forecast_use':'Use completed hours strictly before the forecast origin; lag1 at origin t uses t-1h',
                                'same_hour_leakage':True,'physical_candidate':c!='activePower',
                                'duplicate_of':f'm{m}__energy_kwh' if c=='activePower' and stat=='mean' else f'm{m}__std_power_kw' if c=='activePower' else None})
        for c in DERIVED:
            name=f'm{m}__{c}';feature[name]=q[c].to_numpy(dtype=np.float64)
            catalog.append({'name':name,'module_id':m,'source_column':'counter_last-counter_first' if c=='counter_delta' else c,
                            'aggregation':'existing complete hourly derived statistic','unit':'kWh (provisional)' if c=='energy_kwh' else '%' if c.endswith('mean') else 'source counter unit, unverified' if c=='counter_delta' else 'kW (guide; conditional)',
                            'family':'module_energy' if c=='energy_kwh' else 'power_variability' if c=='std_power_kw' else 'counter_delta' if c=='counter_delta' else 'power_factor' if c=='pf_mean' else 'unbalance',
                            'interval':'[hour,hour+1h),720 observations; counter_delta spans first..last (719 increments/3595s)',
                            'available_at':'hour+1h','same_hour_leakage':True,'physical_candidate':c!='energy_kwh',
                            'forecast_use':'Past completed-hour values only; no forward difference or interpolation'})
        max_power_mean_diff=max(max_power_mean_diff,float(np.max(np.abs(means[-1,i*nh:(i+1)*nh]-q.mean_power_kw.to_numpy()))))
        max_power_std_diff=max(max_power_std_diff,float(np.max(np.abs(stds[-1,i*nh:(i+1)*nh]-q.std_power_kw.to_numpy()))))
    target=pd.read_csv(TARGET,float_precision='round_trip'); target.index=pd.to_datetime(target.hour)
    y=target.provisional_sum_energy_kwh.reindex(hours).to_numpy(dtype=np.float64)
    energy_columns=[f'm{m}__energy_kwh' for m in ids]
    delta=float(np.max(np.abs(feature[energy_columns].sum(axis=1).to_numpy()-y)))
    integrated=float(np.max(np.abs(means[-1].reshape(len(ids),nh).sum(axis=0)-y)))
    if delta>1e-7 or integrated>1e-7:raise ValueError(f'Energy target mismatch {delta}, {integrated}')
    feature.insert(0,'target_energy_kwh',y)
    if not np.isfinite(feature.to_numpy()).all():raise ValueError('Nonfinite feature table')
    path=OUT/'data/hourly_features.csv';feature.to_csv(path,float_format='%.17g')
    safe_physical=[r['name'] for r in catalog if r['physical_candidate']]
    all_sensors=[r['name'] for r in catalog if r['family']!='module_energy' and r['source_column']!='activePower']
    manifest={'schema_version':1,'time_column':'hour','target_column':'target_energy_kwh',
              'hours':nh,'module_count':len(ids),'module_ids':ids,'feature_count':len(catalog),
              'time_key':'recorded naive localtime; known Unix/DST inconsistency not silently corrected',
              'source_resolution_seconds':5,'complete_observations_per_module_hour':720,
              'target_contract':'Sum of13 module activePower integration, unknown meter hierarchy; provisional, not certified plant consumption',
              'target_same_hour_feature_warning':'Current-hour module energy/activePower mean sum algebraically forms y(t); diagnostic only unless shifted strictly into the past',
              'groups':{'module_energy':energy_columns,'physical_sensors':safe_physical,'all_sensors':all_sensors},
              'features':catalog,'excluded':[
                  {'column':'operation/operation_rate','reason':'Constant1 in exhaustive existing profile; no predictive variation'},
                  {'column':'accumActiveEnergy/counter_first/counter_last','reason':'Cumulative level trend may give spurious correlation; excluded from model candidates'},
                  {'column':'timestamp','reason':'Known inconsistent UTC/DST offsets; no timestamp feature'},
                  {'column':'may_bill/may_carbon/agg_pow/future_actual','reason':'Unavailable future labels or direct target transforms; never joined'},
                  {'column':'activePower_mean','reason':'Kept for diagnostic identity but excluded from physical groups; duplicate hourly energy for1h'},
                  {'column':'activePower_std','reason':'Kept for extraction verification but std_power_kw is the canonical physical candidate'}],
              'availability_policy':'At origin t, all sensor/energy observations must have hour<t; never supply unknown future sensor trajectories',
              'target_available_at':'hour+1h; same completed bucket as constituent module energy',
              'telemetry_latency_assumption':'Availability assumes bucket completion without additional reporting/ingestion delay. Source has no trustworthy ingestion timestamp; production may require further lag.',
              'selection_policy':'Fold training-prefix only; April/May labels excluded from feature selection. Dec-March global analysis is explanatory, not reused as fold selection.',
              'post_EDA_limit':'April/May have been seen in prior EDA; this is controlled post-EDA revision, not untouched prospective validation'}
    dump(OUT/'data/feature_catalog.json',manifest)
    verify={'rows':rows,'raw_reads':1,'excluded_endpoint_rows':excluded,'complete_hours':nh,
            'complete_module_hours':groups,'counts_min':int(counts.min()),'counts_max':int(counts.max()),
            'feature_columns':len(catalog),'energy_sum_vs_existing_target_max_absdiff':delta,
            'raw_activePower_integrated_vs_target_max_absdiff':integrated,
            'raw_activePower_mean_vs_old_max_absdiff':max_power_mean_diff,
            'raw_activePower_std_vs_old_max_absdiff':max_power_std_diff,
            'source_size_unchanged':RAW.stat().st_size==original_stat.st_size,
            'source_mtime_unchanged':RAW.stat().st_mtime_ns==original_stat.st_mtime_ns,
            'target_sha256':hashlib.sha256(TARGET.read_bytes()).hexdigest(),
            'feature_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'elapsed_seconds':time.perf_counter()-started,'all_finite':True}
    dump(OUT/'data/extraction_verification.json',verify)
    # Small original derived levels are retained only for spurious-trend diagnosis.
    complete[['module_id','hour','operation_rate','counter_first','counter_last','std_power_kw','energy_kwh']].to_csv(OUT/'data/derived_audit_only.csv',index=False,float_format='%.17g')
    print(json.dumps(verify),flush=True)

if __name__=='__main__':main()
