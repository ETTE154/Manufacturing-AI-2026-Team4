"""Stream the supplied RTU CSV without changing the source files."""
from pathlib import Path
import json, time, hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'output' / 'analysis'
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = ROOT / 'traing-dataset' / 'rtu_data_full.csv'
CHUNK = 300_000
NUM = ['voltageR','voltageS','voltageT','voltageRS','voltageST','voltageTR',
       'currentR','currentS','currentT','activePower','powerFactorR','powerFactorS',
       'powerFactorT','reactivePowerLagging','accumActiveEnergy']
parts, profiles, samples = [], {}, []
last, module_blocks, previous_module = {}, [], None
rows = 0
start = time.monotonic()
missing = {}
time_offsets = {}
for ci, d in enumerate(pd.read_csv(SOURCE, chunksize=CHUNK)):
    rows += len(d)
    for c, n in d.isna().sum().items():
        missing[c] = missing.get(c, 0) + int(n)
    d['dt'] = pd.to_datetime(d['localtime'].astype(str), format='%Y%m%d%H%M%S', errors='coerce')
    offsets = (d['dt'].astype('datetime64[ms]').astype('int64') - d['timestamp']).value_counts()
    for k, v in offsets.items():
        time_offsets[str(k)] = time_offsets.get(str(k), 0) + int(v)
    d['hour'] = d['dt'].dt.floor('h')
    d['pf_mean'] = d[['powerFactorR','powerFactorS','powerFactorT']].mean(axis=1)
    d['current_unbalance_pct'] = (d[['currentR','currentS','currentT']].sub(d[['currentR','currentS','currentT']].mean(axis=1),axis=0).abs().max(axis=1) / d[['currentR','currentS','currentT']].mean(axis=1))*100
    d['voltage_unbalance_pct'] = (d[['voltageR','voltageS','voltageT']].sub(d[['voltageR','voltageS','voltageT']].mean(axis=1),axis=0).abs().max(axis=1) / d[['voltageR','voltageS','voltageT']].mean(axis=1))*100
    # Timestamp grid allows exact key completeness checks without retaining 30M keys.
    for m, q in d.groupby('module(equipment)', sort=False):
        if m != previous_module:
            module_blocks.append(m)
            previous_module = m
        p = profiles.setdefault(m, {'rows':0,'first':None,'last':None,'operation_counts':{},
            'zero_diffs':0,'negative_time_diffs':0,'gap_intervals':0,'missing_5s_slots':0,
            'offgrid':0,'counter_negative_diffs':0,'counter_zero_diffs':0,'counter_delta_sum':0.,
            'counter_delta_count':0,'energy_delta_sum_for_positive_time':0.,
            'delta_energy_kwh_sum':0.,'delta_energy_power_denominator':0.,
            'sum':{},'sumsq':{},'min':{},'max':{}})
        p['rows'] += len(q)
        p['first'] = min(p['first'], str(q['dt'].iloc[0])) if p['first'] else str(q['dt'].iloc[0])
        p['last'] = max(p['last'], str(q['dt'].iloc[-1])) if p['last'] else str(q['dt'].iloc[-1])
        for k,v in q['operation'].value_counts(dropna=False).items():
            p['operation_counts'][str(k)] = p['operation_counts'].get(str(k),0)+int(v)
        ts = q['timestamp'].to_numpy()
        ac = q['accumActiveEnergy'].to_numpy()
        pw = q['activePower'].to_numpy()
        lt = last.get(m)
        dif = np.diff(ts, prepend=lt[0] if lt else ts[0]-5000)
        adif = np.diff(ac, prepend=lt[1] if lt else ac[0])
        mask = np.ones(len(q), dtype=bool)
        if lt is None:
            mask[0] = False
            p['counter_first'] = float(ac[0])
        last[m] = (int(ts[-1]),float(ac[-1]))
        p['counter_last'] = float(ac[-1])
        p['zero_diffs'] += int((dif[mask]==0).sum())
        p['negative_time_diffs'] += int((dif[mask]<0).sum())
        p['gap_intervals'] += int((dif[mask]>5000).sum())
        p['missing_5s_slots'] += int(np.maximum(dif[mask]//5000-1,0).sum())
        p['offgrid'] += int((ts % 5000 != 0).sum())
        p['counter_negative_diffs'] += int((adif[mask]<0).sum())
        p['counter_zero_diffs'] += int((adif[mask]==0).sum())
        p['counter_delta_sum'] += float(adif[mask].sum())
        p['counter_delta_count'] += int(mask.sum())
        valid = mask & (dif==5000) & (adif>=0)
        p['delta_energy_kwh_sum'] += float(adif[valid].sum())
        p['delta_energy_power_denominator'] += float((pw[valid]*5/3600).sum())
        for c in NUM+['pf_mean','current_unbalance_pct','voltage_unbalance_pct']:
            a = q[c].to_numpy(dtype=float)
            p['sum'][c] = p['sum'].get(c,0.)+float(np.nansum(a))
            p['sumsq'][c] = p['sumsq'].get(c,0.)+float(np.nansum(a*a))
            p['min'][c] = min(p['min'].get(c,float('inf')),float(np.nanmin(a)))
            p['max'][c] = max(p['max'].get(c,float('-inf')),float(np.nanmax(a)))
        p['low_pf_rows'] = p.get('low_pf_rows',0)+int((q['pf_mean']<90).sum())
        p['negative_power_rows'] = p.get('negative_power_rows',0)+int((q['activePower']<0).sum())
    h = d.groupby(['module(equipment)','hour'],sort=False).agg(
        records=('activePower','size'),power_sum=('activePower','sum'),power_sumsq=('activePower',lambda x:np.sum(x*x)),
        power_min=('activePower','min'),power_max=('activePower','max'),operation_sum=('operation','sum'),
        pf_sum=('pf_mean','sum'),voltage_unbalance_sum=('voltage_unbalance_pct','sum'),current_unbalance_sum=('current_unbalance_pct','sum'),
        counter_first=('accumActiveEnergy','first'),counter_last=('accumActiveEnergy','last'))
    parts.append(h.reset_index())
    samples.append(d.iloc[::2000][['module(equipment)','dt']+NUM+['pf_mean','current_unbalance_pct','voltage_unbalance_pct']])
    if ci % 5 == 0:
        print(f'PROGRESS rows={rows:,} elapsed={time.monotonic()-start:.1f}s modules={len(profiles)}',flush=True)

raw = pd.concat(parts,ignore_index=True)
hourly = raw.groupby(['module(equipment)','hour'],sort=True).agg(
    records=('records','sum'),power_sum=('power_sum','sum'),power_sumsq=('power_sumsq','sum'),
    power_min=('power_min','min'),power_max=('power_max','max'),operation_sum=('operation_sum','sum'),
    pf_sum=('pf_sum','sum'),voltage_unbalance_sum=('voltage_unbalance_sum','sum'),current_unbalance_sum=('current_unbalance_sum','sum'),
    counter_first=('counter_first','first'),counter_last=('counter_last','last')).reset_index()
hourly['energy_kwh'] = hourly['power_sum']*5/3600
hourly['mean_power_kw'] = hourly['power_sum']/hourly['records']
hourly['std_power_kw'] = np.sqrt(np.maximum(hourly['power_sumsq']/hourly['records']-hourly['mean_power_kw']**2,0))
hourly['operation_rate'] = hourly['operation_sum']/hourly['records']
for c in ['pf','voltage_unbalance','current_unbalance']:
    hourly[c+'_mean'] = hourly[c+'_sum']/hourly['records']
hourly['complete_720'] = hourly['records'].eq(720)
hourly.to_csv(OUT/'hourly_by_equipment.csv',index=False,encoding='utf-8-sig')
sm = pd.concat(samples,ignore_index=True)
sm.to_csv(OUT/'sampled_sensor_rows.csv',index=False,encoding='utf-8-sig')
module_rows = []
for m,p in profiles.items():
    n=p['rows']
    r={'equipment':m,'records':n,'first':p['first'],'last':p['last'],
       'energy_kwh':p['sum']['activePower']*5/3600,'operation_rate':sum(float(k)*v for k,v in p['operation_counts'].items())/n,
       'low_pf_rate':p['low_pf_rows']/n,'counter_unit_ratio':p['delta_energy_kwh_sum']/p['delta_energy_power_denominator']}
    for c in NUM+['pf_mean','current_unbalance_pct','voltage_unbalance_pct']:
        r[c+'_mean']=p['sum'][c]/n
        r[c+'_std']=float(np.sqrt(max(p['sumsq'][c]/n-r[c+'_mean']**2,0)))
        r[c+'_min']=p['min'][c];r[c+'_max']=p['max'][c]
    for c in ['zero_diffs','negative_time_diffs','gap_intervals','missing_5s_slots','offgrid','counter_negative_diffs','counter_zero_diffs','negative_power_rows','counter_first','counter_last']:
        r[c]=p[c]
    module_rows.append(r)
pd.DataFrame(module_rows).to_csv(OUT/'equipment_summary.csv',index=False,encoding='utf-8-sig')
summary={'source':str(SOURCE.relative_to(ROOT)),'source_bytes':SOURCE.stat().st_size,'records':rows,
    'equipment_count':len(profiles),'columns':list(missing),'missing':missing,
    'timestamp_localtime_offset_ms':time_offsets,'equipment_blocks':module_blocks,
    'elapsed_seconds':round(time.monotonic()-start,1),'profiles':profiles,
    'sample_method':'every 2000th row of each 300000-row chunk; systematic descriptive sample, not unbiased inference',
    'aggregation':'sum(activePower)*5/3600; recorded intervals only; no imputation',
    'complete_hours':int(hourly['complete_720'].sum()),'hourly_equipment_rows':len(hourly)}
(OUT/'training_profile.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['profiles','columns','timestamp_localtime_offset_ms']},ensure_ascii=False),flush=True)
