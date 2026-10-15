"""Independent full-data time/key audit and descriptive anomaly candidates."""
from pathlib import Path
import json, time
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output'/'analysis'
SOURCE=ROOT/'traing-dataset'/'rtu_data_full.csv'
cols=['module(equipment)','timestamp','localtime','operation','activePower',
      'voltageR','voltageS','voltageT','currentR','currentS','currentT',
      'powerFactorR','powerFactorS','powerFactorT']
offsets={}; transitions=[]; states={}; candidates=[]; counts={}; rows=0
t0=time.monotonic()
for ci,d in enumerate(pd.read_csv(SOURCE,usecols=cols,chunksize=500_000)):
    rows+=len(d)
    dt=pd.to_datetime(d.localtime.astype(str),format='%Y%m%d%H%M%S',errors='raise')
    localms=dt.astype('datetime64[ms]').astype('int64').to_numpy()
    d['dt']=dt; d['localms']=localms
    d['offset']=localms-d.timestamp.to_numpy()
    for k,v in d['offset'].value_counts().items():offsets[str(k)]=offsets.get(str(k),0)+int(v)
    for m,q in d.groupby('module(equipment)',sort=False):
        s=states.setdefault(m,{'local_origin':int(q.localms.iloc[0]),'unix_origin':int(q.timestamp.iloc[0]),
            'local_seen':np.zeros(3_000_000,dtype=np.uint8),'unix_seen':np.zeros(3_000_000,dtype=np.uint8),
            'local_duplicate_rows':0,'unix_duplicate_rows':0,'local_negative_diffs':0,
            'local_non_5s_diffs':0,'local_last':None,'offset_last':None})
        for field,origin,seenkey,outkey in [('localms','local_origin','local_seen','local_duplicate_rows'),('timestamp','unix_origin','unix_seen','unix_duplicate_rows')]:
            a=q[field].to_numpy()
            idx=(a-s[origin])//5000
            assert np.all((a-s[origin])%5000==0) and idx.min()>=0 and idx.max()<3_000_000
            keys,n=np.unique(idx,return_counts=True)
            s[outkey]+=int(np.sum(n-1)+np.sum(s[seenkey][keys]>0))
            s[seenkey][keys]=1
        a=q.localms.to_numpy()
        dif=np.diff(a,prepend=s['local_last'] if s['local_last'] is not None else a[0]-5000)
        s['local_negative_diffs']+=int((dif<0).sum())
        s['local_non_5s_diffs']+=int((dif!=5000).sum())
        s['local_last']=int(a[-1])
        off=q.offset.to_numpy()
        changed=off!=np.r_[s['offset_last'] if s['offset_last'] is not None else off[0],off[:-1]]
        for _,row in q.loc[changed].iterrows():
            transitions.append({'equipment':m,'localtime':str(row['dt']),'offset_ms':int(row['offset'])})
        s['offset_last']=int(off[-1])
    voltage=(d[['voltageR','voltageS','voltageT']]<205).any(axis=1)|(d[['voltageR','voltageS','voltageT']]>225).any(axis=1)
    pf=(d[['powerFactorR','powerFactorS','powerFactorT']]<80).any(axis=1)|(d[['powerFactorR','powerFactorS','powerFactorT']]>100).any(axis=1)
    domain=(d[['currentR','currentS','currentT']]<0).any(axis=1)|(d.activePower<0)|~d.operation.isin([0,1])
    for k,mask in [('voltage_outside_205_225',voltage),('phase_pf_outside_80_100',pf),('invalid_sign_or_operation',domain)]:
        counts[k]=counts.get(k,0)+int(mask.sum())
    mask=voltage|pf|domain
    if mask.any():
        q=d.loc[mask].copy()
        q['voltage_candidate']=voltage[mask];q['pf_candidate']=pf[mask];q['domain_candidate']=domain[mask]
        candidates.append(q)
    if ci%10==0:print(f'AUDIT rows={rows:,} elapsed={time.monotonic()-t0:.1f}s candidate_rows={sum(counts.values()):,}',flush=True)

detail={m:{k:v for k,v in s.items() if not k.endswith('_seen')} for m,s in states.items()}
summary={'rows':rows,'offset_local_naive_minus_unix_ms':offsets,'offset_transitions':transitions,
    'time_key_checks':detail,'rule_counts':counts,
    'rules_note':'EDA screening: phase voltage outside [205,225] V; phase power factor outside [80,100] percent. Thresholds based on typical training ranges, not validated physical limits or anomaly labels.',
    'elapsed_seconds':round(time.monotonic()-t0,1)}
ca=pd.concat(candidates,ignore_index=True) if candidates else pd.DataFrame()
ca.to_csv(OUT/'anomaly_candidate_rows.csv',index=False,encoding='utf-8-sig')
events=[]
if not ca.empty:
    for m,q in ca.groupby('module(equipment)',sort=False):
        q=q.sort_values('dt').copy()
        g=q.dt.diff().dt.total_seconds().ne(5).cumsum()
        for _,z in q.groupby(g):
            events.append({'equipment':m,'start':str(z.dt.min()),'last_sample':str(z.dt.max()),
                'end_exclusive':str(z.dt.max()+pd.Timedelta(seconds=5)),'records':len(z),'observed_duration_seconds':len(z)*5,
                'voltage_candidate_rows':int(z.voltage_candidate.sum()),'pf_candidate_rows':int(z.pf_candidate.sum()),
                'minimum_phase_voltage':float(z[['voltageR','voltageS','voltageT']].min().min()),
                'minimum_phase_pf':float(z[['powerFactorR','powerFactorS','powerFactorT']].min().min())})
pd.DataFrame(events).to_csv(OUT/'anomaly_events.csv',index=False,encoding='utf-8-sig')
summary['candidate_rows_unique']=len(ca);summary['event_count']=len(events)
summary['longest_consecutive_candidate_runs']=sorted(events,key=lambda x:x['records'],reverse=True)[:25]
summary['full_event_detail_csv']='anomaly_events.csv'
(OUT/'time_anomaly_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
# Repair only the generated profile. Full-data audit uses explicit milliseconds.
prof_path=OUT/'training_profile.json'
p=json.loads(prof_path.read_text(encoding='utf-8'))
p['timestamp_localtime_offset_ms']=offsets
p['timestamp_offset_method']='localtime parsed to explicit datetime64[ms], integer minus Unix timestamp; independently verified on all rows'
p['time_key_checks']=detail
prof_path.write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['time_key_checks','events','offset_transitions']},ensure_ascii=False),flush=True)
