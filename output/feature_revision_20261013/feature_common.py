"""Hourly feature contract and train-prefix-only selection; no import side effects."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SEED=20261013
THREADS=4
TARGET='target_energy_kwh'

def load_catalog():return json.loads((OUT/'data/feature_catalog.json').read_text(encoding='utf-8'))

def load_hourly():
    frame=pd.read_csv(OUT/'data/hourly_features.csv',index_col='hour',parse_dates=['hour'],float_precision='round_trip')
    expected=pd.date_range('2024-12-01','2025-04-29 23:00',freq='h')
    if not frame.index.equals(expected) or not np.isfinite(frame.to_numpy()).all():raise ValueError('Feature grid/values invalid')
    return frame

def load_target():return load_hourly()[TARGET].copy()

def feature_columns(group='physical_sensors'):
    catalog=load_catalog()
    if group not in catalog['groups']:raise ValueError(f'Unknown group {group}')
    return list(catalog['groups'][group])

def bh_fdr(pvalues):
    p=np.asarray(pvalues,dtype=float);out=np.full(len(p),np.nan);good=np.flatnonzero(np.isfinite(p))
    if len(good):
        order=good[np.argsort(p[good])];adjust=p[order]*len(good)/np.arange(1,len(good)+1)
        out[order]=np.clip(np.minimum.accumulate(adjust[::-1])[::-1],0,1)
    return out

def correlation(x,y,method='pearson'):
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    finite=np.isfinite(x)&np.isfinite(y);x,y=x[finite],y[finite];n=len(x)
    if n<20 or np.std(x)==0 or np.std(y)==0:return np.nan,np.nan,n
    if method=='spearman':x=stats.rankdata(x);y=stats.rankdata(y)
    r=float(np.corrcoef(x,y)[0,1])
    ax=float(np.corrcoef(x[:-1],x[1:])[0,1]);ay=float(np.corrcoef(y[:-1],y[1:])[0,1])
    product=np.clip(ax*ay,-.99,.99)
    neff=float(np.clip(n*(1-product)/(1+product),3,n))
    t=abs(r)*np.sqrt((neff-2)/max(1-r*r,1e-15))
    p=float(2*stats.t.sf(t,neff-2))
    return r,p,n

def select_sensor_features(history_frame,max_features=12,horizons=(0,5,23),
                           min_abs_r=.08,fdr=.05,stability_fraction=.65,redundancy=.95):
    """Return (columns,audit) using only a caller-supplied, pre-origin prefix.

    Each row t uses completed feature row t-1 to forecast y(t+h). Labels must
    stay within both prefix and each stability subperiod. No April/May input
    is permitted during CV/HPO selection; callers refit final selected schemas.
    P-values use approximate AR(1) effective-N correction, not causal evidence.
    """
    prefix=history_frame.copy()
    if prefix.empty or not prefix.index.is_monotonic_increasing or not prefix.index.is_unique:raise ValueError('Invalid prefix')
    if prefix.index.max()>=pd.Timestamp('2025-04-01'):raise ValueError('April/May labels cannot select features; freeze pre-April schema')
    cols=[c for c in feature_columns('physical_sensors') if c in prefix]
    cut=int(len(prefix)*stability_fraction)
    pieces=[prefix.iloc[:cut],prefix.iloc[cut:]]
    records=[]
    for h in horizons:
        for c in cols:
            rs=[];ps=[];ns=[]
            for piece in [prefix]+pieces:
                r,p,n=correlation(piece[c].shift(1),piece[TARGET].shift(-h))
                rs.append(r);ps.append(p);ns.append(n)
            stable=bool(np.isfinite(rs).all() and np.sign(rs[0])==np.sign(rs[1])==np.sign(rs[2])
                        and min(abs(rs[1]),abs(rs[2]))>=min_abs_r)
            records.append({'feature':c,'target_offset_hours':int(h),'lag_hours':1,
                            'r_full':rs[0],'r_early':rs[1],'r_late':rs[2],
                            'p_approx_AR1':ps[0],'n':ns[0],'stable':stable})
    q=bh_fdr([r['p_approx_AR1'] for r in records])
    for r,qi in zip(records,q):r['q_BH']=float(qi)
    eligible={}
    for r in records:
        if r['stable'] and r['q_BH']<=fdr and abs(r['r_full'])>=min_abs_r:
            score=min(abs(r['r_full']),abs(r['r_early']),abs(r['r_late']))
            if score>eligible.get(r['feature'],-1):eligible[r['feature']]=score
    selected=[];dropped=[]
    for c in sorted(eligible,key=lambda c:(-eligible[c],c)):
        duplicate=next((other for other in selected if abs(prefix[c].corr(prefix[other]))>redundancy),None)
        if duplicate:dropped.append({'feature':c,'redundant_with':duplicate})
        elif len(selected)<max_features:selected.append(c)
    audit={'training_start':str(prefix.index.min()),'training_end':str(prefix.index.max()),'training_rows':len(prefix),
           'selected_columns':selected,'selected_count':len(selected),'candidate_count':len(cols),
           'max_features':max_features,'horizons_offsets':list(horizons),'min_abs_r':min_abs_r,'BH_fdr':fdr,
           'stability_fraction':stability_fraction,'redundancy_threshold':redundancy,
           'selection_uses_April_or_May':False,'records':records,'redundancy_dropped':dropped,
           'alignment':'Row t uses sensor row t-1h; target offsets0/5/23 are leads1/6/24 from last observation. Nonwrapping shifts drop labels beyond each prefix/subperiod end.',
           'statistical_limit':'Approximate AR1 effective sample size and BH screening; stable correlation is association, not causality or assured predictive improvement. Zero selection is allowed.'}
    return selected,audit
