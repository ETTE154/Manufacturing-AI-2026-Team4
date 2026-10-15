"""Strict historical prefixes, reversible numerical scalers, native kWh scores."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import os,json,hashlib,random
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
TARGET=ROOT/'output/analysis/baseline_audit/complete_hour_module_sum.csv'
MODULE_SOURCE=ROOT/'output/analysis/hourly_by_equipment.csv'
SEED=20261014;THREADS=4;POINTS=745;VALID=pd.to_datetime(['2025-02-15','2025-02-22','2025-02-25'])
TEST=pd.Timestamp('2025-03-29');FINAL=pd.Timestamp('2025-04-30')
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray,pd.Index)):return [clean(x) for x in v]
    if isinstance(v,(pd.Timestamp,Path)):return str(v)
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def write(path,r):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(clean(r),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');tmp.replace(p)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def assert_training_permitted():
    if (OUT/'global_selection_lock.json').exists():
        raise RuntimeError('This experiment is selection-frozen. New tuning requires a new experiment directory, not resetting this May-evaluated comparison.')
def seed():
    random.seed(SEED);np.random.seed(SEED)
    import torch
    torch.manual_seed(SEED);torch.set_num_threads(THREADS);torch.use_deterministic_algorithms(True)
def load_y():
    assert sha(TARGET)=='729b149a3e3165dfb339f0386249bfbda0326657f9e722d86da8138db36117de'
    f=pd.read_csv(TARGET,float_precision='round_trip');y=pd.Series(f.provisional_sum_energy_kwh.to_numpy(float),index=pd.to_datetime(f.hour),name='energy_kwh')
    assert y.index.equals(pd.date_range('2024-12-01',periods=3600,freq='h'))
    return y
def prefix(y,origin):
    h=y.loc[y.index<pd.Timestamp(origin)].copy();h.attrs={}
    assert len(h)>0 and h.index[-1]+pd.Timedelta(hours=1)==pd.Timestamp(origin)
    return h
def daily(h):
    d=h.resample('D').agg(['mean','count']);return d.loc[d['count']==24,'mean']
def native(h,res):return h if res=='hourly' else daily(h)
def hourly_map(p,res):return np.asarray(p,float)[:745] if res=='hourly' else np.repeat(np.asarray(p,float),24)[:745]
def norm_fit(values,kind):
    a=np.asarray(values,float)
    if a.ndim==1:a=a[:,None]
    center=a.mean(axis=0);r=a-center
    if kind=='zscore':offset=np.zeros(a.shape[1]);scale=r.std(axis=0)
    elif kind=='robust':offset=np.median(r,axis=0);scale=np.quantile(r,.75,axis=0)-np.quantile(r,.25,axis=0)
    elif kind=='minmax':offset=r.min(axis=0);scale=r.max(axis=0)-r.min(axis=0)
    else:raise ValueError(kind)
    scale=np.where(scale>1e-10,scale,1.)
    return {'kind':kind,'center':center.tolist(),'offset':offset.tolist(),'scale':scale.tolist(),
            'fit_rows':len(a),'residual_centering':'Train-prefix arithmetic mean before numerical scaling',
            'native_unit':'provisional guide kWh per hour','physical_unit_conversion':False}
def transform(a,n):return (np.asarray(a,float)-np.asarray(n['center'])-np.asarray(n['offset']))/np.asarray(n['scale'])
def inverse(a,n):return np.asarray(a,float)*np.asarray(n['scale'])+np.asarray(n['offset'])+np.asarray(n['center'])
def zero_prior(n):return -np.asarray(n['offset'])/np.asarray(n['scale'])
def metrics(a,p):
    a=np.asarray(a,float);p=np.asarray(p,float);assert a.shape==p.shape and np.isfinite(a).all() and np.isfinite(p).all()
    e=p-a;den=((a-a.mean())**2).sum()
    return {'count':len(a),'mae_kwh':float(abs(e).mean()),'rmse_kwh':float(np.sqrt((e*e).mean())),
            'mape_percent':float((abs(e)/abs(a)).mean()*100),'r2':float(1-(e*e).sum()/den),
            'bias_pred_minus_actual_kwh':float(e.mean())}
def frame(name,split,fold,origin,y,p):
    idx=pd.date_range(origin,periods=745,freq='h');a=y.reindex(idx).to_numpy(float);p=np.asarray(p,float)
    return pd.DataFrame({'id':idx,'model':name,'split':split,'fold':fold,'origin':origin,'horizon_offset_hours':np.arange(745),
         'actual_kwh':a,'predicted_kwh':p,'absolute_error_kwh':abs(p-a)})
def summary(f):
    late=f[f.horizon_offset_hours>=240]
    return {'full745':metrics(f.actual_kwh,f.predicted_kwh),'late505':metrics(late.actual_kwh,late.predicted_kwh),
            'mean_origin_late505_mae_kwh':float(late.groupby('origin').absolute_error_kwh.mean().mean()),
            'per_origin_late505_mae':late.groupby('origin').absolute_error_kwh.mean().to_dict(),
            'origin_count':int(f.origin.nunique()),'late_unique_target_hours':int(f.loc[f.horizon_offset_hours>=240,'id'].nunique())}
def matrix(h,cfg):
    s=native(h,cfg['resolution']);return pd.DataFrame({'target':s.to_numpy(float)},index=s.index)
def audit_units():
    y=load_y();h=prefix(y,VALID[0]);tests=[]
    for kind in ['zscore','robust','minmax']:
        n=norm_fit(h.values,kind);re=inverse(transform(h.values[:,None],n),n).ravel()
        error=float(abs(re-h.values).max());assert error<1e-8
        p=np.repeat(h.mean(),len(h));r=metrics(h,p)['r2']
        invariant={str(factor):float(abs(metrics(h*factor,p*factor)['r2']-r)) for factor in [1/288,1/1000,1000.]}
        assert max(invariant.values())<1e-8
        tests.append({'scaler':kind,'roundtrip_max_error_kwh':error,'normalizer':n,'fit_end':str(h.index[-1]),'R2_both_truth_and_prediction_unit_change_errors':invariant})
    return {'status':'pass','tests':tests,'physical_unit_status':'Guide kW/kWh provisional; no evidence authorizes a divide288 or divide1000 correction',
            'conclusion':'Correct reversible normalization improves conditioning. Scaling truth and prediction together cannot repair R2 or forecast direction.',
            'target_sha256':sha(TARGET),'guide_target_retained':True}
