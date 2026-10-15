"""Six contextual candidate patterns and frozen AI scoring; no training on import.

Class probabilities refer to explicit synthetic transformations, not real-fault
probabilities. All natural alarms remain unconfirmed, because labels are absent.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import re,json
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn
OUT=Path(__file__).resolve().parent
RAW=['voltageR','voltageS','voltageT','voltageRS','voltageST','voltageTR','currentR','currentS','currentT',
     'activePower','powerFactorR','powerFactorS','powerFactorT','reactivePowerLagging']
DERIVED=['I_mean','I_unbalance','V_unbalance','PF_mean','PF_min','Q_over_P','physical_log_ratio']
TYPE_IDS=['load_imbalance','reactive_power_surge','power_factor_drop','abnormal_pattern','night_operation','current_only_increase']
TYPE_LABELS=['부하 불균형','무효전력 급증','역률 급감','비정상 패턴','야간 이상가동','전류만 증가']
CLASS_NAMES=['unmodified_reference']+TYPE_IDS
FEATURE_NAMES=['z__'+c for c in RAW+DERIVED]+['z__P_hour','z__I_hour','night_indicator']

def prepare(rows):
    frame=pd.DataFrame(rows).copy()
    if frame.empty:return frame
    if 'module(equipment)' not in frame:
        if 'module_id' not in frame:raise ValueError('module(equipment) or module_id is required')
        frame['module(equipment)']=frame.module_id.astype(str)
    labels=frame['module(equipment)'].astype(str)
    mapping={label:(re.match(r'^(\d+)',label).group(1) if re.match(r'^(\d+)',label) else None) for label in labels.unique()}
    frame['module_id']=labels.map(mapping)
    if frame.module_id.isna().any():raise ValueError('Invalid module identity')
    if 'localtime' in frame:
        text=frame.localtime.astype(str)
        if not text.str.fullmatch(r'\d{14}').all():raise ValueError('localtime must be14digit YYYYMMDDHHMMSS')
        frame['dt']=pd.to_datetime(text,format='%Y%m%d%H%M%S',errors='raise')
    elif 'time' in frame:frame['dt']=pd.to_datetime(frame.time,format='ISO8601',errors='raise')
    else:raise ValueError('localtime or ISO time is required')
    if frame.dt.dt.tz is not None:raise ValueError('Use recorded naive localtime; no Unix offset correction')
    missing=[c for c in RAW if c not in frame]
    if missing:raise ValueError('Missing sensors: '+', '.join(missing))
    frame[RAW]=frame[RAW].apply(pd.to_numeric,errors='raise')
    if not np.isfinite(frame[RAW].to_numpy(dtype=float)).all():raise ValueError('Nonfinite sensor values; no zero imputation')
    return frame

def derive(frame,kappa=None):
    V=frame[['voltageR','voltageS','voltageT']].to_numpy(dtype=float)
    I=frame[['currentR','currentS','currentT']].to_numpy(dtype=float)
    PF=frame[['powerFactorR','powerFactorS','powerFactorT']].to_numpy(dtype=float)
    p=frame.activePower.to_numpy(dtype=float);q=frame.reactivePowerLagging.to_numpy(dtype=float)
    im=I.mean(axis=1);vm=V.mean(axis=1)
    apparent_pf=np.sum(V*I*PF/100.,axis=1)
    raw_ratio=np.maximum(p,1e-6)/np.maximum(apparent_pf,1e-6)
    calibration=np.ones(len(frame)) if kappa is None else np.array([kappa[str(m)] for m in frame.module_id])
    return pd.DataFrame({'I_mean':im,'I_unbalance':np.max(np.abs(I-im[:,None]),axis=1)/np.maximum(np.abs(im),1e-6),
          'V_unbalance':np.max(np.abs(V-vm[:,None]),axis=1)/np.maximum(np.abs(vm),1e-6),
          'PF_mean':PF.mean(axis=1),'PF_min':PF.min(axis=1),'Q_over_P':q/np.maximum(np.abs(p),1e-6),
          'physical_log_ratio':np.log(raw_ratio/np.maximum(calibration,1e-12)),
          'raw_P_over_sumVIPF':raw_ratio},index=frame.index)

def robust_stats(a):
    a=np.asarray(a,dtype=float);median=np.median(a,axis=0)
    mad=1.4826*np.median(np.abs(a-median),axis=0)
    iqr=(np.quantile(a,.75,axis=0)-np.quantile(a,.25,axis=0))/1.349
    scale=np.maximum.reduce([mad,iqr,np.maximum(np.abs(median)*.01,1e-6)])
    return {'median':median,'scale':scale}

def fit_baseline(frame):
    initial=derive(frame);kappa={m:float(np.median(initial.loc[frame.module_id.eq(m),'raw_P_over_sumVIPF'])) for m in frame.module_id.unique()}
    d=derive(frame,kappa);joined=pd.concat([frame[RAW],d[DERIVED]],axis=1);by_module={};hours={}
    for m,g in frame.groupby('module_id',sort=True):
        by_module[m]=robust_stats(joined.loc[g.index].to_numpy())
        for hour,h in g.groupby(g.dt.dt.hour):
            hours[f'{m}:{int(hour)}']=robust_stats(np.column_stack([h.activePower,d.loc[h.index,'I_mean']]))
    return {'by_module':by_module,'by_hour':hours,'kappa':kappa,'equipment':{m:str(g['module(equipment)'].iloc[0]) for m,g in frame.groupby('module_id')},
            'fit_end':str(frame.dt.max()),'fit_rows':len(frame),'kappa_limit':'Empirical module-specific ratio corrects raw numeric scale/measurement scope. It is not a certified unit conversion or ideal three-phase power equation.'}

def features(frame,baseline):
    unknown=set(frame.module_id.astype(str))-set(baseline['kappa'])
    if unknown:raise ValueError('Unknown module '+', '.join(sorted(unknown)))
    d=derive(frame,baseline['kappa']);joined=np.column_stack([frame[RAW].to_numpy(dtype=float),d[DERIVED].to_numpy(dtype=float)])
    X=np.empty((len(frame),len(FEATURE_NAMES)),dtype=np.float64)
    for m in frame.module_id.unique():
        if m not in baseline['by_module']:raise ValueError('Unknown module '+str(m))
        mask=frame.module_id.eq(m).to_numpy();b=baseline['by_module'][m]
        X[mask,:len(RAW)+len(DERIVED)]=(joined[mask]-b['median'])/b['scale']
    hour_values=np.column_stack([frame.activePower.to_numpy(dtype=float),d.I_mean.to_numpy(dtype=float)])
    for key in pd.unique(frame.module_id+':'+frame.dt.dt.hour.astype(str)):
        m,h=key.split(':');mask=(frame.module_id.eq(m)&frame.dt.dt.hour.eq(int(h))).to_numpy();b=baseline['by_hour'][key]
        X[mask,-3:-1]=(hour_values[mask]-b['median'])/b['scale']
    X[:,-1]=((frame.dt.dt.hour>=22)|(frame.dt.dt.hour<6)).to_numpy(dtype=float)
    assert np.isfinite(X).all()
    return X,d

def rule_scores(X):
    def col(name):return X[:,FEATURE_NAMES.index(name)]
    pos=lambda v:np.maximum(v,0.)
    phase_current=np.maximum.reduce([col('z__currentR'),col('z__currentS'),col('z__currentT')])
    scores=np.column_stack([
       pos(phase_current)+pos(col('z__I_unbalance')),
       pos(col('z__reactivePowerLagging'))+pos(col('z__Q_over_P'))+.25*pos(-col('z__PF_mean')),
       pos(-col('z__PF_min'))+pos(-col('z__PF_mean'))+.25*pos(col('z__Q_over_P')),
       np.maximum(np.max(np.abs(X[:,:len(RAW)]),axis=1),np.abs(col('z__physical_log_ratio'))),
       col('night_indicator')*(pos(col('z__P_hour'))+pos(col('z__I_hour'))),
       pos(col('z__I_mean'))+pos(-col('z__physical_log_ratio'))-.5*np.abs(col('z__activePower'))])
    return np.maximum(scores,0.)

class AE(nn.Module):
    def __init__(self,p):
        super().__init__();self.net=nn.Sequential(nn.Linear(p,32),nn.ReLU(),nn.Linear(32,12),nn.ReLU(),nn.Linear(12,32),nn.ReLU(),nn.Linear(32,p))
    def forward(self,x):return self.net(x)

def ae_scores(model,X,chunk=8192):
    result=[]
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(X),chunk):
            t=torch.from_numpy(np.asarray(X[start:start+chunk],dtype=np.float32))
            result.append((model(t)-t).pow(2).mean(dim=1).numpy().astype(np.float64))
    return np.concatenate(result) if result else np.empty(0,dtype=float)

class Detector:
    def __init__(self,path=None):
        torch.set_num_threads(4)
        self.path=Path(path) if path else OUT
        self.saved=joblib.load(self.path/'models.joblib')
        ck=torch.load(self.path/'ae.pt',map_location='cpu',weights_only=True)
        self.ae=AE(len(FEATURE_NAMES));self.ae.load_state_dict(ck['state_dict']);self.ae.eval()
        if ck['feature_names']!=FEATURE_NAMES or self.saved['feature_names']!=FEATURE_NAMES:raise ValueError('Feature schema mismatch')
    def scores(self,frame,full_ai=True):
        X,d=features(frame,self.saved['baseline']);r=rule_scores(X);a=ae_scores(self.ae,X)
        iso=-np.asarray(self.saved['iforest'].score_samples(X),dtype=np.float64) if full_ai else np.full(len(frame),np.nan)
        probs=np.asarray(self.saved['classifiers'][self.saved['selected_classifier']].predict_proba(X),dtype=np.float64) if full_ai else None
        thresholds=np.empty((len(frame),len(TYPE_IDS)),dtype=np.float64)
        for m in frame.module_id.unique():
            thresholds[frame.module_id.eq(m).to_numpy()]=[self.saved['thresholds']['rules'][str(m)][t] for t in TYPE_IDS]
        return {'features':X,'derived':d,'rule_scores':r,'rule_thresholds':thresholds,'rule_flags':r>thresholds,
                'ae_score':a,'ae_alarm':a>self.saved['thresholds']['ae'],
                'iforest_score':iso,'iforest_alarm':iso>self.saved['thresholds']['iforest'],'class_probs':probs}
    def diagnose_batch(self,rows):
        f=prepare(rows)
        if f.empty:return []
        scored=self.scores(f,True);result=[]
        for i,(_,row) in enumerate(f.iterrows()):
            rules=[];derived=scored['derived'].iloc[i]
            for t,kind in enumerate(TYPE_IDS):
                if not scored['rule_flags'][i,t]:continue
                rules.append({'type':kind,'label':TYPE_LABELS[t],'score':float(scored['rule_scores'][i,t]),
                    'threshold':float(scored['rule_thresholds'][i,t]),'evidence':{
                    'active_power_raw':float(row.activePower),'current_mean_A':float(derived.I_mean),
                    'current_unbalance_fraction':float(derived.I_unbalance),'PF_min_percent':float(derived.PF_min),
                    'reactive_power_raw':float(row.reactivePowerLagging),'Q_over_P':float(derived.Q_over_P),
                    'calibrated_log_P_over_sumVIPF':float(derived.physical_log_ratio),'hour':int(row['dt'].hour),
                    'context':'December module/hour robust baseline; January empirical upper-tail thresholds; physical coefficient is empirical.'}})
            ai={'iforest_score':float(scored['iforest_score'][i]),'iforest_threshold':float(self.saved['thresholds']['iforest']),
                'iforest_alarm':bool(scored['iforest_alarm'][i]),'ae_score':float(scored['ae_score'][i]),
                'ae_threshold':float(self.saved['thresholds']['ae']),'ae_alarm':bool(scored['ae_alarm'][i]),
                'class_probs':{c:float(p) for c,p in zip(CLASS_NAMES,scored['class_probs'][i])},
                'classifier':self.saved['selected_classifier'],'probability_meaning':'Synthetic transformation class similarity; not a real-fault probability.'}
            alarm=bool(rules or ai['ae_alarm'] or ai['iforest_alarm'])
            result.append({'module_id':str(row.module_id),'equipment':self.saved['baseline']['equipment'].get(str(row.module_id),str(row['module(equipment)'])),
                    'time':row['dt'].isoformat(),'source_kind':str(row.get('source_kind','recorded')),
                    'source_tag':str(row.get('source_tag','original_5s_readonly' if row.get('source_kind','recorded')=='recorded' else 'caller_synthetic_scenario')),'rules':rules,'ai':ai,
                    'model_available_from':'2025-02-01T00:00:00',
                    'evaluation_scope':'chronological_after_calibration' if row['dt']>=pd.Timestamp('2025-02-01') else 'retrospective_fit_or_calibration_period',
                    'status':'unconfirmed_candidate' if alarm else 'no_candidate',
                    'confidence':{'evidence_level':'multiple_signals' if rules and(ai['ae_alarm'] or ai['iforest_alarm']) else 'limited',
                                  'fault_probability':None,'reason':'Actual fault labels/times are unavailable; no confirmed diagnosis.'}})
        return result

_DEFAULT=None
def diagnose_batch(rows):
    global _DEFAULT
    if _DEFAULT is None:_DEFAULT=Detector()
    return _DEFAULT.diagnose_batch(rows)
