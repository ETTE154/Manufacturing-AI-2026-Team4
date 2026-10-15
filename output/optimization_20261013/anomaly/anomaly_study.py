"""Isolated, reproducible anomaly comparison on existing RTU data.

Never treats an exploratory electrical rule or synthetic perturbation as real-fault
ground truth. All writes are confined to this script's anomaly directory.
Prepare manifests without source approval; run extraction/training only after the
parent source audit confirms the downloaded archive is the same dataset.
"""
from pathlib import Path
import argparse, hashlib, json, os, pickle, shutil, sys, time
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
OLD = ROOT/'output'/'modeling'/'diagnosis'
ANA = ROOT/'output'/'analysis'
SOURCE = ROOT/'traing-dataset'/'rtu_data_full.csv'
SEED = 20261013
KEY = 'module(equipment)'
SENSORS = ['voltageR','voltageS','voltageT','voltageRS','voltageST','voltageTR',
           'currentR','currentS','currentT','activePower','powerFactorR',
           'powerFactorS','powerFactorT','reactivePowerLagging']
TRANSFER = ['currentR','currentS','currentT','activePower','reactivePowerLagging']
LEGACY = ['xgb_rule_distillation','xgb_rule_sensor_excluded_transfer',
          'ae_unsupervised_all_training','ae_weak_normal_proxy']
NAMES = {'xgb_rule_distillation':'기존 규칙 모사 XGB',
         'xgb_rule_sensor_excluded_transfer':'기존 규칙 센서 제외 XGB',
         'ae_unsupervised_all_training':'기존 전체 학습 AE',
         'ae_weak_normal_proxy':'기존 약한 정상 AE',
         'isolation_forest':'Isolation Forest', 'robust_z':'Robust z',
         'xgb_synthetic_augmented':'합성 증강 XGB'}

def dump(obj, name):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def readjson(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def localkey(t): return int(pd.Timestamp(t).strftime('%Y%m%d%H%M%S'))
def weak(d):
    return (((d[['voltageR','voltageS','voltageT']]<205)|(d[['voltageR','voltageS','voltageT']]>225)).any(axis=1)
            |((d[['powerFactorR','powerFactorS','powerFactorT']]<80)|(d[['powerFactorR','powerFactorS','powerFactorT']]>100)).any(axis=1)
            |(d[['currentR','currentS','currentT','activePower']]<0).any(axis=1)
            |~d.operation.isin([0,1])).to_numpy(dtype=np.int8)

def prepare():
    OUT.mkdir(exist_ok=True,parents=True)
    for name in ['models','figures','tmp']: (OUT/name).mkdir(exist_ok=True)
    events=pd.read_csv(ANA/'anomaly_events.csv',parse_dates=['start','last_sample','end_exclusive'])
    candidates=pd.read_csv(ANA/'anomaly_candidate_rows.csv',usecols=[KEY,'localtime'])
    candidates['dt']=pd.to_datetime(candidates.localtime.astype(str),format='%Y%m%d%H%M%S')
    modules=sorted(pd.read_csv(OLD/'uniform_raw_sample.csv.gz',usecols=[KEY])[KEY].unique())
    bymod={m:np.sort(d.dt.to_numpy(dtype='datetime64[s]').astype(np.int64)) for m,d in candidates.groupby(KEY)}
    def empty(m,start,seconds=1800):
        seq=bymod.get(m,np.empty(0,dtype=np.int64));a=int(pd.Timestamp(start).timestamp())
        return np.searchsorted(seq,a+seconds)==np.searchsorted(seq,a)
    windows=[]
    for i,m in enumerate(['13(3호기)','15(예비건조기)','17(6호기)']):
        e=events[(events.equipment==m)&events.start.between('2025-04-10','2025-04-20')].sort_values('start').iloc[0]
        start=e.start-pd.Timedelta(minutes=15)
        windows.append({'window_id':f'candidate_{i+1}','role':'candidate_visual','split':'test','module':m,
                        'start':str(start),'end_exclusive':str(start+pd.Timedelta(minutes=30)),
                        'focus_event':str(e.start),'selection':'first chronological candidate on/after April 10, with ±15 minutes context'})
        normal=start+pd.Timedelta(hours=2)
        for _ in range(1000):
            if empty(m,normal):break
            normal+=pd.Timedelta(minutes=30)
        else:raise ValueError('No comparison proxy-normal window')
        windows.append({'window_id':f'normal_{i+1}','role':'normal_visual','split':'test','module':m,
                        'start':str(normal),'end_exclusive':str(normal+pd.Timedelta(minutes=30)),
                        'selection':'first candidate-free 30-minute window at least two hours after visual candidate start'})
    rng=np.random.default_rng(SEED)
    ranges={'train':('2024-12-01','2025-03-01',12),
            'validation':('2025-03-01','2025-04-01',6),
            'test':('2025-04-01','2025-04-30',6)}
    for split,(a,b,n) in ranges.items():
        pool=pd.date_range(a,pd.Timestamp(b)-pd.Timedelta(minutes=30),freq='30min')
        for mi,m in enumerate(modules):
            taken=[]
            for ix in rng.permutation(len(pool)):
                start=pool[ix]
                if empty(m,start):taken.append(start)
                if len(taken)==n:break
            if len(taken)!=n:raise ValueError(f'Insufficient weak-normal base windows {m}/{split}')
            for j,start in enumerate(sorted(taken)):
                windows.append({'window_id':f'base_{split}_{mi:02d}_{j:02d}','role':'synthetic_base','split':split,
                                'module':m,'start':str(start),'end_exclusive':str(start+pd.Timedelta(minutes=30)),
                                'selection':'seeded module-stratified random candidate-free 30-minute source window'})
    manifest={'seed':SEED,'windows':windows,'source':str(SOURCE),
              'expected_rows_per_window':360,'cadence_seconds':5,'source_weak_candidate_rows':len(candidates),
              'normal_definition':'No exploratory-rule candidates in the complete 30-minute base; unknown real faults may remain',
              'candidate_rules':'phase voltage outside [205,225] V, phase PF outside [80,100] %, negative power/current, invalid operation',
              'synthetic_sampling':'12/6/6 windows per module train/validation/test, 30 minutes each; no source windows cross split boundaries',
              'event_summary':{'events':len(events),'candidate_rows':int(events.records.sum()),
                               'single_record_events':int(events.records.eq(1).sum()),
                               'single_record_event_fraction':float(events.records.eq(1).mean()),
                               'by_equipment':events.groupby('equipment').agg(events=('records','size'),rows=('records','sum'),single_record_events=('records',lambda x:int(x.eq(1).sum()))).reset_index().to_dict('records')}}
    dump(manifest,'window_manifest.json');print(f'Prepared {len(windows)} continuous windows',flush=True)
    return manifest

def extract(manifest,verification):
    """Single source pass, retaining true contiguous rows, plus every weak candidate."""
    allparts=[];cparts=[];seen=0;started=time.monotonic()
    bymod={}
    for w in manifest['windows']:
        bymod.setdefault(w['module'],[]).append((localkey(w['start']),localkey(w['end_exclusive']),w['window_id']))
    for ci,d in enumerate(pd.read_csv(SOURCE,chunksize=300000)):
        seen+=len(d);mask=weak(d)
        if mask.any():
            p=d.loc[mask.astype(bool)].copy();p['weak_candidate']=1;cparts.append(p)
        lk=d.localtime.to_numpy(dtype=np.int64)
        for m,idx in d.groupby(KEY,sort=False).indices.items():
            for a,b,wid in bymod.get(m,[]):
                ix=idx[(lk[idx]>=a)&(lk[idx]<b)]
                if len(ix):
                    p=d.iloc[ix].copy();p['window_id']=wid;allparts.append(p)
        if ci%15==0:print(f'raw pass {seen:,} rows, {time.monotonic()-started:.1f}s',flush=True)
    frames=pd.concat(allparts,ignore_index=True)
    frames['dt']=pd.to_datetime(frames.localtime.astype(str),format='%Y%m%d%H%M%S')
    frames['weak_candidate']=weak(frames)
    frames=frames.sort_values(['window_id','dt']).reset_index(drop=True)
    assert len(frames)==len(manifest['windows'])*360
    for wid,d in frames.groupby('window_id'):
        assert len(d)==360 and d.dt.diff().dropna().eq(pd.Timedelta(seconds=5)).all(),wid
        w=next(w for w in manifest['windows'] if w['window_id']==wid)
        assert str(d.dt.min())==w['start'] and str(d.dt.max()+pd.Timedelta(seconds=5))==w['end_exclusive']
        if w['role']!='candidate_visual':assert d.weak_candidate.sum()==0,wid
    cand=pd.concat(cparts,ignore_index=True)
    assert len(cand)==manifest['source_weak_candidate_rows']==63963
    cand['dt']=pd.to_datetime(cand.localtime.astype(str),format='%Y%m%d%H%M%S')
    cand['split']=np.select([cand.dt<'2025-03-01',cand.dt<'2025-04-01'],['train','validation'],default='test')
    frames.to_csv(OUT/'continuous_source_windows.csv.gz',index=False,compression='gzip',float_format='%.17g')
    cand.to_csv(OUT/'full_weak_candidate_rows.csv.gz',index=False,compression='gzip',float_format='%.17g')
    dump({'source':str(SOURCE),'source_verification':verification,'source_rows_read':seen,
          'retained_continuous_window_rows':len(frames),'full_weak_candidate_rows':len(cand),
          'continuity_all_windows_passed':True,'elapsed_seconds':time.monotonic()-started},'extraction_audit.json')
    return frames,cand

def legacy_models():
    import torch
    from torch import nn
    from xgboost import XGBClassifier
    class AE(nn.Module):
        def __init__(self,p):
            super().__init__();self.network=nn.Sequential(nn.Linear(p,32),nn.ReLU(),nn.Linear(32,12),nn.ReLU(),nn.Linear(12,4),nn.ReLU(),nn.Linear(4,12),nn.ReLU(),nn.Linear(12,32),nn.ReLU(),nn.Linear(32,p))
        def forward(self,x):return self.network(x)
    meta=readjson(OLD/'diagnosis_model_metrics.json');models={};provenance={}
    for name in LEGACY:
        info=meta[name];metrics=info.get('test',info.get('test_weak_proxy_agreement'))
        files=[f'{name}.json'] if name.startswith('xgb') else [f'{name}.pt',f'{name}_scaler.pkl']
        provenance[name]={}
        for fn in files:
            src=OLD/fn;dst=OUT/'models'/fn;shutil.copy2(src,dst)
            provenance[name][fn]={'source':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest()}
        if name.startswith('xgb'):
            model=XGBClassifier(n_jobs=4);model.load_model(OUT/'models'/f'{name}.json');model.set_params(n_jobs=4);kind='xgb';scaler=None
        else:
            ck=torch.load(OUT/'models'/f'{name}.pt',map_location='cpu',weights_only=True)
            assert ck['features']==SENSORS and float(ck['threshold'])==float(metrics['threshold'])
            model=AE(len(SENSORS));model.load_state_dict(ck['state_dict']);model.eval();kind='ae'
            with (OUT/'models'/f'{name}_scaler.pkl').open('rb') as f:scaler=pickle.load(f)
        models[name]={'model':model,'kind':kind,'scaler':scaler,'features':info['features'],
                      'threshold':float(metrics['threshold']),'threshold_basis':info['threshold_selection'],
                      'training':'Existing frozen model; original Dec–Feb training, March threshold, no new fitting',
                      'legacy_metric_reference':metrics}
    dump(provenance,'legacy_model_provenance.json');return models

def score(d,saved):
    values=d[saved['features']]
    kind=saved['kind']
    if kind=='xgb':return np.asarray(saved['model'].predict_proba(values)[:,1],dtype=np.float64)
    if kind=='if':return -saved['model'].score_samples(values).astype(np.float64)
    if kind=='z':return np.abs((values.to_numpy(dtype=float)-saved['median'])/saved['mad_scale']).max(axis=1)
    import torch
    x=saved['scaler'].transform(values).astype(np.float32);parts=[]
    with torch.no_grad():
        for start in range(0,len(x),20000):
            t=torch.from_numpy(x[start:start+20000]);parts.append((saved['model'](t)-t).pow(2).mean(dim=1).numpy().astype(np.float64))
    return np.concatenate(parts)

def metrics(y,p,threshold):
    from sklearn.metrics import average_precision_score,roc_auc_score,confusion_matrix
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);a=p>=threshold
    tn,fp,fn,tp=map(int,confusion_matrix(y,a,labels=[0,1]).ravel())
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
    return {'rows':len(y),'positive_rows':int(y.sum()),'positive_rate':float(y.mean()),
            'average_precision':float(average_precision_score(y,p)) if y.sum() else None,
            'auroc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.,
            'false_positive_rate':fp/(fp+tn) if fp+tn else 0.,'true_positive':tp,'false_positive':fp,
            'false_negative':fn,'true_negative':tn,'alarm_rows':int(a.sum()),'threshold':float(threshold)}

def synthesize(windows,manifest,train):
    """Perturb actual ordered base windows, preserving source keys and exact mask.

    Features are measurement perturbations, not a validated physical fault model.
    Sensor-only detectors see each row; they cannot establish temporal drift cause.
    """
    scales=(train[SENSORS].quantile(.75)-train[SENSORS].quantile(.25)).to_dict()
    parts=[];events=[]
    for wi,w in enumerate([w for w in manifest['windows'] if w['role']=='synthetic_base']):
        base=windows[windows.window_id==w['window_id']].copy().reset_index(drop=True)
        for ti,typ in enumerate(['spike','dip','gradual_drift','level_shift']):
            seed=SEED+wi*100+ti;rng=np.random.default_rng(seed)
            feature=SENSORS[(wi+ti*3)%len(SENSORS)]
            # Test strengths span below and above training amplitudes.
            strength=float(rng.uniform(1.5,4.0)) if w['split']=='train' else float([.5,1.,2.,4.][(wi+ti)%4])
            duration=1 if typ in ['spike','dip'] else int(rng.integers(12,37))
            start=int(rng.integers(60,len(base)-duration-60));stop=start+duration
            direction=-1 if typ=='dip' else (1 if typ=='spike' else int(rng.choice([-1,1])))
            amount=direction*strength*max(float(scales[feature]),1e-6)
            profile=np.linspace(1/duration,1,duration) if typ=='gradual_drift' else np.ones(duration)
            d=base.copy();changed=d.loc[start:stop-1,feature].to_numpy()+amount*profile
            clipped=int(np.sum(changed<0));d.loc[start:stop-1,feature]=np.maximum(changed,0)
            eid=f'{w["window_id"]}_{typ}';d['event_id']=eid;d['synthetic_type']=typ;d['synthetic_label']=0
            d.loc[start:stop-1,'synthetic_label']=1;d['split']=w['split'];d['perturbed_feature']=feature
            d['base_value']=base[feature].to_numpy();d['injected_delta']=d[feature].to_numpy()-d.base_value.to_numpy()
            events.append({'event_id':eid,'split':w['split'],'base_window_id':w['window_id'],'module':w['module'],
                           'source_base_start':w['start'],'source_base_end_exclusive':w['end_exclusive'],
                           'seed':seed,'type':typ,'feature':feature,'start_localtime':int(d.loc[start,'localtime']),
                           'end_exclusive':str(d.loc[stop-1,'dt']+pd.Timedelta(seconds=5)),
                           'start_offset':start,'stop_offset':stop,'duration_seconds':duration*5,'mask_rows':duration,
                           'signed_amplitude_iqr':direction*strength,'signed_peak_delta_original_unit':amount,
                           'training_iqr_original_unit':float(scales[feature]),'nonnegative_clip_rows':clipped,
                           'generation':'Additive single-sensor measurement perturbation; drift is an ordered ramp, other sensors not physically re-simulated'})
            parts.append(d)
    synt=pd.concat(parts,ignore_index=True)
    synt.to_csv(OUT/'synthetic_observations.csv.gz',index=False,compression='gzip',float_format='%.17g')
    pd.DataFrame(events).to_csv(OUT/'augmented_sample_manifest.csv',index=False,float_format='%.17g')
    dump({'seed':SEED,'events':events,'features':SENSORS,'train_iqr':scales,
          'train_amplitude_iqr':'uniform [1.5,4.0]','validation_test_amplitude_iqr':[.5,1.,2.,4.],
          'real_fault_ground_truth':False,'point_label_semantics':'Known synthetic injection mask only',
          'scope':'Injected measurement perturbations on previously weak-rule-normal source windows',
          'split_isolation':'Base dates disjoint, seeds disjoint; templates shared, no test-driven choice'},'augmented_sample_manifest.json')
    return synt,events

def train_new(train,valid,synt):
    from sklearn.ensemble import IsolationForest
    from xgboost import XGBClassifier
    tr=train[train.weak_candidate==0];va=valid[valid.weak_candidate==0];models={}
    isolation=IsolationForest(n_estimators=256,max_samples=1024,contamination='auto',n_jobs=4,random_state=SEED)
    isolation.fit(tr[SENSORS]);models['isolation_forest']={'model':isolation,'kind':'if','features':SENSORS,
                'training':'Fit Dec–Feb weak-normal proxy only; no candidate oversampling; semi-supervised exclusion'}
    with (OUT/'models'/'isolation_forest.pkl').open('wb') as f:pickle.dump(isolation,f)
    med=tr[SENSORS].median().to_numpy(dtype=float);mad=np.median(np.abs(tr[SENSORS].to_numpy()-med),axis=0)*1.4826
    mad=np.where(mad>1e-8,mad,1.)
    models['robust_z']={'kind':'z','features':SENSORS,'median':med,'mad_scale':mad,
                        'training':'Dec–Feb weak-normal median/MAD; max absolute 14-sensor robust z'}
    dump({'features':SENSORS,'median':med.tolist(),'mad_scale':mad.tolist()},'models/robust_z.json')
    inj=synt[(synt.split=='train')&(synt.synthetic_label==1)]
    vj=synt[(synt.split=='validation')&(synt.synthetic_label==1)]
    x=pd.concat([tr[SENSORS],inj[SENSORS]],ignore_index=True);y=np.r_[np.zeros(len(tr)),np.ones(len(inj))]
    xv=pd.concat([va[SENSORS],vj[SENSORS]],ignore_index=True);yv=np.r_[np.zeros(len(va)),np.ones(len(vj))]
    model=XGBClassifier(n_estimators=700,max_depth=5,learning_rate=.055,min_child_weight=8,
                         subsample=.9,colsample_bytree=.9,reg_lambda=5.,tree_method='hist',n_jobs=4,
                         random_state=SEED,eval_metric='aucpr',early_stopping_rounds=40,
                         scale_pos_weight=float(len(tr)/len(inj)))
    model.fit(x,y,eval_set=[(xv,yv)],verbose=False);model.save_model(OUT/'models'/'xgb_synthetic_augmented.json')
    models['xgb_synthetic_augmented']={'model':model,'kind':'xgb','features':SENSORS,
                 'training':'Dec–Feb original weak-normal negative rows plus known synthetic injected positives; no physical rule-positive training labels',
                 'training_negative_rows':len(tr),'training_injected_positive_rows':len(inj),
                 'best_iteration':int(model.best_iteration),'early_stopping':'March original weak-normal + known March synthetic rows, aucpr; no April use'}
    for name,saved in models.items():
        q=float(np.quantile(score(va,saved),.995));saved['threshold']=float(np.nextafter(q,np.inf))
        saved['threshold_basis']='March original weak-normal scores 99.5th percentile, nextafter(float64,+inf) for strict tie handling; frozen before April'
    return models

def evaluate(sample,windows,candidates,synt,events,models,manifest,verification):
    report={'status':'complete','seed':SEED,'source':readjson(OUT/'extraction_audit.json'),
            'protocol':{'split':'Dec–Feb training / March validation / April 1–29 test',
                        'fpr_target':.005,'threshold':'March ORIGINAL weak-normal q99.5, not synthetic/test tuned',
                        'legacy_threshold':'Existing March locks preserved exactly',
                        'time_axis':'Guide localtime; Unix timestamp timezone mismatch is not corrected heuristically',
                        'sample':'187200 original 4/hour/module uniform source observations; not continuous line-chart input',
                        'window_sampling':manifest['synthetic_sampling'],
                        'known_fault_labels':False,'target':'Current 5-second electrical-quality candidates and artificial measurement perturbations'},
            'event_summary':manifest['event_summary'],'windows':manifest['windows'],'models':{},'synthetic':{},
            'limitations':['Exploratory thresholds and reconstruction alarms do not confirm real faults or insertion locations',
                           'Full candidate-only recall has no normal denominator; precision comes only from representative original test sample',
                           'Synthetic test has constructed event/type/amplitude prevalence; its precision is not a real-world estimate',
                           'Sensor-only row models do not use temporal order to identify drift cause; ramp labels evaluate amplitude response along a real ordered source window',
                           'Other sensors are not re-simulated after one sensor perturbation; synthetic data test robustness, not validated physical failures',
                           'Weak-normal training/calibration can retain unknown faults; all new detectors are proxy semi-supervised',
                           'Representative windows selected using exploratory candidates are illustrations, not independent performance evaluation',
                           'The original held-out April data have already been inspected previously; this follow-up is not a pristine prospective test']}
    tr=sample[sample.split=='train'];va=sample[sample.split=='validation'];te=sample[sample.split=='test']
    original_predictions=te[[KEY,'localtime','dt','weak_candidate']].copy()
    window_predictions=windows[[KEY,'localtime','dt','window_id','weak_candidate']].copy()
    candpred=candidates[[KEY,'localtime','dt','split']].copy()
    st=synt[synt.split=='test'].copy().reset_index(drop=True)
    synthetic_predictions=st[['event_id',KEY,'localtime','dt','synthetic_type','perturbed_feature','synthetic_label']].copy()
    locks={};rowbyid={e['event_id']:e for e in events}
    base_index=windows.set_index(['window_id','localtime'])
    # Reconstruct paired original controls for exactly the same event points.
    control_keys=pd.MultiIndex.from_arrays([st.window_id.to_numpy(),st.localtime.to_numpy()],names=['window_id','localtime'])
    controls=base_index[SENSORS].reindex(control_keys).reset_index(drop=True)
    assert len(controls)==len(st) and np.isfinite(controls.to_numpy()).all()
    for name,saved in models.items():
        threshold=saved['threshold'];p=score(te,saved);vp=score(va,saved);wp=score(windows,saved);cp=score(candidates,saved)
        sp=score(st,saved);bp=score(controls,saved)
        info={k:v for k,v in saved.items() if k not in ['model','scaler','median','mad_scale']}
        info['display_name']=NAMES[name]
        info['original_weak_agreement']=metrics(te.weak_candidate,p,threshold)
        info['validation_weak_agreement']=metrics(va.weak_candidate,vp,threshold)
        info['all_source_candidate_capture']={split:{'rows':int((candidates.split==split).sum()),
                                    'captured':int((cp[candidates.split==split]>=threshold).sum()),
                                    'capture_rate':float((cp[candidates.split==split]>=threshold).mean())} for split in ['train','validation','test']}
        info['synthetic_heldout_metrics']=metrics(st.synthetic_label,sp,threshold)
        info['synthetic_by_type']={typ:metrics(d.synthetic_label,sp[d.index],threshold) for typ,d in st.groupby('synthetic_type')}
        eventrows=[]
        for eid,idx in st.groupby('event_id',sort=False).indices.items():
            em=rowbyid[eid];m=st.iloc[idx].synthetic_label.to_numpy().astype(bool)
            sa=sp[idx][m]>=threshold;ba=bp[idx][m]>=threshold
            hit=np.flatnonzero(sa)
            eventrows.append({'event_id':eid,'type':em['type'],'feature':em['feature'],
                              'amplitude_iqr':abs(em['signed_amplitude_iqr']),'duration_seconds':em['duration_seconds'],
                              'detected_any':bool(sa.any()),'original_control_alarm_any':bool(ba.any()),
                              'injected_point_recall':float(sa.mean()),'original_control_point_alarm_rate':float(ba.mean()),
                              'first_alarm_delay_seconds':int(hit[0]*5) if len(hit) else None})
        eframe=pd.DataFrame(eventrows)
        info['synthetic_event_detection']={'events':len(eframe),'detected_any_fraction':float(eframe.detected_any.mean()),
                'paired_original_control_any_alarm_fraction':float(eframe.original_control_alarm_any.mean()),
                'any_alarm_limit':'Longer events have more chances to trigger incidental alarms; compare paired original controls',
                'by_type':eframe.groupby('type').agg(events=('event_id','size'),detected_any_fraction=('detected_any','mean'),paired_original_any_alarm_fraction=('original_control_alarm_any','mean'),mean_injected_point_recall=('injected_point_recall','mean')).reset_index().to_dict('records'),
                'by_amplitude_iqr':eframe.groupby('amplitude_iqr').agg(events=('event_id','size'),detected_any_fraction=('detected_any','mean'),mean_injected_point_recall=('injected_point_recall','mean')).reset_index().to_dict('records')}
        eframe.to_csv(OUT/f'synthetic_event_results_{name}.csv',index=False,float_format='%.17g')
        for df,sc in [(original_predictions,p),(window_predictions,wp),(candpred,cp),(synthetic_predictions,sp)]:
            df[f'{name}_score']=sc;df[f'{name}_alarm']=(sc>=threshold).astype(np.int8)
        synthetic_predictions[f'{name}_paired_original_score']=bp
        report['models'][name]=info;locks[name]={'threshold':threshold,'threshold_basis':saved['threshold_basis'],'alarm_comparison':'float64 score >= float64 threshold','features':saved['features'],'kind':saved['kind']}
        print(f'{name}: original precision {info["original_weak_agreement"]["precision"]:.4f}; synthetic point recall {info["synthetic_heldout_metrics"]["recall"]:.4f}',flush=True)
    # Median suppression is post-hoc illustration, never used to select a model.
    medrows=[]
    for wid,d in windows.groupby('window_id'):
        raw=weak(d);smooth=d.copy()
        smooth[SENSORS]=d[SENSORS].rolling(5,min_periods=5).median().to_numpy()
        available=np.arange(len(d))>=4;filtered=weak(smooth)
        medrows.append({'window_id':wid,'eligible_rows':int(available.sum()),'raw_candidate_rows':int(raw[available].sum()),
                        'causal_5point_median_candidate_rows':int(filtered[available].sum()),
                        'raw_candidates_removed':int(((raw==1)&(filtered==0)&available).sum()),
                        'note':'25-second trailing median changes the observation; isolated low values can disappear'})
    report['median_suppression']=medrows
    report['synthetic']={'manifest':'augmented_sample_manifest.json','observations':'synthetic_observations.csv.gz',
                        'events_by_split':pd.DataFrame(events).groupby('split').size().to_dict(),
                        'test_rows':len(st),'injected_test_rows':int(st.synthetic_label.sum()),
                        'test_injection_prevalence':float(st.synthetic_label.mean()),
                        'injection_truth':'Known synthetic mask only; not original dataset insertion ground truth'}
    for df,name in [(original_predictions,'original_test_scores.csv.gz'),(window_predictions,'continuous_window_scores.csv.gz'),
                    (candpred,'full_candidate_scores.csv.gz'),(synthetic_predictions,'synthetic_test_scores.csv.gz')]:
        df.to_csv(OUT/name,index=False,compression='gzip',float_format='%.17g')
    dump(locks,'threshold_locks.json');dump(report,'anomaly_metrics.json')
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--source-verification',help='Read-only JSON from parent audit confirming source identity')
    parser.add_argument('--source-confirmed',action='store_true',help='Parent explicitly confirmed archive/source identity; required for execution')
    args=parser.parse_args()
    manifest=prepare() if not (OUT/'window_manifest.json').exists() else readjson(OUT/'window_manifest.json')
    if args.prepare_only:return
    if not args.source_confirmed:parser.error('Wait for parent same-source confirmation; then pass --source-confirmed')
    verification=readjson(args.source_verification) if args.source_verification else {'confirmation':'Parent explicitly confirmed same source before run'}
    if args.source_verification:
        assert verification.get('all_byte_identical') is True, 'Source audit must confirm exact byte identity'
    if (OUT/'continuous_source_windows.csv.gz').exists():
        windows=pd.read_csv(OUT/'continuous_source_windows.csv.gz',parse_dates=['dt'])
        cand=pd.read_csv(OUT/'full_weak_candidate_rows.csv.gz',parse_dates=['dt'])
    else:windows,cand=extract(manifest,verification)
    sample=pd.read_csv(OLD/'uniform_raw_sample.csv.gz',parse_dates=['dt'])
    train=sample[sample.split=='train'];valid=sample[sample.split=='validation']
    import torch
    torch.set_num_threads(4);torch.manual_seed(SEED)
    models=legacy_models()
    if (OUT/'synthetic_observations.csv.gz').exists() and (OUT/'augmented_sample_manifest.json').exists():
        synt=pd.read_csv(OUT/'synthetic_observations.csv.gz',parse_dates=['dt'])
        events=readjson(OUT/'augmented_sample_manifest.json')['events']
        print('Reused frozen synthetic manifest and observations',flush=True)
    else:synt,events=synthesize(windows,manifest,train[train.weak_candidate==0])
    models.update(train_new(train,valid,synt))
    report=evaluate(sample,windows,cand,synt,events,models,manifest,verification)
    print(f'Complete {len(models)} models; {len(report["windows"])} continuous windows',flush=True)

if __name__=='__main__':main()
