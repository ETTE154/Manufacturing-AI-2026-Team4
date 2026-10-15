"""Train-prefix-selected multivariate official-core iTransformer and PatchTST.

All new artifacts stay below this neural folder. Old core/vendor files are read
only. This is compact local retraining, not paper benchmark reproduction. No May
labels are read. Configurations and epochs lock on chronological validation.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace
import argparse,copy,hashlib,json,os,random,time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader

OUT=Path(__file__).resolve().parent
FEATURE=OUT.parent
ROOT=FEATURE.parents[1]
OLD=ROOT/'output'/'modeling'/'forecast'
sys.path.insert(0,str(FEATURE));sys.path.insert(0,str(OLD));sys.path.insert(0,str(OLD/'patchtst'))
from feature_common import load_hourly,feature_columns,select_sensor_features,TARGET
from patchtst_vendor.models.PatchTST import Model as PatchCore
from itransformer.vendor.model.iTransformer import Model as InvertedCore

SEED=20261013;THREADS=4;MAX_EPOCHS=16;PATIENCE=3;BATCH=64
DAY_STARTS=['2025-01-10','2025-01-24','2025-02-07','2025-02-21','2025-03-07']
DAY_CONFIGS=[
 {'group':'module_energy','context':24,'d_model':32,'layers':1,'loss':'MSE','instance_norm':False,'dropout':.1,'output_penalty':.03},
 {'group':'module_energy','context':72,'d_model':32,'layers':1,'loss':'MSE','instance_norm':True,'dropout':.1,'output_penalty':.03},
 {'group':'module_energy','context':168,'d_model':32,'layers':1,'loss':'L1','instance_norm':False,'dropout':.2,'output_penalty':.05},
 {'group':'selected_sensors','context':24,'d_model':32,'layers':1,'loss':'MSE','instance_norm':False,'dropout':.1,'output_penalty':.03},
 {'group':'selected_sensors','context':72,'d_model':32,'layers':1,'loss':'MSE','instance_norm':True,'dropout':.1,'output_penalty':.03},
 {'group':'selected_sensors','context':168,'d_model':32,'layers':1,'loss':'L1','instance_norm':False,'dropout':.2,'output_penalty':.05},
]
LONG_CONFIGS=[
 {'group':'module_energy','context':24,'d_model':32,'layers':1,'loss':'MSE','instance_norm':False,'dropout':.1,'output_penalty':.03},
 {'group':'module_energy','context':72,'d_model':32,'layers':1,'loss':'L1','instance_norm':False,'dropout':.2,'output_penalty':.05},
]

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readjson(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def seed():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    torch.set_num_threads(THREADS);torch.use_deterministic_algorithms(True)

def core_provenance(family):
    path=OLD/('patchtst/provenance.json' if family=='patchtst' else 'itransformer/vendor/provenance.json')
    p=readjson(path)
    for r in p['files']:
        file=Path(r['vendored_path']) if family=='patchtst' else OLD/'itransformer'/'vendor'/r['path']
        expected=r['vendored_sha256'] if family=='patchtst' else r['local_sha256']
        assert sha(file)==expected,(family,file,'core changed')
    return {'provenance_file':str(path),'sha256':sha(path),'official_core_hashes_verified':True,'record':p,
            'claim':'Pinned official core with existing namespace/lazy unused-import changes only; compact local retraining'}

def architecture(family,cfg,points,n):
    common=dict(seq_len=cfg['context'],pred_len=points,d_model=cfg['d_model'],d_ff=cfg['d_model']*2,e_layers=cfg['layers'],n_heads=4,dropout=cfg['dropout'],enc_in=n)
    if family=='patchtst':
        return {**common,'fc_dropout':cfg['dropout'],'head_dropout':cfg['dropout'],'individual':False,
                'patch_len':16,'stride':8,'padding_patch':'end','revin':cfg['instance_norm'],
                'affine':True,'subtract_last':False,'decomposition':False,'kernel_size':25,'core_norm':'LayerNorm'}
    return {**common,'output_attention':False,'use_norm':cfg['instance_norm'],'embed':'timeF','freq':'h',
            'class_strategy':'projection','factor':5,'activation':'gelu'}

def build(family,arch):
    return PatchCore(SimpleNamespace(**arch),norm=arch['core_norm']).cpu() if family=='patchtst' else InvertedCore(SimpleNamespace(**arch)).cpu()
def forward(family,model,x):return model(x) if family=='patchtst' else model(x,None,None,None)

def metrics(actual,pred):
    y=np.asarray(actual,float);p=np.asarray(pred,float);e=p-y;denom=np.sum((y-y.mean())**2)
    assert y.shape==p.shape and np.isfinite(y).all() and np.isfinite(p).all()
    return {'count':len(y),'mae_kwh':float(np.mean(np.abs(e))),'rmse_kwh':float(np.sqrt(np.mean(e**2))),
            'mape_percent':float(np.mean(np.abs(e)/np.abs(y))*100),'bias_pred_minus_actual_kwh':float(e.mean()),
            'r2':float(1-np.sum(e**2)/denom) if denom>0 else None}

def summary(frame,points):
    result={'path_metrics':metrics(frame.actual_kwh,frame.predicted_kwh),
            'origin_count':int(frame.origin.nunique()),'mean_origin_path_mae_kwh':float(frame.groupby('origin').absolute_error_kwh.mean().mean()),
            'endpoint_metrics':{f'lead{lead}':metrics(d.actual_kwh,d.predicted_kwh) for lead in [1,6,24]
                                for d in [frame[frame.lead_hours==lead]]}}
    if points==745:
        d=frame[frame.horizon_offset_hours>=240];result['LateP505']=metrics(d.actual_kwh,d.predicted_kwh)
    return result

class Windows(Dataset):
    def __init__(self,values,context,points):
        self.values=torch.from_numpy(np.asarray(values,dtype=np.float32));self.context=context;self.points=points
        step=6 if points==24 else 24
        last=len(values)-points
        self.origins=np.arange(context,last+1,step,dtype=int)
        if last>=context and (not len(self.origins) or self.origins[-1]!=last):self.origins=np.r_[self.origins,last]
        if not len(self.origins):raise ValueError('Prefix too short for complete context and labels')
        assert self.origins[-1]+points==len(values)
    def __len__(self):return len(self.origins)
    def __getitem__(self,index):
        o=int(self.origins[index]);return self.values[o-self.context:o],self.values[o:o+self.points]

def prefix(frame,origin):
    h=frame.loc[frame.index<pd.Timestamp(origin)]
    assert not h.empty and h.index[-1]==pd.Timestamp(origin)-pd.Timedelta(hours=1)
    return h

SELECTION_CACHE={}
def schema(hist,group,dest):
    energies=feature_columns('module_energy');assert len(energies)==13
    selected=[];audit={'method':'Fixed known 13-module energy group, no label-based feature selection',
                       'training_end':str(hist.index[-1]),'selected_columns':[],'selected_count':0,'selection_uses_April_or_May':False}
    if group=='selected_sensors':
        key=str(hist.index[-1])
        if key not in SELECTION_CACHE:SELECTION_CACHE[key]=select_sensor_features(hist,max_features=12)
        selected,audit=SELECTION_CACHE[key]
        dump(dest/'selection_audits'/f'prefix_{hist.index[-1]:%Y%m%d%H}.json',audit)
    names=energies+list(selected)+[TARGET]
    assert len(set(names))==len(names) and names[-1]==TARGET
    record={'input_features':names,'input_feature_count':len(names),'target_column':TARGET,'target_index':len(names)-1,
            'module_energy_columns':energies,'module_energy_indices':list(range(13)),
            'selected_sensor_columns':list(selected),'selected_sensor_count':len(selected),'feature_group':group,
            'selection_audit':{k:v for k,v in audit.items() if k!='records'},
            'selection_fit_end':str(hist.index[-1]),'prediction_input':'Completed rows origin-context through origin-1h only'}
    return record

def setup(family,hist,cfg,points,sch):
    seed();values=hist[sch['input_features']].to_numpy(float)
    mean=values.mean(axis=0);scale=values.std(axis=0,ddof=0);scale=np.where(scale>1e-8,scale,1.)
    z=(values-mean)/scale;assert np.isfinite(z).all()
    ds=Windows(z,cfg['context'],points)
    loader=DataLoader(ds,batch_size=BATCH,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(SEED))
    arch=architecture(family,cfg,points,len(mean));model=build(family,arch)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    normal={'mean':mean.tolist(),'scale':scale.tolist(),'features':sch['input_features'],
            'fit_start':str(hist.index[0]),'fit_end':str(hist.index[-1]),'fit_rows':len(hist),'fit_scope':'Training prefix only',
            'near_constant_scale_fallback_count':int((values.std(axis=0)<=1e-8).sum())}
    return model,opt,loader,arch,normal,ds

def epoch(family,model,opt,loader,cfg,sch):
    model.train();total=0.;objective_total=0.;rows=0
    for x,y in loader:
        opt.zero_grad(set_to_none=True);pred=forward(family,model,x)
        assert pred.shape==y.shape and pred.shape[-1]==sch['input_feature_count']
        if family=='itransformer':
            idx=sch['target_index'];delta=pred[:,:,idx]-y[:,:,idx];penalty=pred[:,:,idx].pow(2).mean()
        else:
            # Each channel is standardized with its own TRAIN-prefix scale.
            # Sensor channels are auxiliary shared-weight targets, not cross-channel attention.
            delta=pred-y;penalty=pred.pow(2).mean()
        data_loss=delta.abs().mean() if cfg['loss']=='L1' else delta.pow(2).mean()
        objective=data_loss+cfg['output_penalty']*penalty
        assert torch.isfinite(objective);objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        total+=float(data_loss.detach())*len(x);objective_total+=float(objective.detach())*len(x);rows+=len(x)
    return {'data_loss_standardized':total/rows,'regularized_objective':objective_total/rows,'loss':cfg['loss']}

def infer(family,model,hist,arch,norm,sch):
    names=sch['input_features'];context=arch['seq_len'];assert len(hist)>=context
    assert norm['features']==names and arch['enc_in']==len(names) and 0<=sch['target_index']<len(names)
    x=((hist[names].iloc[-context:].to_numpy(float)-np.asarray(norm['mean']))/np.asarray(norm['scale'])).astype(np.float32)
    tensor=torch.from_numpy(x).unsqueeze(0);model.eval()
    with torch.no_grad():out=forward(family,model,tensor)
    assert tuple(out.shape)==(1,arch['pred_len'],len(names)),tuple(out.shape)
    channels=out[0].numpy().astype(np.float64)*np.asarray(norm['scale'])+np.asarray(norm['mean'])
    pred=channels[:,sch['target_index']] if family=='itransformer' else channels[:,sch['module_energy_indices']].sum(axis=1)
    assert np.isfinite(pred).all()
    return pred,channels

def frame_at(family,scenario,split,fold,full,origin,model,arch,norm,sch):
    h=prefix(full,origin);pred,channels=infer(family,model,h,arch,norm,sch)
    idx=pd.date_range(origin,periods=arch['pred_len'],freq='h')
    assert idx.isin(full.index).all();actual=full.loc[idx,TARGET].to_numpy(float)
    result=pd.DataFrame({'model':family,'scenario':scenario,'split':split,'fold':fold,'origin':str(pd.Timestamp(origin)),
                         'id':idx,'horizon_offset_hours':np.arange(len(idx)),'lead_hours':np.arange(1,len(idx)+1),
                         'actual_kwh':actual,'predicted_kwh':pred,'absolute_error_kwh':np.abs(pred-actual)})
    assert h.index[-1]<idx[0]
    return result

def choose_epochs(family,hist,cfg,points,dest):
    hold=168 if points==24 else 745;fit=hist.iloc[:-hold];tail=hist.iloc[-hold:]
    sch=schema(fit,cfg['group'],dest);model,opt,loader,arch,norm,ds=setup(family,fit,cfg,points,sch)
    origins=pd.date_range(tail.index[0],periods=7,freq='D') if points==24 else [tail.index[0]]
    best=float('inf');best_epoch=1;stale=0;logs=[];started=time.perf_counter()
    for e in range(1,MAX_EPOCHS+1):
        loss=epoch(family,model,opt,loader,cfg,sch);errors=[]
        for origin in origins:
            pred,_=infer(family,model,prefix(hist,origin),arch,norm,sch)
            idx=pd.date_range(origin,periods=points,freq='h');errors.append(metrics(hist.loc[idx,TARGET],pred)['mae_kwh'])
        mae=float(np.mean(errors));logs.append({'epoch':e,**loss,'inner_tail_mae_kwh':mae})
        if mae<best-1e-7:best=mae;best_epoch=e;stale=0
        else:stale+=1
        if stale>=PATIENCE:break
    return best_epoch,{'inner_fit_end':str(fit.index[-1]),'inner_fit_rows':len(fit),'inner_tail_start':str(tail.index[0]),
                       'inner_tail_end':str(tail.index[-1]),'inner_holdout_rows':hold,'schema':sch,'normalizer':norm,
                       'maximum_supervised_label':str(fit.index[-1]),'training_windows':len(ds),'best_epoch':best_epoch,
                       'best_inner_mae_kwh':best,'epochs_run':len(logs),'logs':logs,'seconds':time.perf_counter()-started,
                       'future_label_overlap':False,'feature_selection_and_scaler_exclude_inner_tail':True}

def save_model(path,family,model,arch,norm,sch,hist,epochs,source_sha):
    payload={'family':family,'state_dict':model.state_dict(),'architecture':arch,'normalizer':norm,'schema':sch,
             'epochs':int(epochs),'seed':SEED,'data_sha256':source_sha,'history_start':str(hist.index[0]),
             'history_end':str(hist.index[-1]),'history_rows':len(hist),
             'forecast_context':hist[sch['input_features']].iloc[-arch['seq_len']:].to_numpy(float).tolist(),
             'forecast_context_time':[str(t) for t in hist.index[-arch['seq_len']:]],
             'target_mode':'aggregate channel selected by explicit target_index' if family=='itransformer' else 'sum of 13 independently forecast module energy channels after per-channel inverse scaling',
             'units':'provisional guide kWh, meter hierarchy unconfirmed'}
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,path)
    return {'path':str(path),'sha256':sha(path),'parameters':sum(p.numel() for p in model.parameters()),'epochs':int(epochs),
            'history_end':str(hist.index[-1]),'history_rows':len(hist),'input_features':sch['input_features'],
            'input_feature_count':len(sch['input_features']),'target_index':sch['target_index'],'schema':sch,'normalizer':norm,
            'target_mode':payload['target_mode'],'forecast_context_stored':True}

def reload(path):
    seed();p=torch.load(path,map_location='cpu',weights_only=True);model=build(p['family'],p['architecture'])
    model.load_state_dict(p['state_dict'],strict=True);model.eval();return model,p

def refit(family,hist,cfg,points,epochs,dest,path,source_sha,sch=None):
    started=time.perf_counter();sch=schema(hist,cfg['group'],dest) if sch is None else copy.deepcopy(sch)
    model,opt,loader,arch,norm,ds=setup(family,hist,cfg,points,sch);logs=[]
    for _ in range(epochs):logs.append(epoch(family,model,opt,loader,cfg,sch))
    record=save_model(path,family,model,arch,norm,sch,hist,epochs,source_sha)
    record.update({'fit_seconds':time.perf_counter()-started,'training_loss_logs':logs,'training_windows':len(ds),
                   'maximum_supervised_label':str(hist.index[-1]),'normalizer_fit_matches_training_prefix':norm['fit_end']==str(hist.index[-1]),
                   'target_loss':'aggregate target only, other variables are input covariates' if family=='itransformer' else 'uniform per-channel standardized MSE/L1, sensors auxiliary shared-parameter targets',
                   'schema_frozen_before_April':sch['selection_fit_end']<'2025-04-01'})
    return model,arch,norm,sch,record

def tensor_checks(family,model,hist,arch,norm,sch):
    p,channels=infer(family,model,hist,arch,norm,sch)
    assert np.allclose(p,channels[:,sch['target_index']] if family=='itransformer' else channels[:,:13].sum(axis=1),rtol=0,atol=1e-10)
    values=hist[sch['input_features']].iloc[-arch['seq_len']:].to_numpy(float)
    mean=np.asarray(norm['mean']);scale=np.asarray(norm['scale']);z=(values-mean)/scale
    inverse_error=float(np.abs(z*scale+mean-values).max());assert inverse_error<1e-8
    # Non-target module perturbation: measures inferred cross-variable sensitivity,
    # not feature importance proof or guaranteed positive attention use.
    altered=hist.copy();col=sch['module_energy_columns'][0]
    altered.loc[altered.index[-arch['seq_len']:],col]+=scale[0]*np.linspace(-2,2,arch['seq_len'])
    pp,cc=infer(family,model,altered,arch,norm,sch)
    other=np.array([i for i in range(len(sch['input_features'])) if i!=0])
    ci_error=float(np.abs(channels[:,other]-cc[:,other]).max())
    if family=='patchtst':assert ci_error<1e-8,'Official CI inference unexpectedly mixes current channels'
    norm_error=None
    enabled=arch.get('use_norm',arch.get('revin',False))
    if enabled:
        # Constant shifts of every channel must be undone using THAT channel's
        # input-only mean; target_index may never accidentally select module0.
        shifted=hist.copy();shifts=np.linspace(.15,.7,len(mean))*scale
        shifted.loc[shifted.index[-arch['seq_len']:],sch['input_features']]+=shifts
        _,shift_channels=infer(family,model,shifted,arch,norm,sch)
        norm_error=float(np.abs((shift_channels-channels)-shifts).max())
        assert norm_error<.005,'Per-channel instance denormalization mismatch'
    return {'output_shape':[1,arch['pred_len'],len(mean)],'target_index':sch['target_index'],'target_name':sch['input_features'][sch['target_index']],
            'aggregate_output_alignment_verified':True,'global_scaler_inverse_max_error':inverse_error,
            'non_target_module_perturbation_aggregate_max_change_kwh':float(np.abs(pp-p).max()),
            'other_output_channels_max_change_kwh':ci_error,
            'patch_channel_independence_verified':family=='patchtst',
            'input_only_instance_norm_enabled':bool(enabled),'per_channel_shift_inverse_max_error_kwh':norm_error}

def run(family,scenario,resume=False):
    started=time.perf_counter();seed();full=load_hourly();energy=feature_columns('module_energy')
    assert len(full)==3600 and len(energy)==13
    assert np.abs(full[energy].sum(axis=1)-full[TARGET]).max()<1e-7
    data_sha=sha(FEATURE/'data'/'hourly_features.csv');vendor=core_provenance(family)
    dest=OUT/scenario/family;dest.mkdir(parents=True,exist_ok=True)
    points=24 if scenario=='dayahead24' else 745;configs=DAY_CONFIGS if points==24 else LONG_CONFIGS
    dump(dest/'search_plan.json',{'family':family,'scenario':scenario,'configs':configs,'data_sha256':data_sha,'vendor':vendor,
         'max_epochs':MAX_EPOCHS,'patience':PATIENCE,'batch_size':BATCH,'training_origin_stride_hours':6 if points==24 else 24,
         'input_scope':'13 module energies + explicit aggregate target + optional prefix-stable physical sensors',
         'feature_selection':'Independent inner/outer TRAIN prefixes; pretest schema (March31 day, March28 long) reused for final April29 weight/scaler fit',
         'day_CV':'Jan10/Jan24/Feb7/Feb21/Mar7 x14 daily origins, aggregate path MAE selects config; endpoints1/6/24 reported',
         'loss_definition':'Aggregate-target only for iTransformer; train-standardized uniform all-channel targets and module-sum output for PatchTST',
         'no_May_labels_read':True,'seed':SEED,'threads':THREADS,'post_EDA':True})
    candidate_results=[];frames=[];fit_cache={};inner_cache={}
    for ci,cfg in enumerate(configs,1):
        cpath=dest/f'candidate{ci}.json';csvpath=dest/f'candidate{ci}_validation.csv'
        if resume and cpath.exists() and csvpath.exists():
            item=readjson(cpath);assert item['config']==cfg and item['data_sha256']==data_sha
            cv=pd.read_csv(csvpath,float_precision='round_trip');candidate_results.append(item);frames.append(cv)
            print(f'REUSED {family} {scenario} candidate{ci}',flush=True);continue
        cvs=[];folds=[];starts=DAY_STARTS if points==24 else ['2025-02-25']
        for fi,start in enumerate(starts,1):
            hist=prefix(full,start);inner_schema=schema(hist.iloc[:-168 if points==24 else -745],cfg['group'],dest)
            signature=json.dumps({k:v for k,v in cfg.items() if k!='group'},sort_keys=True)+str(hist.index[-1])+'|'+','.join(inner_schema['input_features'])
            if signature in inner_cache:epochs,inner=inner_cache[signature];inner={**inner,'exact_duplicate_fit_reused':True}
            else:epochs,inner=choose_epochs(family,hist,cfg,points,dest);inner_cache[signature]=(epochs,inner)
            outer_schema=schema(hist,cfg['group'],dest)
            outerkey=json.dumps({k:v for k,v in cfg.items() if k!='group'},sort_keys=True)+str(hist.index[-1])+'|'+','.join(outer_schema['input_features'])+'|epochs='+str(epochs)
            if outerkey in fit_cache:
                model,arch,norm,sch,record=fit_cache[outerkey];record={**record,'exact_duplicate_fit_reused':True}
                # Preserve current feature-group interpretation for an empty selection.
                sch={**sch,'feature_group':cfg['group']}
            else:
                model,arch,norm,sch,record=refit(family,hist,cfg,points,epochs,dest,dest/'checkpoints'/f'candidate{ci}_fold{fi}.pt',data_sha,outer_schema)
                fit_cache[outerkey]=(model,arch,norm,sch,record)
            origins=pd.date_range(start,periods=14,freq='D') if points==24 else [pd.Timestamp(start)]
            for origin in origins:cvs.append(frame_at(family,scenario,'validation',fi,full,origin,model,arch,norm,sch))
            folds.append({'fold':fi,'origin':str(pd.Timestamp(start)),'epochs':epochs,'inner':inner,'refit':record,'input_features':sch['input_features']})
            print(f'FOLD {family} {scenario} candidate{ci} fold{fi}, channels={len(sch["input_features"])}, sensors={sch["selected_sensor_count"]}, epochs={epochs}',flush=True)
        cv=pd.concat(cvs,ignore_index=True);item={'candidate':ci,'config':cfg,'validation':summary(cv,points),'fold_fits':folds,'data_sha256':data_sha}
        cv.to_csv(csvpath,index=False,float_format='%.17g');dump(cpath,item);candidate_results.append(item);frames.append(cv)
        print(f'CANDIDATE {family} {scenario} {ci} CV_MAE={item["validation"]["path_metrics"]["mae_kwh"]:.6f}',flush=True)
    winner=min(range(len(configs)),key=lambda i:candidate_results[i]['validation']['mean_origin_path_mae_kwh'])
    chosen=candidate_results[winner];cfg=chosen['config'];cv=frames[winner]
    cv.to_csv(dest/'validation_predictions.csv',index=False,float_format='%.17g')
    lock={'family':family,'scenario':scenario,'selected_candidate':winner+1,'selected_config':cfg,'selection':'CV aggregate path MAE only',
          'validation':chosen['validation'],'uses_April_test_for_selection':False,'uses_May_for_selection':False,
          'data_sha256':data_sha,'vendor':vendor,'post_EDA_lock':True,'all_candidate_feature_schemas':[{'candidate':r['candidate'],'fold_schemas':r['fold_fits']} for r in candidate_results]}
    dump(dest/'selection_lock.json',lock)
    test_start=pd.Timestamp('2025-04-01' if points==24 else '2025-03-29');hist=prefix(full,test_start)
    if points==24:epochs,test_inner=choose_epochs(family,hist,cfg,points,dest)
    else:epochs=chosen['fold_fits'][0]['epochs'];test_inner={'policy':'Frozen long validation inner epoch choice reused; internal test does not select epochs'}
    lock['frozen_epochs']=epochs;lock['pretest_epoch_choice']=test_inner;dump(dest/'selection_lock.json',lock)
    test_model,arch,norm,sch,test_fit=refit(family,hist,cfg,points,epochs,dest,dest/'checkpoints'/'internal_test.pt',data_sha)
    origins=pd.date_range(test_start,periods=29,freq='D') if points==24 else [test_start]
    tests=[frame_at(family,scenario,'April_test' if points==24 else 'internal_test',0,full,o,test_model,arch,norm,sch) for o in origins]
    test=pd.concat(tests,ignore_index=True);test.to_csv(dest/'test_predictions.csv',index=False,float_format='%.17g')
    tm,saved=reload(dest/'checkpoints'/'internal_test.pt');reload_errors=[]
    for o in origins:
        p,_=infer(family,tm,prefix(full,o),saved['architecture'],saved['normalizer'],saved['schema'])
        expected=test.loc[pd.to_datetime(test.origin)==o,'predicted_kwh'].to_numpy(float)
        error=float(np.abs(p-expected).max());assert error<1e-8;reload_errors.append(error)
    # Poison future covariates/labels after a fixed origin; inference must be unchanged.
    poisoned=full.copy();poisoned.loc[poisoned.index>=test_start]=1e9
    p0,_=infer(family,tm,prefix(full,test_start),saved['architecture'],saved['normalizer'],saved['schema'])
    p1,_=infer(family,tm,prefix(poisoned,test_start),saved['architecture'],saved['normalizer'],saved['schema'])
    poison_error=float(np.abs(p0-p1).max());assert poison_error==0.
    checks=tensor_checks(family,tm,hist,saved['architecture'],saved['normalizer'],saved['schema'])
    # Final May-origin fit uses ALL known April history, frozen pre-April schema and epochs.
    final_sch=copy.deepcopy(sch);final_sch['final_schema_policy']='Reuse pre-April schema; do not reselect sensors with April test labels'
    fm,farch,fnorm,fsch,final_fit=refit(family,full,cfg,points,epochs,dest,dest/'saved_model.pt',data_sha,final_sch)
    origin=full.index[-1]+pd.Timedelta(hours=1);p,_=infer(family,fm,full,farch,fnorm,fsch)
    reloaded,fsaved=reload(dest/'saved_model.pt');p2,_=infer(family,reloaded,full,fsaved['architecture'],fsaved['normalizer'],fsaved['schema'])
    final_error=float(np.abs(p-p2).max());assert final_error<1e-8
    idx=pd.date_range(origin,periods=points,freq='h')
    future_frame=pd.DataFrame({'id':idx,'model':family,'scenario':scenario,'origin':str(origin),
                               'horizon_offset_hours':np.arange(points),'lead_hours':np.arange(1,points+1),'predicted_kwh':p})
    future_frame.to_csv(dest/'future_path.csv',index=False,float_format='%.17g')
    # Reload winning CV paths on all 70 origins with their own fold feature schemas.
    cv_replay=[]
    for fold in chosen['fold_fits']:
        vm,vs=reload(Path(fold['refit']['path']))
        subset=cv[cv.fold==fold['fold']]
        for o,d in subset.groupby('origin',sort=False):
            q,_=infer(family,vm,prefix(full,pd.Timestamp(o)),vs['architecture'],vs['normalizer'],vs['schema'])
            err=float(np.abs(q-d.sort_values('horizon_offset_hours').predicted_kwh.to_numpy()).max());assert err<1e-8;cv_replay.append(err)
    result={'status':'complete','family':family,'scenario':scenario,'selected_config':cfg,'selected_candidate':winner+1,
            'validation':chosen['validation'],'test':summary(test,points),'validation_metrics':chosen['validation']['path_metrics'],
            'test_metrics':summary(test,points)['path_metrics'],'validation_lead_metrics':chosen['validation']['endpoint_metrics'],
            'test_lead_metrics':summary(test,points)['endpoint_metrics'],'input_features':sch['input_features'],
            'input_feature_count':sch['input_feature_count'],'selected_sensor_count':sch['selected_sensor_count'],
            'target_index':sch['target_index'],'target_mode':final_fit['target_mode'],
            'cross_channel_or_auxiliary':'iTransformer attention across named variate tokens; aggregate target supervised' if family=='itransformer' else 'Official channel-independent PatchTST; all standardized channels train shared parameters, module predictions sum; sensors cannot influence other channels during inference',
            'loss_definition':test_fit['target_loss'],'candidate_results':candidate_results,'test_fit':test_fit,'final_fit':final_fit,
            'saved_model':str(dest/'saved_model.pt'),'selection_lock':str(dest/'selection_lock.json'),
            'prediction_csv':{'validation':str(dest/'validation_predictions.csv'),'test':str(dest/'test_predictions.csv'),'future':str(dest/'future_path.csv')},
            'reload_evidence':{'validation_origins':len(cv_replay),'validation_max_absdiff_kwh':max(cv_replay),'test_origins':len(reload_errors),
                               'test_max_absdiff_kwh':max(reload_errors),'final_max_absdiff_kwh':final_error,
                               'future_poison_max_absdiff_kwh':poison_error,'tensor_checks':checks},
            'runtime_seconds':time.perf_counter()-started,'may_labels_read':False,'data_sha256':data_sha,'vendor':vendor,
            'limitations':['Post-EDA chronological revision, not pristine prospective holdout','Budgeted compact local search, no guaranteed improvement or global optimum',
                           '13-module provisional energy sum may double count meter hierarchy','May forecast uses direct path and no unavailable future sensor context',
                           'Empty stable sensor selection is allowed; identical input/architecture fits are reused and documented',
                           'More input variables do not guarantee effective cross-channel skill; PatchTST input roles differ from iTransformer']}
    dump(dest/'results.json',result)
    # Family/scenario completion files are independent; aggregate built by --summarize.
    print(f'COMPLETE {family} {scenario}: CV {result["validation_metrics"]["mae_kwh"]:.6f}, TEST {result["test_metrics"]["mae_kwh"]:.6f}, sensors={sch["selected_sensor_count"]}',flush=True)
    return result

def predict_saved(checkpoint,history_csv=None):
    model,saved=reload(Path(checkpoint));names=saved['schema']['input_features']
    if history_csv:
        hist=pd.read_csv(history_csv,index_col='hour',parse_dates=['hour'],float_precision='round_trip')
        if any(c not in hist for c in names):raise ValueError('New history must contain every saved named input feature')
        if not hist.index.is_unique or not hist.index.is_monotonic_increasing or not hist.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all():raise ValueError('History must be unique consecutive completed-hour rows')
        if not np.isfinite(hist[names].to_numpy(float)).all():raise ValueError('New history contains missing/nonfinite input features')
    else:hist=pd.DataFrame(saved['forecast_context'],columns=names,index=pd.to_datetime(saved['forecast_context_time']))
    pred,_=infer(saved['family'],model,hist,saved['architecture'],saved['normalizer'],saved['schema'])
    origin=hist.index[-1]+pd.Timedelta(hours=1);idx=pd.date_range(origin,periods=len(pred),freq='h')
    return pd.DataFrame({'id':idx,'origin':str(origin),'horizon_offset_hours':np.arange(len(pred)),'lead_hours':np.arange(1,len(pred)+1),'predicted_kwh':pred})

def aggregate():
    result={'status':'complete','scenarios':{},'may_labels_read':False,
            'scope':'New neural models only, official pinned core; old outputs read-only',
            'cli':'neural_revision.py --predict --checkpoint <saved_model.pt> --output <new.csv>; optional --history consecutive wide CSV with hour and every named saved input feature'}
    for scenario in ['dayahead24','long745']:
        rows={}
        for family in ['itransformer','patchtst']:
            path=OUT/scenario/family/'results.json'
            if path.exists():rows[family]=readjson(path)
        if rows:result['scenarios'][scenario]={'models':rows}
    result['completed_models']=sum(len(s['models']) for s in result['scenarios'].values())
    result['status']='complete' if result['completed_models']==4 else 'partial'
    dump(OUT/'results.json',result);print(f'SUMMARY {result["completed_models"]} models',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--family',choices=['itransformer','patchtst']);p.add_argument('--scenario',choices=['dayahead24','long745'],default='dayahead24')
    p.add_argument('--resume',action='store_true');p.add_argument('--summarize',action='store_true');p.add_argument('--predict',action='store_true')
    p.add_argument('--checkpoint');p.add_argument('--history');p.add_argument('--output');args=p.parse_args()
    if args.summarize:aggregate();return
    if args.predict:
        if not args.checkpoint or not args.output:p.error('--predict needs --checkpoint and --output')
        target=Path(args.output).resolve()
        if target.exists():p.error('Inference output must be a new file')
        data=predict_saved(args.checkpoint,args.history);target.parent.mkdir(parents=True,exist_ok=True);data.to_csv(target,index=False,float_format='%.17g')
        print(f'PREDICTED {len(data)} rows',flush=True);return
    if not args.family:p.error('--family required for training')
    run(args.family,args.scenario,args.resume)

if __name__=='__main__':main()
