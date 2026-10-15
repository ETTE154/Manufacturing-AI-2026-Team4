"""Past-only H/D official-core forecasts, CV-locked before test/future export.

No May labels are opened. Daily models forecast completed-day hourly-average
energy rates, not period totals: 32 rates repeat 24 times then crop to 745h.
Old official cores/checkpoint files are hash-verified and read only.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from types import SimpleNamespace
import argparse, hashlib, json, os, random, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

OUT = Path(__file__).resolve().parent
EXP = OUT.parent
ROOT = EXP.parents[1]
OLD = ROOT/'output/modeling/forecast'
sys.path.insert(0,str(EXP));sys.path.insert(0,str(OLD));sys.path.insert(0,str(OLD/'patchtst'))
from resample_common import load_hourly_target, periods as common_periods, VALID_ORIGINS
from patchtst_vendor.models.PatchTST import Model as PatchCore
from itransformer.vendor.model.iTransformer import Model as InvertedCore

SEED=20261013;THREADS=4;MAX_EPOCHS=24;PATIENCE=4
VALID=pd.to_datetime(['2025-02-15','2025-02-22','2025-02-25'])
assert VALID.equals(VALID_ORIGINS)
TEST=pd.Timestamp('2025-03-29');FINAL=pd.Timestamp('2025-04-30')
POINTS=745;LATE=240
H_CONFIGS=[
 {'group':'target_only','context':24,'loss':'MSE','dropout':.1,'instance_norm':False,'output_penalty':.03},
 {'group':'target_only','context':72,'loss':'L1','dropout':.2,'instance_norm':False,'output_penalty':.05},
]
D_CONFIGS=[
 {'group':'target_only','context':7,'loss':'MSE','dropout':.1,'instance_norm':False,'output_penalty':.03},
 {'group':'target_only','context':7,'loss':'L1','dropout':.2,'instance_norm':False,'output_penalty':.05},
]
MULTI={'group':'completed_multires','context':24,'loss':'MSE','dropout':.1,'instance_norm':False,'output_penalty':.03}

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(Path,pd.Timestamp)):return str(x)
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');temp.replace(path)
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def seed():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    torch.set_num_threads(THREADS);torch.use_deterministic_algorithms(True)
def prefix(y,origin):
    h=y.loc[y.index<pd.Timestamp(origin)]
    assert h.index[-1]+pd.Timedelta(hours=1)==pd.Timestamp(origin)
    return h

def complete_periods(h,freq):
    # The input is a freshly sliced prefix; attrs from an earlier origin cannot
    # relax the completion clock of this particular fit or input sequence.
    h=h.copy();h.attrs['origin']=str(h.index[-1]+pd.Timedelta(hours=1))
    p=common_periods(h,freq,complete_only=True)
    return pd.DataFrame({'rate':p.rate_kwh_per_hour,'available_at':p.available_at,
                         'expected_hours':p.expected_hours},index=p.index)

def native_frame(h,resolution,group):
    if resolution=='daily':
        d=complete_periods(h,'D');assert len(d)>0
        return pd.DataFrame({'target_rate':d.rate.to_numpy()},index=d.index)
    result=pd.DataFrame(index=h.index)
    if group=='completed_multires':
        # Row t is available at t+1h, including energy for that completed hour.
        # A coarse bucket becomes a covariate only at its completion clock.
        clock=h.index+pd.Timedelta(hours=1)
        fallback=h.expanding().mean().to_numpy(float)
        for name,freq in [('daily','D'),('weekly','W-SUN'),('monthly','MS')]:
            p=complete_periods(h,freq)
            available=pd.DatetimeIndex(p.available_at)
            ix=available.searchsorted(clock,side='right')-1
            valid=ix>=0
            rate=fallback.copy();rate[valid]=p.rate.to_numpy()[ix[valid]]
            result[f'completed_{name}_rate']=rate
            result[f'completed_{name}_available']=valid.astype(float)
            if valid.any():assert (available[ix[valid]]<=clock[valid]).all()
    result['target_rate']=h.to_numpy(float)
    return result

def metrics(a,p):
    a=np.asarray(a,float);p=np.asarray(p,float);assert a.shape==p.shape and np.isfinite(a).all() and np.isfinite(p).all()
    e=p-a;den=((a-a.mean())**2).sum()
    return {'count':len(a),'mae_kwh':float(np.abs(e).mean()),'rmse_kwh':float(np.sqrt(np.mean(e**2))),
            'mape_percent':float(np.mean(np.abs(e)/np.abs(a))*100),'r2':float(1-(e**2).sum()/den) if den>0 else None,
            'bias_pred_minus_actual_kwh':float(e.mean())}
def summary(f):
    late=f[f.horizon_offset_hours>=LATE]
    return {'full745':metrics(f.actual_kwh,f.predicted_kwh),'late505':metrics(late.actual_kwh,late.predicted_kwh),
            'origin_count':int(f.origin.nunique()),'unique_target_hours':int(f.id.nunique()),
            'late_unique_target_hours':int(late.id.nunique()),
            'origin_late505_mae_kwh':{str(k):float(v) for k,v in late.groupby('origin').absolute_error_kwh.mean().items()},
            'mean_origin_late505_mae_kwh':float(late.groupby('origin').absolute_error_kwh.mean().mean())}
def hourly_map(native,resolution):
    p=np.asarray(native,float)
    return p[:POINTS] if resolution=='hourly' else np.repeat(p,24)[:POINTS]
def pred_frame(family,resolution,split,fold,origin,y,native):
    idx=pd.date_range(origin,periods=POINTS,freq='h');p=hourly_map(native,resolution)
    a=y.reindex(idx).to_numpy(float)
    return pd.DataFrame({'id':idx,'model':f'{resolution}_{family}','split':split,'fold':fold,'origin':origin,
                         'horizon_offset_hours':np.arange(POINTS),'lead_hours':np.arange(1,POINTS+1),
                         'actual_kwh':a,'predicted_kwh':p,'absolute_error_kwh':np.abs(p-a)})

class Windows(Dataset):
    def __init__(self,z,context,points,resolution):
        self.values=torch.from_numpy(np.asarray(z,np.float32));self.context=context;self.points=points
        last=len(z)-points;step=24 if resolution=='hourly' else 1
        self.origins=np.arange(context,last+1,step,dtype=int)
        if last>=context and (not len(self.origins) or self.origins[-1]!=last):self.origins=np.r_[self.origins,last]
        if not len(self.origins):raise ValueError(f'Not enough training periods: {len(z)} < {context}+{points}')
        assert self.origins[-1]+points==len(z)
    def __len__(self):return len(self.origins)
    def __getitem__(self,index):
        o=self.origins[index];return self.values[o-self.context:o],self.values[o:o+self.points]

def architecture(family,cfg,resolution,n):
    p=POINTS if resolution=='hourly' else 32
    common=dict(seq_len=cfg['context'],pred_len=p,d_model=32,d_ff=64,e_layers=1,n_heads=4,dropout=cfg['dropout'],enc_in=n)
    if family=='patchtst':
        return {**common,'fc_dropout':cfg['dropout'],'head_dropout':cfg['dropout'],'individual':False,
                'patch_len':16 if resolution=='hourly' else 4,'stride':8 if resolution=='hourly' else 2,
                'padding_patch':'end','revin':cfg['instance_norm'],'affine':True,'subtract_last':False,
                'decomposition':False,'kernel_size':25,'core_norm':'LayerNorm'}
    return {**common,'output_attention':False,'use_norm':cfg['instance_norm'],'embed':'timeF',
            'freq':'h' if resolution=='hourly' else 'd','class_strategy':'projection','factor':5,'activation':'gelu'}
def build(family,arch):
    return PatchCore(SimpleNamespace(**arch),norm=arch['core_norm']).cpu() if family=='patchtst' else InvertedCore(SimpleNamespace(**arch)).cpu()
def forward(family,model,x):return model(x) if family=='patchtst' else model(x,None,None,None)

def setup(family,data,cfg,resolution):
    seed();values=data.to_numpy(float);mean=values.mean(0);s=values.std(0);scale=np.where(s>1e-8,s,1.)
    ds=Windows((values-mean)/scale,cfg['context'],POINTS if resolution=='hourly' else 32,resolution)
    loader=DataLoader(ds,batch_size=32 if resolution=='hourly' else 8,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(SEED))
    arch=architecture(family,cfg,resolution,len(mean));model=build(family,arch)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    norm={'features':list(data.columns),'target_index':list(data.columns).index('target_rate'),
          'mean':mean.tolist(),'scale':scale.tolist(),'fit_rows':len(data),'fit_start':str(data.index[0]),
          'fit_end':str(data.index[-1]),'fit_scope':'Strict training prefix only','near_constant_fallback_count':int((s<=1e-8).sum())}
    return model,optimizer,loader,arch,norm,ds
def epoch(family,model,opt,loader,cfg,target_index):
    model.train();total=0.;objtotal=0.;n=0
    for x,y in loader:
        opt.zero_grad(set_to_none=True);pred=forward(family,model,x)
        assert pred.shape==y.shape
        delta=pred[:,:,target_index]-y[:,:,target_index]
        loss=delta.abs().mean() if cfg['loss']=='L1' else delta.pow(2).mean()
        objective=loss+cfg['output_penalty']*pred[:,:,target_index].pow(2).mean()
        assert torch.isfinite(objective)
        objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        total+=float(loss.detach())*len(x);objtotal+=float(objective.detach())*len(x);n+=len(x)
    return {'loss':cfg['loss'],'target_data_loss_standardized':total/n,'regularized_objective':objtotal/n}
def infer(family,model,data,arch,norm):
    assert list(data.columns)==norm['features'];assert arch['enc_in']==len(norm['features'])
    assert norm['features'][norm['target_index']]=='target_rate'
    x=((data.iloc[-arch['seq_len']:].to_numpy(float)-np.asarray(norm['mean']))/np.asarray(norm['scale'])).astype(np.float32)
    model.eval()
    with torch.no_grad():p=forward(family,model,torch.from_numpy(x).unsqueeze(0))
    assert tuple(p.shape)==(1,arch['pred_len'],len(norm['features']))
    idx=norm['target_index'];return p[0,:,idx].numpy().astype(np.float64)*norm['scale'][idx]+norm['mean'][idx]

def choose_epochs(family,h,cfg,resolution):
    native=native_frame(h,resolution,cfg['group']);hold=POINTS if resolution=='hourly' else 32
    fit=native.iloc[:-hold];tail=native.iloc[-hold:]
    assert len(fit)>=cfg['context']+(POINTS if resolution=='hourly' else 32)
    model,opt,loader,arch,norm,ds=setup(family,fit,cfg,resolution)
    inner_origin=tail.index[0]
    # Native D holdout is 32 completed days; target evaluation is its first745h.
    actual=h.reindex(pd.date_range(inner_origin,periods=POINTS,freq='h')).to_numpy(float)
    assert np.isfinite(actual).all()
    best=np.inf;best_epoch=1;stale=0;logs=[];started=time.perf_counter()
    for e in range(1,MAX_EPOCHS+1):
        train=epoch(family,model,opt,loader,cfg,norm['target_index'])
        p=hourly_map(infer(family,model,fit,arch,norm),resolution)
        m=metrics(actual[LATE:],p[LATE:])['mae_kwh']
        logs.append({'epoch':e,**train,'inner_late505_mae_kwh':m})
        if m<best-1e-7:best=m;best_epoch=e;stale=0
        else:stale+=1
        if stale>=PATIENCE:break
    return best_epoch,{'fit_end':str(fit.index[-1]),'fit_rows':len(fit),'fit_history_hour_end':str(inner_origin-pd.Timedelta(hours=1)),
          'holdout_native_periods':hold,'inner_forecast_origin':str(inner_origin),'inner_scoring_offsets':'240..744',
          'best_epoch':best_epoch,'best_inner_late505_mae_kwh':best,'epochs_run':len(logs),
          'best_at_epoch_cap':best_epoch==MAX_EPOCHS,'training_windows':len(ds),'logs':logs,'normalizer':norm,
          'future_label_overlap':False,'scaler_excludes_inner_holdout':True,'seconds':time.perf_counter()-started}

def fit_save(family,h,cfg,resolution,epochs,path,source_hash):
    data=native_frame(h,resolution,cfg['group']);started=time.perf_counter()
    model,opt,loader,arch,norm,ds=setup(family,data,cfg,resolution);logs=[]
    for e in range(1,epochs+1):logs.append({'epoch':e,**epoch(family,model,opt,loader,cfg,norm['target_index'])})
    payload={'family':family,'resolution':resolution,'config':cfg,'state_dict':model.state_dict(),
             'architecture':arch,'normalizer':norm,'epochs':int(epochs),'seed':SEED,'data_sha256':source_hash,
             'history_start':str(h.index[0]),'history_end':str(h.index[-1]),'history_hour_count':len(h),
             'native_period_count':len(data),'forecast_context':data.iloc[-arch['seq_len']:].to_numpy(float).tolist(),
             'forecast_context_time':[str(t) for t in data.index[-arch['seq_len']:]],
             'native_forecast_points':arch['pred_len'],'hourly_forecast_points':POINTS,
             'mapping':'native H target path' if resolution=='hourly' else '32 completed-day mean rates uniformly repeat24 and crop745; final day1h',
             'mapping_profile':{'kind':'uniform','hour_factors':[1.]*24,'fit_uses_future':False},
             'target_loss':'Explicit target_rate channel only; multires covariates never future targets',
             'units':'Provisional guide kWh per hour; meter hierarchy unresolved'}
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,path)
    native=infer(family,model,data,arch,norm)
    record={k:v for k,v in payload.items() if k not in ['state_dict','forecast_context','forecast_context_time']}
    record.update({'path':str(path),'sha256':sha(path),'training_windows':len(ds),'training_loss_logs':logs,
                   'fit_seconds':time.perf_counter()-started,'trainable_parameters':sum(p.numel() for p in model.parameters()),
                   'normalizer_fit_matches_prefix':norm['fit_end']==str(data.index[-1]),
                   'maximum_supervised_native_label':str(data.index[-1]),'model_input_features':list(data.columns)})
    return native,record

def reload_predict(path,history=None):
    seed();p=torch.load(path,map_location='cpu',weights_only=True)
    model=build(p['family'],p['architecture']);model.load_state_dict(p['state_dict'],strict=True)
    if history is None:
        data=pd.DataFrame(p['forecast_context'],columns=p['normalizer']['features'],index=pd.to_datetime(p['forecast_context_time']))
        origin=pd.Timestamp(p['history_end'])+pd.Timedelta(hours=1)
    else:
        f=pd.read_csv(history,float_precision='round_trip')
        if not {'timestamp','energy_kwh'}.issubset(f.columns):raise ValueError('Custom history needs timestamp,energy_kwh columns')
        h=pd.Series(f.energy_kwh.to_numpy(float),index=pd.to_datetime(f.timestamp),name='energy_kwh')
        assert h.index.equals(pd.date_range(h.index[0],periods=len(h),freq='h')) and np.isfinite(h).all()
        origin=h.index[-1]+pd.Timedelta(hours=1)
        if p['resolution']=='daily' and origin.hour!=0:raise ValueError('Daily model origin must be midnight after a completed day')
        data=native_frame(h,p['resolution'],p['config']['group'])
        if len(data)<p['architecture']['seq_len']:raise ValueError('History shorter than saved native context')
    native=infer(p['family'],model,data,p['architecture'],p['normalizer'])
    return native,origin,p

def core_provenance(family):
    path=OLD/('patchtst/provenance.json' if family=='patchtst' else 'itransformer/vendor/provenance.json');p=read(path)
    for row in p['files']:
        file=Path(row['vendored_path']) if family=='patchtst' else OLD/'itransformer/vendor'/row['path']
        expected=row['vendored_sha256'] if family=='patchtst' else row['local_sha256'];assert sha(file)==expected
    return {'file':str(path),'sha256':sha(path),'official_core_hashes_verified':True,
            'claim':'Pinned official 2023 PatchTST / 2024 iTransformer core, compact retraining; no claim of current SOTA superiority'}

def native_csv(path,native,origin,resolution):
    idx=pd.date_range(origin,periods=len(native),freq='h' if resolution=='hourly' else 'D')
    pd.DataFrame({'timestamp':idx,'forecast_rate_kwh_per_hour':native,
                  'forecast_full_period_energy_kwh':native*(1 if resolution=='hourly' else 24),
                  'expected_period_hours':1 if resolution=='hourly' else 24}).to_csv(path,index=False,encoding='utf-8-sig')

def run_family(family,resolution,y):
    dest=OUT/resolution/family;dest.mkdir(parents=True,exist_ok=True)
    source=ROOT/'output/analysis/baseline_audit/complete_hour_module_sum.csv';source_hash=sha(source)
    configs=list(H_CONFIGS if resolution=='hourly' else D_CONFIGS)
    if family=='itransformer' and resolution=='hourly':configs.append(MULTI)
    write(dest/'search_plan.json',{'configs':configs,'valid_origins':list(VALID),'objective':'Mean three origin late505 MAE only',
         'daily_context14_excluded':'First origin76days minus32day inner holdout=44days < context14+P32=46',
         'epoch_policy':f'Nested prefix-only early stopping max{MAX_EPOCHS},patience{PATIENCE}; final epoch median of selected CV epochs, no April tuning',
         'coarse_partial_policy':'Complete training days only. Native32days to hourly745 includes last forecast day first hour only.'})
    records=[];selected_frames={};started=time.perf_counter()
    for ci,cfg in enumerate(configs):
        frames=[];fold_records=[]
        for fold,origin in enumerate(VALID,1):
            h=prefix(y,origin);epochs,audit=choose_epochs(family,h,cfg,resolution)
            native,model=fit_save(family,h,cfg,resolution,epochs,dest/'cv_models'/f'candidate{ci}_fold{fold}.pt',source_hash)
            f=pred_frame(family,resolution,'validation',fold,origin,y,native);frames.append(f)
            fold_records.append({'fold':fold,'origin':str(origin),'inner_early_stopping':audit,'fitted_model':model})
            print(f'{resolution}/{family} candidate{ci} fold{fold} bestepoch{epochs} late505={summary(f)["mean_origin_late505_mae_kwh"]:.6f}',flush=True)
            write(OUT/'training_progress.json',{'family':family,'resolution':resolution,'candidate':ci,'fold':fold,'status':'CV in progress','seconds':time.perf_counter()-started})
        f=pd.concat(frames,ignore_index=True);csv=dest/f'candidate{ci}_validation_predictions.csv';f.to_csv(csv,index=False,encoding='utf-8-sig')
        stats=summary(f);records.append({'candidate':ci,'config':cfg,'selection_objective_mae_kwh':stats['mean_origin_late505_mae_kwh'],
                                         'validation':stats,'folds':fold_records,'prediction_csv':str(csv)})
        selected_frames[ci]=f
        write(dest/'candidate_results.json',{'candidate_results':records,'completed_candidates':len(records)})
    chosen=min(records,key=lambda x:x['selection_objective_mae_kwh']);cfg=chosen['config']
    final_epochs=int(np.median([r['inner_early_stopping']['best_epoch'] for r in chosen['folds']]))
    lock={'model':family,'scenario':resolution,'selected_candidate':chosen['candidate'],'selected_config':cfg,
          'selection_objective':'Mean of three validation-origin MAEs on offsets240..744 only','validation_origins':list(VALID),
          'latest_validation_label':str(max(VALID)+pd.Timedelta(hours=POINTS-1)),'internal_test_origin':str(TEST),
          'selection_has_test_or_May_access':False,'final_refit_epochs':final_epochs,
          'final_epoch_policy':'Median selected-candidate inner early-stopping epochs across three CV origins; April labels never tune epoch',
          'candidate_results':records,'validation_targets_overlap_across_origins':True,
          'daily_mapping_locked':'Uniform repeat24 crop745','locked_before_internal_test':True}
    assert pd.Timestamp(lock['latest_validation_label'])<TEST
    write(dest/'selection_lock.json',lock)
    selected_frames[chosen['candidate']].to_csv(dest/'validation_predictions.csv',index=False,encoding='utf-8-sig')
    replay=[];fits={};stats={}
    for split,origin in [('internal_test',TEST),('future_unlabelled',FINAL)]:
        h=prefix(y,origin)
        if split=='internal_test':epochs,inner=choose_epochs(family,h,cfg,resolution)
        else:epochs,inner=final_epochs,{'epoch_count_frozen_on_validation':True,'no_April_epoch_selection':True}
        path=dest/'saved_models'/f'{split}.pt';native,model=fit_save(family,h,cfg,resolution,epochs,path,source_hash)
        f=pred_frame(family,resolution,split,0,origin,y,native)
        csv=dest/('test_predictions.csv' if split=='internal_test' else 'future_path.csv');f.to_csv(csv,index=False,encoding='utf-8-sig')
        native_csv(dest/f'{split}_native_predictions.csv',native,origin,resolution)
        again,clock,p=reload_predict(path);err=float(np.abs(again-native).max());assert err==0 and clock==origin
        # Raw input poison changes all unavailable data, and cannot change prefix/covariates.
        poisoned=y.copy();poisoned.loc[poisoned.index>=origin]=1e12
        a=native_frame(prefix(y,origin),resolution,cfg['group']);b=native_frame(prefix(poisoned,origin),resolution,cfg['group'])
        assert a.equals(b)
        mean=a.to_numpy().mean(0);sd=a.to_numpy().std(0);sd=np.where(sd>1e-8,sd,1.)
        scaler_error=float(max(np.abs(mean-np.asarray(model['normalizer']['mean'])).max(),np.abs(sd-np.asarray(model['normalizer']['scale'])).max()))
        assert scaler_error==0
        replay.append({'split':split,'checkpoint':str(path),'prediction_csv':str(csv),'max_abs_native_replay_error':err,
                       'hourly_mapping_exact':bool(np.array_equal(hourly_map(again,resolution),f.predicted_kwh.to_numpy())),
                       'future_poison_prefix_covariates_equal':True,'prefix_scaler_recompute_max_error':scaler_error,
                       'forecast_start':str(clock),'forecast_end':str(clock+pd.Timedelta(hours=744))})
        fits[split]={'inner_early_stopping':inner,'fitted_model':model,'prediction_csv':str(csv),'native_prediction_csv':str(dest/f'{split}_native_predictions.csv')}
        if split=='internal_test':stats[split]=summary(f)
    result={'family':family,'scenario':resolution,'selected_config':cfg,'selected_candidate':chosen['candidate'],
            'validation':chosen['validation'],'internal_test':stats['internal_test'],'fits':fits,'replay_verification':replay,
            'final_epoch_count':final_epochs,'selection_lock':str(dest/'selection_lock.json'),
            'validation_prediction_csv':str(dest/'validation_predictions.csv'),'candidate_results_json':str(dest/'candidate_results.json'),
            'input_scope':'Aggregate hourly target only' if cfg['group']=='target_only' and resolution=='hourly' else
                          'Completed-day hourly-average aggregate energy rate' if resolution=='daily' else
                          'Aggregate target + only already completed D/W/MS rates and availability flags; cross-variate attention',
            'target_mode':'Explicit target_rate output channel; target-only loss. No auxiliary future-covariate labels.',
            'daily_mapping':'32 daily mean-rate predictions uniformly repeated24, truncated to745h (last day1h)' if resolution=='daily' else None,
            'patch_channel_independence_note':'Official channel-independent core; no cross-resolution covariate benefit is claimed' if family=='patchtst' else None,
            'single_token_limit':'Pure hourly/daily univariate iTransformer has one variate token; multires hourly candidate is separately screened' if family=='itransformer' else None,
            'training_budget':{'maximum_inner_epochs':MAX_EPOCHS,'patience':PATIENCE,'cpu_threads':THREADS},
            'daily_tiny_sample_limit':'First inner fit44days, context7 + prediction32 leaves6 overlapping windows, not6 independent series' if resolution=='daily' else None,
            'may_labels_read':False,'official_core_provenance':core_provenance(family),'runtime_seconds':time.perf_counter()-started}
    write(dest/'results.json',result)
    print(json.dumps({'done':f'{resolution}/{family}','chosen':cfg,'cv505':chosen['selection_objective_mae_kwh'],
                      'test505':stats['internal_test']['late505']['mae_kwh'],'seconds':result['runtime_seconds']},ensure_ascii=False),flush=True)
    return result

def update_summary():
    scenarios={}
    for resolution in ['hourly','daily']:
        models={}
        for family in ['patchtst','itransformer','chronos2']:
            path=OUT/resolution/family/'results.json'
            if path.exists():models[family]=read(path)
        scenarios[resolution]={'models':models}
    write(OUT/'results.json',{'status':'complete' if all(len(x['models'])>=2 for x in scenarios.values()) else 'in_progress',
         'scenarios':scenarios,'selection_objective':'Mean validation-origin Late505 MAE only','validation_origins':list(VALID),
         'internal_test_origin':str(TEST),'final_origin':str(FINAL),'future_path_points':POINTS,'may_labels_read':False,
         'weekly_monthly_neural_training':'Not attempted: too few complete coarse periods, transparent coarse baselines are evaluated elsewhere',
         'validation_target_overlap':'Three paths overlap; score rows are not independent hours. Unique target counts provided.',
         'post_EDA_provenance':'April/May were inspected in earlier work; this is revised historical evaluation, not a pristine prospective blind test',
         'units':'Provisional guide kWh; 13-module meter hierarchy unresolved','script':str(Path(__file__).resolve()),
         'replay_command':f'"{ROOT}/output/modeling/.venv/Scripts/python.exe" "{Path(__file__).resolve()}" --predict --checkpoint <saved_model.pt> --output <path.csv>',
         'custom_history_contract':'Optional --history CSV with timestamp,energy_kwh contiguous hourly observations; saved scaler/weights stay frozen; D origin midnight required.'})

def main():
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['patchtst','itransformer']);p.add_argument('--resolution',choices=['hourly','daily'])
    p.add_argument('--predict',action='store_true');p.add_argument('--checkpoint');p.add_argument('--history');p.add_argument('--output')
    p.add_argument('--summary-only',action='store_true');args=p.parse_args()
    if args.summary_only:update_summary();return
    if args.predict:
        if not args.checkpoint or not args.output:p.error('--checkpoint and --output required')
        native,origin,meta=reload_predict(args.checkpoint,args.history)
        idx=pd.date_range(origin,periods=POINTS,freq='h');target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame({'timestamp':idx,'predicted_energy_kwh':hourly_map(native,meta['resolution'])}).to_csv(target,index=False,encoding='utf-8-sig')
        print(f'Saved frozen-checkpoint forecast: {target}');return
    if not args.family or not args.resolution:p.error('--family and --resolution required')
    y=load_hourly_target();assert len(y)==3600
    assert y.index.equals(pd.date_range('2024-12-01',periods=3600,freq='h'))
    run_family(args.family,args.resolution,y);update_summary()
if __name__=='__main__':main()
