"""Official H/D cores, reversible residual normalization, plateau-driven fitting."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace
import time,copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from common import *
OLD=ROOT/'output/modeling/forecast';sys.path.insert(0,str(OLD));sys.path.insert(0,str(OLD/'patchtst'))
from patchtst_vendor.models.PatchTST import Model as PatchCore
from itransformer.vendor.model.iTransformer import Model as InvertedCore

def configs(family):
    cs=[{'resolution':'hourly','context':c,'scaler':s,'loss':loss,'instance_norm':inside,'group':'target'} for c,s,loss,inside in
        [(24,'zscore','MAE',False),(72,'robust','Huber',True),(168,'minmax','MSE',False),(72,'zscore','Huber',False)]]
    cs +=[{'resolution':'daily','context':7,'scaler':s,'loss':loss,'instance_norm':False,'group':'target'} for s,loss in
        [('zscore','MSE'),('zscore','MAE'),('robust','Huber'),('minmax','MSE')]]
    if family=='itransformer':cs.append({'resolution':'hourly','context':72,'scaler':'robust','loss':'Huber','instance_norm':True,'group':'module13_target'})
    return [{**c,'d_model':64,'layers':2,'dropout':.1,'mean_prior_penalty':.01} for c in cs]

MODULES=None
def data_frame(h,cfg):
    global MODULES
    s=native(h,cfg['resolution']);d=pd.DataFrame({'target':s.to_numpy(float)},index=s.index)
    if cfg['group']=='module13_target':
        if MODULES is None:
            f=pd.read_csv(MODULE_SOURCE,float_precision='round_trip');f=f[f.complete_720].copy();f['hour']=pd.to_datetime(f.hour)
            MODULES=f.pivot(index='hour',columns='module(equipment)',values='energy_kwh');MODULES.index.name=None
        d=MODULES.reindex(h.index).copy();assert np.isfinite(d.to_numpy()).all();d['target']=h.to_numpy(float)
        assert np.allclose(d.iloc[:,:13].sum(axis=1),d.target,rtol=0,atol=1e-7)
    return d
class Windows(Dataset):
    def __init__(self,z,context,points,res):
        self.z=torch.from_numpy(np.asarray(z,np.float32));self.context=context;self.points=points
        last=len(z)-points;step=24 if res=='hourly' else 1;self.o=np.arange(context,last+1,step)
        if last>=context and (not len(self.o) or self.o[-1]!=last):self.o=np.r_[self.o,last]
        assert len(self.o)>0
    def __len__(self):return len(self.o)
    def __getitem__(self,i):o=int(self.o[i]);return self.z[o-self.context:o],self.z[o:o+self.points]
def arch(family,cfg,n):
    c={'seq_len':cfg['context'],'pred_len':745 if cfg['resolution']=='hourly' else 32,'enc_in':n,'d_model':cfg['d_model'],
       'd_ff':128,'e_layers':cfg['layers'],'n_heads':4,'dropout':cfg['dropout']}
    if family=='patchtst':return {**c,'fc_dropout':.1,'head_dropout':.1,'individual':False,'patch_len':16 if cfg['resolution']=='hourly' else 4,
          'stride':8 if cfg['resolution']=='hourly' else 2,'padding_patch':'end','revin':cfg['instance_norm'],'affine':True,
          'subtract_last':False,'decomposition':False,'kernel_size':25}
    return {**c,'output_attention':False,'use_norm':cfg['instance_norm'],'embed':'timeF','freq':'h','class_strategy':'projection','factor':5,'activation':'gelu'}
def build(family,a):return PatchCore(SimpleNamespace(**a),norm='LayerNorm').cpu() if family=='patchtst' else InvertedCore(SimpleNamespace(**a)).cpu()
def forward(family,m,x):return m(x) if family=='patchtst' else m(x,None,None,None)
def setup(family,d,cfg):
    seed();n=norm_fit(d.values,cfg['scaler']);n.update({'fit_end':str(d.index[-1]),'features':list(d.columns),'target_index':d.columns.get_loc('target')})
    a=arch(family,cfg,len(d.columns));m=build(family,a)
    if not cfg['instance_norm']:
        head=m.model.head.linear if family=='patchtst' else m.projector
        with torch.no_grad():head.weight.zero_();head.bias.fill_(float(zero_prior(n)[n['target_index']]))
    ds=Windows(transform(d.values,n),cfg['context'],a['pred_len'],cfg['resolution'])
    dl=DataLoader(ds,batch_size=32 if cfg['resolution']=='hourly' else 8,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(SEED))
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001)
    return m,opt,dl,a,n,len(ds)
def train_epoch(family,m,opt,dl,cfg,n):
    m.train();tot=0.;count=0;idx=n['target_index'];prior=float(zero_prior(n)[idx])
    for x,y in dl:
        opt.zero_grad(set_to_none=True);p=forward(family,m,x);assert p.shape==y.shape
        delta=p[:,:,idx]-y[:,:,idx]
        loss=delta.abs().mean() if cfg['loss']=='MAE' else torch.nn.functional.smooth_l1_loss(p[:,:,idx],y[:,:,idx]) if cfg['loss']=='Huber' else delta.pow(2).mean()
        obj=loss+cfg['mean_prior_penalty']*(p[:,:,idx]-prior).pow(2).mean();obj.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();tot+=float(loss.detach())*len(x);count+=len(x)
    return tot/count
def infer(family,m,d,a,n):
    assert list(d.columns)==n['features'] and n['features'][n['target_index']]=='target'
    z=transform(d.iloc[-a['seq_len']:].values,n).astype(np.float32);m.eval()
    with torch.no_grad():out=forward(family,m,torch.from_numpy(z).unsqueeze(0))
    assert tuple(out.shape)==(1,a['pred_len'],a['enc_in'])
    return inverse(out[0].numpy().astype(float),n)[:,n['target_index']]
def choose_epochs(family,h,cfg,best_path=None):
    d=data_frame(h,cfg);hold=745 if cfg['resolution']=='hourly' else 32;fit=d.iloc[:-hold];tail=d.iloc[-hold:]
    m,opt,dl,a,n,windows=setup(family,fit,cfg);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=8,
        threshold=1e-5,threshold_mode='rel',min_lr=1e-6)
    origin=tail.index[0];actual=h.reindex(pd.date_range(origin,periods=745,freq='h')).to_numpy(float)
    assert np.isfinite(actual).all();best=np.inf;best_epoch=0;stale=0;logs=[];epoch=0
    best_state=None
    while True:
        epoch+=1;loss=train_epoch(family,m,opt,dl,cfg,n);p=hourly_map(infer(family,m,fit,a,n),cfg['resolution'])
        mae=metrics(actual[240:],p[240:])['mae_kwh'];scheduler.step(mae);lr=float(opt.param_groups[0]['lr'])
        logs.append({'epoch':epoch,'train_scaled_target_loss':loss,'inner_Late505_mae_kwh':mae,'learning_rate':lr})
        improved=(mae<best-max(1e-6,best*1e-5)) if np.isfinite(best) else True
        if improved:best=mae;best_epoch=epoch;stale=0;best_state=copy.deepcopy(m.state_dict())
        else:stale+=1
        # No arbitrary24-epoch stop. Stop only after LR floor and sustained
        # native-unit validation plateau, keeping the measured best epoch.
        if stale>=30 and lr<=1.00001e-6:break
        if epoch%100==0:print(f'{family}/{cfg["resolution"]} epoch{epoch} best{best:.6f} lr{lr:.1g}',flush=True)
    if best_path is not None:
        best_path.parent.mkdir(parents=True,exist_ok=True)
        payload={'family':family,'config':cfg,'architecture':a,'normalizer':n,'epochs':best_epoch,'state_dict':best_state,
             'history_end':str(origin-pd.Timedelta(hours=1)),'features':list(fit.columns),'stored_context':fit.iloc[-a['seq_len']:].values.tolist(),
             'stored_context_times':[str(t) for t in fit.index[-a['seq_len']:]],'seed':SEED,'unit':'provisional guide kWh'}
        tmp=best_path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(best_path)
        replayed=hourly_map(replay(best_path),cfg['resolution']);actual_best=metrics(actual[240:],replayed[240:])['mae_kwh']
        assert abs(actual_best-best)<1e-7
    schedule=[.001]+[v['learning_rate'] for v in logs[:best_epoch-1]]
    return best_epoch,{'fit_end':str(fit.index[-1]),'fit_rows':len(fit),'holdout_native_periods':hold,'normalizer':n,
       'training_windows':windows,'best_epoch':best_epoch,'epochs_run':epoch,'best_inner505_mae':best,'logs':logs,
       'stop_reason':'Learning-rate floor1e-6 and >=30 epochs without meaningful validation improvement',
       'fixed_epoch_cap':None,'scaler_excludes_holdout':True,'future_label_overlap':False,'best_checkpoint':str(best_path) if best_path else None,
       'refit_learning_rate_schedule':schedule}
def fit_saved(family,h,cfg,epochs,path,lr_schedule=None):
    d=data_frame(h,cfg);m,opt,dl,a,n,windows=setup(family,d,cfg);loss=[]
    # CV-selected epoch count is frozen for refit; no test/future label evaluation.
    for epoch in range(epochs):
        if lr_schedule is not None:
            for group in opt.param_groups:group['lr']=lr_schedule[epoch]
        loss.append(train_epoch(family,m,opt,dl,cfg,n))
    payload={'family':family,'config':cfg,'architecture':a,'normalizer':n,'epochs':epochs,'state_dict':m.state_dict(),
             'history_end':str(h.index[-1]),'features':list(d.columns),'stored_context':d.iloc[-a['seq_len']:].values.tolist(),
             'stored_context_times':[str(t) for t in d.index[-a['seq_len']:]],'seed':SEED,'unit':'provisional guide kWh'}
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)
    pred=infer(family,m,d,a,n)
    again=replay(path);error=float(abs(again-pred).max());assert error==0
    return pred,{'saved_model':str(path),'normalizer':n,'history_end':str(h.index[-1]),'training_windows':windows,
       'epochs':epochs,'training_loss_first':loss[0],'training_loss_last':loss[-1],'target_index':n['target_index'],
       'target_loss':'Explicit aggregate target channel only','replay_max_native_error':error,'input_shape':[1,a['seq_len'],a['enc_in']],
       'output_shape':[1,a['pred_len'],a['enc_in']]}
def replay(path):
    seed();p=torch.load(path,map_location='cpu',weights_only=True);m=build(p['family'],p['architecture']);m.load_state_dict(p['state_dict'],strict=True)
    d=pd.DataFrame(p['stored_context'],columns=p['features'],index=pd.to_datetime(p['stored_context_times']))
    return infer(p['family'],m,d,p['architecture'],p['normalizer'])
def run(family):
    assert_training_permitted()
    y=load_y();dest=OUT/family;dest.mkdir(parents=True,exist_ok=True);cs=configs(family)
    write(dest/'search_plan.json',{'configs':cs,'stopping':'Native Late505 validation plateau + LRreduction, no small fixed epoch cap','seed':SEED})
    records=[];paths={};start=time.perf_counter()
    for ci,cfg in enumerate(cs):
        fs=[];folds=[]
        for fold,origin in enumerate(VALID,1):
            h=prefix(y,origin);epochs,early=choose_epochs(family,h,cfg,dest/'inner_best'/f'candidate{ci}_fold{fold}.pt')
            native_p,fit=fit_saved(family,h,cfg,epochs,dest/'cv_models'/f'candidate{ci}_fold{fold}.pt',early['refit_learning_rate_schedule'])
            fs.append(frame(family,'validation',fold,origin,y,hourly_map(native_p,cfg['resolution'])));folds.append({'origin':str(origin),'early':early,'fit':fit})
            print(f'{family}candidate{ci} fold{fold}:best{epochs}/run{early["epochs_run"]} epochs',flush=True)
            write(dest/'search_progress.json',{'candidate':ci,'fold':fold,'seconds':time.perf_counter()-start,'early':early})
        f=pd.concat(fs,ignore_index=True);stats=summary(f);records.append({'candidate':ci,'config':cfg,'validation':stats,'objective_mae_kwh':stats['mean_origin_late505_mae_kwh'],'folds':folds});paths[ci]=f
        write(dest/'candidate_results.json',{'candidates':records})
    chosen=min(records,key=lambda r:r['objective_mae_kwh']);cfg=chosen['config'];final_epochs=int(np.median([f['early']['best_epoch'] for f in chosen['folds']]))
    final_schedule=[float(np.median([f['early']['refit_learning_rate_schedule'][min(e,len(f['early']['refit_learning_rate_schedule'])-1)] for f in chosen['folds']])) for e in range(final_epochs)]
    write(dest/'selection_lock.json',{'chosen':chosen,'final_epochs':final_epochs,'selection_uses_test_or_May':False,'locked_before_test':True})
    paths[chosen['candidate']].to_csv(dest/'validation_predictions.csv',index=False,encoding='utf-8-sig');fits={};test_stats=None
    for split,origin in [('internal_test',TEST),('future_unlabelled',FINAL)]:
        h=prefix(y,origin)
        epochs,early=choose_epochs(family,h,cfg,dest/'inner_best'/'internal_test.pt') if split=='internal_test' else (final_epochs,{'epochs_frozen_on_CV':True,'no_April_tuning':True,'refit_learning_rate_schedule':final_schedule})
        native_p,fit=fit_saved(family,h,cfg,epochs,dest/f'{split}.pt',early['refit_learning_rate_schedule']);p=hourly_map(native_p,cfg['resolution'])
        f=frame(family,split,0,origin,y,p);csv=dest/('test_predictions.csv' if split=='internal_test' else 'future_path.csv');f.to_csv(csv,index=False,encoding='utf-8-sig')
        fits[split]={**fit,'early':early,'prediction_csv':str(csv)}
        if split=='internal_test':test_stats=summary(f)
    result={'family':family,'selected_config':cfg,'validation':chosen['validation'],'internal_test':test_stats,'fits':fits,
        'candidate_count':len(cs),'selection_lock':str(dest/'selection_lock.json'),'validation_prediction_csv':str(dest/'validation_predictions.csv'),
        'May_read':False,'seconds':time.perf_counter()-start,'convergence':'LRplateau stopped; selected measured best epoch, not epoch24',
        'model_claim':'Compact official 2023/2024 cores; no claim of contemporary SOTA superiority'}
    write(dest/'results.json',result);print({family:result['validation']['mean_origin_late505_mae_kwh']},flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['patchtst','itransformer'],required=True);run(p.parse_args().family)
