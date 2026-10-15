"""Compact official iTransformer: direct P745, past 13 module-energy variates.

Nested chronological epoch selection, outer validation configuration selection,
frozen internal test. This is local architecture training, not paper reproduction.
"""
from pathlib import Path
from types import SimpleNamespace
import sys, json, copy, time, argparse, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from common import (ROOT, OUT, load_training, future_index, history_before,
                    actual_at, score, score_prefixes, prediction_frame,
                    write_json, VALIDATION_ORIGIN, TEST_ORIGIN)
from itransformer.vendor.model.iTransformer import Model

DEST = OUT/'itransformer'
SEED = 2026
P = 745

def seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)

def module_data(series):
    raw = pd.read_csv(ROOT/'output/analysis/hourly_by_equipment.csv')
    raw = raw[raw['complete_720'].astype(str).str.lower().isin(['true','1'])].copy()
    raw['hour'] = pd.to_datetime(raw['hour'])
    wide = raw.pivot(index='hour',columns='module(equipment)',values='energy_kwh').sort_index().reindex(series.index)
    if wide.shape!=(3600,13) or wide.isna().any().any():
        raise ValueError('Module hourly matrix is not complete 3600 by 13')
    np.testing.assert_allclose(wide.sum(axis=1),series.values,rtol=0,atol=1e-6)
    return wide

def config(context):
    return dict(seq_len=context,pred_len=P,d_model=32,d_ff=64,e_layers=1,
                n_heads=4,dropout=0.1,output_attention=False,use_norm=True,
                embed='timeF',freq='h',class_strategy='projection',factor=5,activation='gelu')

class Windows(Dataset):
    def __init__(self,x,context):
        self.x=torch.tensor(x,dtype=torch.float32); self.context=context
        self.ends=list(range(context,len(x)-P+1,4))
        if not self.ends: raise ValueError('No fully observed supervised window')
    def __len__(self): return len(self.ends)
    def __getitem__(self,i):
        end=self.ends[i]
        return self.x[end-self.context:end],self.x[end:end+P]

def fit(history,cfg,epochs,holdout=None,earlystop=False):
    seed(); started=time.perf_counter()
    x=np.asarray(history,dtype=float)
    mu=x.mean(axis=0); sigma=x.std(axis=0); sigma=np.maximum(sigma,1e-8)
    ds=Windows((x-mu)/sigma,cfg['seq_len'])
    loader=DataLoader(ds,batch_size=32,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(SEED))
    model=Model(SimpleNamespace(**cfg)); opt=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01)
    logs=[]; best=float('inf'); best_epoch=1; stale=0; best_state=None
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        for xb,yb in loader:
            opt.zero_grad(); pred=model(xb,None,None,None)
            loss=torch.mean((pred-yb)**2); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach()))
        row={'epoch':epoch,'train_mse_standardized':float(np.mean(losses))}
        if holdout is not None:
            pred=infer(model,x,cfg,mu,sigma)
            value=float(np.abs(pred.sum(axis=1)-np.asarray(holdout).sum(axis=1)).mean())
            row['inner_tail_mae_kwh']=value
            if value<best:
                best=value; best_epoch=epoch; stale=0; best_state=copy.deepcopy(model.state_dict())
            else: stale+=1
        logs.append(row)
        if earlystop and stale>=5: break
    if earlystop and best_state is not None: model.load_state_dict(best_state)
    else: best_epoch=epochs
    return model,mu,sigma,{'epochs':best_epoch,'train_windows':len(ds),
             'history_hours':len(x),'latest_supervised_target_index':max(ds.ends)+P-1,
             'fit_seconds':time.perf_counter()-started,'logs':logs,
             'parameters':sum(p.numel() for p in model.parameters())}

def infer(model,history,cfg,mu,sigma):
    model.eval()
    xb=torch.tensor((np.asarray(history)[-cfg['seq_len']:]-mu)/sigma,dtype=torch.float32).unsqueeze(0)
    with torch.no_grad(): pred=model(xb,None,None,None).squeeze(0).numpy()
    return pred*sigma+mu

def save(path,model,cfg,mu,sigma,columns,history_end):
    torch.save({'state_dict':model.state_dict(),'config':cfg,'mean':mu.tolist(),'scale':sigma.tolist(),
                'columns':list(columns),'history_end':str(history_end),'seed':SEED},path)

def load(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    m=Model(SimpleNamespace(**saved['config'])); m.load_state_dict(saved['state_dict']); m.eval()
    return m,saved

def main(final=False):
    series=load_training(); wide=module_data(series); DEST.mkdir(parents=True,exist_ok=True)
    if final:
        lock=json.loads((DEST/'selection_lock.json').read_text(encoding='utf-8'))
        cfg=lock['config']; epochs=lock['epochs']
        m,mu,sd,fitlog=fit(wide.to_numpy(),cfg,epochs)
        pred=infer(m,wide,cfg,mu,sd).sum(axis=1)
        save(DEST/'final_model.pt',m,cfg,mu,sd,wide.columns,series.index[-1])
        reloaded,saved=load(DEST/'final_model.pt')
        check=infer(reloaded,wide,cfg,np.array(saved['mean']),np.array(saved['scale'])).sum(axis=1)
        np.testing.assert_allclose(pred,check,rtol=0,atol=1e-8)
        origin=series.index[-1]+pd.Timedelta(hours=1); future=future_index(origin)
        prediction_frame('iTransformer','may_full_path',origin,future,np.full(P,np.nan),pred).to_csv(DEST/'may_full_path.csv',index=False)
        write_json(DEST/'final_fit.json',fitlog)
        print('iTransformer final refit and reload verified, May labels not read.',flush=True)
        return
    origin=VALIDATION_ORIGIN; hist=wide.loc[wide.index<origin]; future=future_index(origin); actual=actual_at(series,future)
    candidates=[]; frames=[]
    for ctx in [168,336]:
        cfg=config(ctx); inner=hist.iloc[:-P]; inner_tail=hist.iloc[-P:]
        _,_,_,innerlog=fit(inner.to_numpy(),cfg,30,holdout=inner_tail.to_numpy(),earlystop=True)
        m,mu,sd,fitlog=fit(hist.to_numpy(),cfg,innerlog['epochs'])
        pred=infer(m,hist,cfg,mu,sd).sum(axis=1)
        metrics=score_prefixes(actual,pred)
        candidates.append({'config':cfg,'epochs':innerlog['epochs'],'inner_fit':innerlog,'outer_fit':fitlog,'validation_metrics':metrics})
        print('iTransformer',ctx,'epoch',innerlog['epochs'],'validP745MAE',metrics['P745']['mae_kwh'],flush=True)
    best=min(candidates,key=lambda c:c['validation_metrics']['P745']['mae_kwh'])
    lock={'model':'iTransformer','config':best['config'],'epochs':best['epochs'],
          'selection_basis':'Lowest aggregate P745 MAE on blocked Feb25 outer validation only',
          'seed':SEED,'input':'Past 13 module hourly energy variates; output module forecasts summed; extra multivariate history compared with univariate competitors',
          'status':'frozen_before_May_labels','method_note':'Official compact architecture, not published benchmark reproduction'}
    write_json(DEST/'selection_lock.json',lock)
    result={'model':'iTransformer','candidates':candidates,'selected':lock,'splits':{}}
    for name,origin in [('validation',VALIDATION_ORIGIN),('internal_test',TEST_ORIGIN)]:
        hist=wide.loc[wide.index<origin]; future=future_index(origin); actual=actual_at(series,future)
        m,mu,sd,fitlog=fit(hist.to_numpy(),best['config'],best['epochs'])
        pred=infer(m,hist,best['config'],mu,sd).sum(axis=1)
        result['splits'][name]={'metrics':score_prefixes(actual,pred),'fit':fitlog}
        frames.append(prediction_frame('iTransformer',name,origin,future,actual,pred))
        save(DEST/f'{name}_model.pt',m,best['config'],mu,sd,wide.columns,hist.index[-1])
    pd.concat(frames).to_csv(DEST/'internal_predictions.csv',index=False)
    write_json(DEST/'results.json',result)
    print('iTransformer internal results locked.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--final',action='store_true')
    main(parser.parse_args().final)
