"""Compact official PatchTST: honest chronological selection and direct P745.

Architecture is the pinned official supervised core, with import namespaces only
modified. This is not a reproduction of original-paper benchmark scores.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import (load_training,protocol,history_before,future_index,actual_at,
                    VALIDATION_ORIGIN,TEST_ORIGIN,score,score_prefixes,
                    prediction_frame,write_json,may_labels_after_lock)
from patchtst_vendor.models.PatchTST import Model

PRED_LEN = 745
SEED = 2026
MAX_EPOCHS = 30
PATIENCE = 5
THREADS = 4
CONFIGS = [{'seq_len':168},{'seq_len':336}]
FIXED = {'enc_in':1,'pred_len':PRED_LEN,'e_layers':1,'n_heads':4,'d_model':32,
         'd_ff':64,'dropout':0.1,'fc_dropout':0.1,'head_dropout':0.1,
         'individual':False,'patch_len':16,'stride':8,'padding_patch':'end',
         'revin':True,'affine':True,'subtract_last':False,'decomposition':False,'kernel_size':25}
OPTIMIZATION = {'learning_rate':0.001,'weight_decay':0.0001,'batch_size':32,
                'loss':'MSE on training-only globally standardized targets',
                'gradient_clip_norm':1.0,'seed':SEED,'threads':THREADS,
                'max_epochs':MAX_EPOCHS,'early_stop_patience':PATIENCE}

def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)

def verify_provenance():
    provenance = json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))
    for record in provenance['files']:
        original = HERE/'upstream'/record['upstream_path']
        if hashlib.sha256(original.read_bytes()).hexdigest()!=record['original_sha256']:
            raise ValueError('Pinned upstream file changed')
        vendored = Path(record['vendored_path'])
        if hashlib.sha256(vendored.read_bytes()).hexdigest()!=record['vendored_sha256']:
            raise ValueError('Vendored official architecture changed')
        source = original.read_text(encoding='utf-8')
        expected = source.replace('from layers.','from patchtst_vendor.layers.').replace('import layers.','import patchtst_vendor.layers.')
        if vendored.read_text(encoding='utf-8')!=expected:
            raise ValueError('Vendor modification exceeds import namespaces')
    return provenance

class Windows(Dataset):
    def __init__(self,values,context):
        self.values = torch.from_numpy(np.asarray(values,dtype=np.float32)).view(-1,1)
        self.context = int(context)
        self.origins = np.arange(self.context,len(values)-PRED_LEN+1,dtype=int)
        if len(self.origins)==0:
            raise ValueError('History cannot contain a full context plus P745 training window')
        # Adjacent training targets overlap intentionally; all remain inside the
        # supplied training cutoff. Outer/tail evaluation targets are disjoint.
        assert self.origins[-1]+PRED_LEN-1 == len(values)-1
    def __len__(self):
        return len(self.origins)
    def __getitem__(self,index):
        origin = int(self.origins[index])
        return self.values[origin-self.context:origin],self.values[origin:origin+PRED_LEN]

def normalizer(history):
    mean = float(history.mean())
    std = float(history.std(ddof=0))
    if not np.isfinite(std) or std<=0:
        raise ValueError('Nonpositive training scale')
    return mean,std

def build_model(config):
    return Model(SimpleNamespace(**{**FIXED,**config})).cpu()

def predict(model,history,config,mean,std):
    context = int(config['seq_len'])
    inp = ((history.to_numpy(dtype=float)[-context:]-mean)/std).astype(np.float32)
    tensor = torch.from_numpy(inp).view(1,context,1)
    model.eval()
    with torch.no_grad():
        output = model(tensor)
    if tuple(output.shape)!=(1,PRED_LEN,1):
        raise ValueError(f'Unexpected official PatchTST output {tuple(output.shape)}')
    result = output[0,:,0].numpy().astype(float)*std+mean
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite PatchTST forecast')
    return result

def setup_fit(history,config):
    seed_everything()
    mean,std = normalizer(history)
    dataset = Windows((history.to_numpy(dtype=float)-mean)/std,config['seq_len'])
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset,batch_size=OPTIMIZATION['batch_size'],shuffle=True,
                        num_workers=0,generator=generator,drop_last=False)
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(),lr=OPTIMIZATION['learning_rate'],
                                 weight_decay=OPTIMIZATION['weight_decay'])
    return model,optimizer,loader,mean,std,dataset

def one_epoch(model,optimizer,loader):
    model.train()
    losses = []
    for X,y in loader:
        optimizer.zero_grad(set_to_none=True)
        prediction = model(X)
        if prediction.shape!=y.shape or prediction.shape[1]!=PRED_LEN:
            raise ValueError('Training target alignment differs from P745')
        loss = torch.mean((prediction-y)**2)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),OPTIMIZATION['gradient_clip_norm'])
        optimizer.step()
        losses.append((float(loss.detach()),len(X)))
    return sum(loss*n for loss,n in losses)/sum(n for _,n in losses)

def checkpoint(path,model,config,mean,std,epochs,history,metadata):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload = {'state_dict':model.state_dict(),'config':{**FIXED,**config},'normalization_mean':mean,
               'normalization_std':std,'epochs':int(epochs),'seed':SEED,
               'history_count':len(history),'history_start':str(history.index[0]),
               'history_end':str(history.index[-1]),'metadata':metadata}
    torch.save(payload,path)
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'parameter_count':sum(p.numel() for p in model.parameters())}

def load_saved_model(path):
    """Load this task's tensor checkpoint for repeatable CPU inference."""
    verify_provenance()
    payload = torch.load(Path(path),map_location='cpu',weights_only=True)
    for key,value in FIXED.items():
        if payload['config'][key]!=value:
            raise ValueError(f'Saved architecture differs from locked configuration: {key}')
    config = {'seq_len':int(payload['config']['seq_len'])}
    model = build_model(config)
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model,config,float(payload['normalization_mean']),float(payload['normalization_std'])

def choose_epoch(history,config):
    # A full P745 tail prevents overlap between training-window labels and the
    # chronological early-stop target. Tail input context is strictly pre-tail.
    split = len(history)-PRED_LEN
    fitting_history = history.iloc[:split]
    tail = history.iloc[split:]
    if len(fitting_history)<960:
        raise ValueError('Inner fitting history is below 40 days')
    model,optimizer,loader,mean,std,dataset = setup_fit(fitting_history,config)
    assert fitting_history.index[-1] < tail.index[0]
    assert dataset.origins[-1]+PRED_LEN-1 < split
    logs = []
    best_score = float('inf')
    best_epoch = None
    best_state = None
    stale = 0
    started = time.monotonic()
    for epoch in range(1,MAX_EPOCHS+1):
        train_loss = one_epoch(model,optimizer,loader)
        path = predict(model,fitting_history,config,mean,std)
        tail_score = score(tail.to_numpy(),path)['mae_kwh']
        logs.append({'epoch':epoch,'training_mse_standardized':train_loss,'tail_fixed_P745_mae_kwh':tail_score})
        if tail_score<best_score-1e-7:
            best_score = tail_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale>=PATIENCE:
            break
    model.load_state_dict(best_state)
    metadata = {'context':config['seq_len'],'inner_history_count':split,
                'inner_history_end':str(fitting_history.index[-1]),'inner_tail_origin':str(tail.index[0]),
                'inner_tail_end':str(tail.index[-1]),'inner_tail_count':len(tail),
                'inner_training_windows':len(dataset),'training_target_tail_overlap':False,
                'best_epoch':best_epoch,'best_tail_mae_kwh':best_score,'epochs_run':len(logs),
                'epoch_logs':logs,'runtime_seconds':time.monotonic()-started}
    saved = checkpoint(HERE/'checkpoints'/f'inner_context_{config["seq_len"]}.pt',model,config,mean,std,best_epoch,fitting_history,metadata)
    metadata['checkpoint'] = saved
    return best_epoch,metadata

def refit(history,config,epochs,path):
    model,optimizer,loader,mean,std,dataset = setup_fit(history,config)
    logs = []
    started = time.monotonic()
    for epoch in range(1,int(epochs)+1):
        logs.append({'epoch':epoch,'training_mse_standardized':one_epoch(model,optimizer,loader)})
    metadata = {'epochs_frozen':int(epochs),'training_windows':len(dataset),
                'training_max_label':str(history.index[-1]),
                'runtime_seconds':time.monotonic()-started,'epoch_logs':logs}
    metadata['checkpoint'] = checkpoint(path,model,config,mean,std,epochs,history,metadata)
    return model,mean,std,metadata

def internal():
    provenance = verify_provenance()
    series = load_training()
    manifest = protocol(series)
    valid_history = history_before(series,VALIDATION_ORIGIN)
    valid_future = future_index(VALIDATION_ORIGIN)
    valid_actual = actual_at(series,valid_future)
    candidates = []
    for config in CONFIGS:
        epoch,inner = choose_epoch(valid_history,config)
        model,mean,std,fit = refit(valid_history,config,epoch,HERE/'checkpoints'/f'validation_context_{config["seq_len"]}.pt')
        path = predict(model,valid_history,config,mean,std)
        candidate = {'config':config,'frozen_epoch':epoch,'inner_early_stop':inner,
                     'full_validation_history_refit':fit,'validation':score_prefixes(valid_actual,path)}
        candidates.append((candidate,path))
        write_json(HERE/f'candidate_context_{config["seq_len"]}.json',candidate)
        print(f'PatchTST context={config["seq_len"]} epoch={epoch} valid P745 MAE={candidate["validation"]["P745"]["mae_kwh"]:.6f}',flush=True)
    chosen,path = min(candidates,key=lambda item:item[0]['validation']['P745']['mae_kwh'])
    config = chosen['config']
    epochs = chosen['frozen_epoch']
    lock = {'model':'patchtst','selected_config':{**FIXED,**config},'selected_epoch':epochs,
            'optimizer':OPTIMIZATION,'selection':'Lowest outer blocked validation P745 MAE after training-only chronological inner early-stop epoch selection',
            'selection_uses_internal_test_labels':False,'selection_uses_may_labels':False,
            'validation':chosen['validation'],'candidates':[item[0] for item in candidates],
            'protocol':manifest,'provenance_file':str(HERE/'provenance.json'),
            'architecture_scope':provenance['architecture_scope'],
            'input_scope':'Univariate aggregate hourly target only; no module channels or sensor/calendar exogenous inputs',
            'honest_may_EDA_disclosure':manifest['honest_holdout_disclosure']}
    write_json(HERE/'selection_lock.json',lock)
    # The selected architecture and epoch are saved before the internal test fit.
    test_history = history_before(series,TEST_ORIGIN)
    test_future = future_index(TEST_ORIGIN)
    test_actual = actual_at(series,test_future)
    model,mean,std,test_fit = refit(test_history,config,epochs,HERE/'checkpoints'/'internal_test.pt')
    test_path = predict(model,test_history,config,mean,std)
    frames = [prediction_frame('patchtst','validation',VALIDATION_ORIGIN,valid_future,valid_actual,path),
              prediction_frame('patchtst','internal_test',TEST_ORIGIN,test_future,test_actual,test_path)]
    pd.concat(frames,ignore_index=True).to_csv(HERE/'internal_predictions.csv',index=False,encoding='utf-8-sig')
    result = {'model':'patchtst','frozen_config':lock['selected_config'],'frozen_epochs':epochs,
              'validation':chosen['validation'],'internal_test':score_prefixes(test_actual,test_path),
              'internal_test_fit':test_fit,'selection_lock':str(HERE/'selection_lock.json'),
              'provenance_file':str(HERE/'provenance.json'),'architecture_scope':lock['architecture_scope'],
              'final_checkpoint_planned_path':str(HERE/'checkpoints'/'final_3600.pt'),
              'may_labels_read':False}
    write_json(HERE/'results.json',result)
    print(json.dumps({'selected_context':config['seq_len'],'selected_epoch':epochs,
                      'validation_P745':result['validation']['P745'],'internal_test_P745':result['internal_test']['P745']},indent=2),flush=True)

def final_refit():
    verify_provenance()
    locked = json.loads((HERE/'selection_lock.json').read_text(encoding='utf-8'))
    if locked['selection_uses_may_labels'] or locked['selection_uses_internal_test_labels']:
        raise ValueError('Invalid selection lock')
    series = load_training()
    current = protocol(series)
    if current['target_source_sha256']!=locked['protocol']['target_source_sha256']:
        raise ValueError('Training data changed after configuration lock')
    config = {'seq_len':locked['selected_config']['seq_len']}
    model,mean,std,fit = refit(series,config,locked['selected_epoch'],HERE/'checkpoints'/'final_3600.pt')
    origin = series.index[-1]+pd.Timedelta(hours=1)
    future = future_index(origin)
    path = predict(model,series,config,mean,std)
    prediction_frame('patchtst','may_full_bridge',origin,future,np.repeat(np.nan,PRED_LEN),path).to_csv(HERE/'may_full_bridge.csv',index=False,encoding='utf-8-sig')
    result = json.loads((HERE/'results.json').read_text(encoding='utf-8'))
    result['final_full_history_fit'] = fit
    result['final_checkpoint'] = fit['checkpoint']
    result['full_forecast_path_rows'] = len(path)
    result['full_forecast_origin'] = str(origin)
    result['full_forecast_end'] = str(future[-1])
    result['may_labels_read'] = False
    write_json(HERE/'results.json',result)
    print(json.dumps({'final_checkpoint':fit['checkpoint'],'forecast_rows':len(path),
                      'forecast_origin':str(origin),'forecast_end':str(future[-1]),'may_labels_read':False},indent=2),flush=True)

def may_score():
    # A separate post-global-lock step; no refitting or hyperparameter selection.
    locked = json.loads((HERE/'selection_lock.json').read_text(encoding='utf-8'))
    if locked['selection_uses_may_labels'] or locked['selection_uses_internal_test_labels']:
        raise ValueError('Invalid selection lock')
    bridge = pd.read_csv(HERE/'may_full_bridge.csv')
    bridge['id'] = pd.to_datetime(bridge['id'])
    origin = pd.Timestamp(bridge['origin'].iloc[0])
    labels = may_labels_after_lock()
    predicted = bridge.set_index('id')['predicted_kwh'].reindex(labels.index)
    if predicted.isna().any():
        raise ValueError('Incomplete May bridge')
    frame = prediction_frame('patchtst','may_labelled',origin,labels.index,labels.to_numpy(),predicted.to_numpy())
    frame.to_csv(HERE/'may_labelled_predictions.csv',index=False,encoding='utf-8-sig')
    result = json.loads((HERE/'results.json').read_text(encoding='utf-8'))
    result['may_505_labelled_hours'] = score(labels.to_numpy(),predicted.to_numpy())
    result['may_labels_read'] = True
    result['may_scores_do_not_change_selection'] = True
    write_json(HERE/'results.json',result)
    print(json.dumps(result['may_505_labelled_hours'],indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['internal','final-refit','may-score'],default='internal')
    args = parser.parse_args()
    {'internal':internal,'final-refit':final_refit,'may-score':may_score}[args.stage]()

if __name__=='__main__':
    main()
