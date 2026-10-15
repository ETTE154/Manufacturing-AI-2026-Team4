"""Append one synthetic paired-target denoising AE; preserve the seven detectors.

Corrupted inputs are reconstructed against the exact same source observation,
never against the corrupted input itself. March original weak-normal proxy alone
determines early stopping and the frozen threshold. Known artificial masks are
supervision, not real-fault labels. Test data never influence model selection.
"""
from pathlib import Path
import copy,json,pickle,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import RobustScaler
from anomaly_study import SENSORS,OUT,OLD,SEED,score,metrics,dump
from predict_anomaly import AE

NAME='ae_synthetic_denoising'

def main():
    started=time.monotonic();torch.set_num_threads(4);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    report=json.loads((OUT/'anomaly_metrics.json').read_text(encoding='utf-8'))
    if NAME in report['models']:raise RuntimeError('Denoising AE already evaluated; avoid overwriting completed experiment')
    sample=pd.read_csv(OLD/'uniform_raw_sample.csv.gz',parse_dates=['dt'])
    tr=sample[(sample.split=='train')&(sample.weak_candidate==0)]
    va=sample[(sample.split=='validation')&(sample.weak_candidate==0)]
    windows=pd.read_csv(OUT/'continuous_source_windows.csv.gz',parse_dates=['dt'])
    source_index=windows.set_index(['window_id','localtime'])[SENSORS]
    synt=pd.read_csv(OUT/'synthetic_observations.csv.gz',parse_dates=['dt'])
    train_corrupted=synt[synt.split=='train'].copy()
    keys=pd.MultiIndex.from_arrays([train_corrupted.window_id,train_corrupted.localtime],names=['window_id','localtime'])
    clean=source_index.reindex(keys).reset_index(drop=True)
    assert len(clean)==len(train_corrupted) and np.isfinite(clean.to_numpy()).all()
    # Every known mask truly maps to the unmodified source teacher.
    delta=train_corrupted[SENSORS].to_numpy()-clean.to_numpy()
    mask=train_corrupted.synthetic_label.to_numpy().astype(bool)
    assert np.allclose(delta[~mask],0) and (np.abs(delta[mask]).max(axis=1)>0).all()
    scaler=RobustScaler().fit(tr[SENSORS])
    original_x=scaler.transform(tr[SENSORS]).astype(np.float32)
    input_x=scaler.transform(train_corrupted[SENSORS]).astype(np.float32)
    teacher_y=scaler.transform(clean).astype(np.float32)
    train_x=np.vstack([original_x,input_x]);train_y=np.vstack([original_x,teacher_y])
    valid_x=scaler.transform(va[SENSORS]).astype(np.float32)
    dataset=TensorDataset(torch.from_numpy(train_x),torch.from_numpy(train_y))
    loader=DataLoader(dataset,batch_size=1024,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(SEED))
    vt=torch.from_numpy(valid_x);model=AE(len(SENSORS));optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5)
    best=float('inf');state=None;patience=0;history=[]
    for epoch in range(30):
        model.train();loss_sum=0
        for corrupted,teacher in loader:
            optimizer.zero_grad(set_to_none=True);loss=(model(corrupted)-teacher).pow(2).mean();loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(corrupted)
        model.eval()
        with torch.no_grad():
            val_loss=float((model(vt)-vt).pow(2).mean())
        history.append({'epoch':epoch+1,'paired_training_mse':loss_sum/len(dataset),'march_original_weak_normal_mse':val_loss})
        if val_loss<best-1e-6:
            best=val_loss;state=copy.deepcopy(model.state_dict());patience=0
        else:patience+=1
        if epoch%5==0:print(f'denoising epoch {epoch+1}: March original proxy MSE={val_loss:.6f}, {time.monotonic()-started:.1f}s',flush=True)
        if patience>=5:break
    model.load_state_dict(state);model.eval()
    saved={'kind':'ae','model':model,'scaler':scaler,'features':SENSORS}
    threshold=float(np.nextafter(np.quantile(score(va,saved),.995),np.inf));saved['threshold']=threshold
    threshold_basis='March ORIGINAL weak-normal reconstruction scores q99.5, nextafter(float64,+inf), frozen; no synthetic/test threshold optimization'
    torch.save({'state_dict':model.state_dict(),'features':SENSORS,'architecture':[14,32,12,4,12,32,14],
                'threshold':threshold,'learning':'synthetic paired-target denoising, semi-supervised'},OUT/'models'/f'{NAME}.pt')
    with (OUT/'models'/f'{NAME}_scaler.pkl').open('wb') as f:pickle.dump(scaler,f)
    te=sample[sample.split=='test']
    candidates=pd.read_csv(OUT/'full_weak_candidate_rows.csv.gz',parse_dates=['dt'])
    st=synt[synt.split=='test'].reset_index(drop=True)
    basekeys=pd.MultiIndex.from_arrays([st.window_id,st.localtime],names=['window_id','localtime'])
    control=source_index.reindex(basekeys).reset_index(drop=True)
    p=score(te,saved);vp=score(sample[sample.split=='validation'],saved);wp=score(windows,saved);cp=score(candidates,saved);sp=score(st,saved);bp=score(control,saved)
    info={'display_name':'합성 paired-target denoising AE','kind':'ae','features':SENSORS,'threshold':threshold,'threshold_basis':threshold_basis,
          'training':'Synthetic paired corrupted input -> EXACT clean source teacher target, plus original weak-normal identity reconstruction; semi-supervised denoising supervision',
          'architecture':[14,32,12,4,12,32,14],'scaler_fit':'ONLY Dec–Feb original weak-normal RobustScaler median/IQR',
          'original_normal_training_rows':len(tr),'paired_window_training_rows':len(train_corrupted),'paired_injected_training_rows':int(mask.sum()),
          'total_training_pairs':len(dataset),'epochs_run':len(history),'maximum_epochs':30,'early_stop_patience':5,
          'early_stopping':'March ORIGINAL weak-normal reconstruction MSE only; no March synthetic labels/scores used for checkpoint choice',
          'best_march_original_normal_mse':best,'history':history,'elapsed_seconds':time.monotonic()-started,
          'supervision_limit':'Known artificial corruption masks and clean targets are supervision; not fully unsupervised and not actual fault labels',
          'score_definition':'Mean standardized squared (observed input - predicted clean sensors), not distance to an unavailable clean test teacher',
          'original_weak_agreement':metrics(te.weak_candidate,p,threshold),
          'validation_weak_agreement':metrics(sample.loc[sample.split=='validation','weak_candidate'],vp,threshold),
          'all_source_candidate_capture':{split:{'rows':int((candidates.split==split).sum()),'captured':int((cp[candidates.split==split]>=threshold).sum()),'capture_rate':float((cp[candidates.split==split]>=threshold).mean())} for split in ['train','validation','test']},
          'synthetic_heldout_metrics':metrics(st.synthetic_label,sp,threshold),
          'synthetic_by_type':{typ:metrics(d.synthetic_label,sp[d.index],threshold) for typ,d in st.groupby('synthetic_type')}}
    events=json.loads((OUT/'augmented_sample_manifest.json').read_text(encoding='utf-8'))['events'];byid={e['event_id']:e for e in events};rows=[]
    for eid,idx in st.groupby('event_id',sort=False).indices.items():
        e=byid[eid];m=st.iloc[idx].synthetic_label.to_numpy().astype(bool);sa=sp[idx][m]>=threshold;ba=bp[idx][m]>=threshold;hit=np.flatnonzero(sa)
        rows.append({'event_id':eid,'type':e['type'],'feature':e['feature'],'amplitude_iqr':abs(e['signed_amplitude_iqr']),
                     'duration_seconds':e['duration_seconds'],'detected_any':bool(sa.any()),'original_control_alarm_any':bool(ba.any()),
                     'injected_point_recall':float(sa.mean()),'original_control_point_alarm_rate':float(ba.mean()),'first_alarm_delay_seconds':int(hit[0]*5) if len(hit) else None})
    ef=pd.DataFrame(rows)
    info['synthetic_event_detection']={'events':len(ef),'detected_any_fraction':float(ef.detected_any.mean()),
           'paired_original_control_any_alarm_fraction':float(ef.original_control_alarm_any.mean()),
           'any_alarm_limit':'Longer events have more chances for incidental alarms; compare paired original controls',
           'by_type':ef.groupby('type').agg(events=('event_id','size'),detected_any_fraction=('detected_any','mean'),paired_original_any_alarm_fraction=('original_control_alarm_any','mean'),mean_injected_point_recall=('injected_point_recall','mean')).reset_index().to_dict('records'),
           'by_amplitude_iqr':ef.groupby('amplitude_iqr').agg(events=('event_id','size'),detected_any_fraction=('detected_any','mean'),mean_injected_point_recall=('injected_point_recall','mean')).reset_index().to_dict('records')}
    ef.to_csv(OUT/f'synthetic_event_results_{NAME}.csv',index=False,float_format='%.17g')
    for fn,arr in [('original_test_scores.csv.gz',p),('continuous_window_scores.csv.gz',wp),('full_candidate_scores.csv.gz',cp),('synthetic_test_scores.csv.gz',sp)]:
        df=pd.read_csv(OUT/fn,float_precision='round_trip');assert len(df)==len(arr)
        df[NAME+'_score']=arr;df[NAME+'_alarm']=(arr>=threshold).astype(np.int8)
        if fn=='synthetic_test_scores.csv.gz':df[NAME+'_paired_original_score']=bp
        df.to_csv(OUT/fn,index=False,compression='gzip',float_format='%.17g')
    locks=json.loads((OUT/'threshold_locks.json').read_text(encoding='utf-8'));locks[NAME]={'threshold':threshold,'threshold_basis':threshold_basis,'alarm_comparison':'float64 score >= float64 threshold','features':SENSORS,'kind':'ae'}
    dump(locks,'threshold_locks.json');report['models'][NAME]=info
    report['synthetic']['augmentation_training_models']=['xgb_synthetic_augmented',NAME]
    report['synthetic']['clean_teacher_verified']=True
    dump(report,'anomaly_metrics.json')
    dump({'model':NAME,'clean_source_teacher_exact_keys_verified':True,'unmodified_mask_false_rows_verified':True,
          'training_only_clean_proxy_scaler_fit':True,'march_original_only_early_stop_threshold':True,
          'test_usage':'Only final evaluation; no model/checkpoint/threshold selection','training_pairs':len(dataset),'elapsed_seconds':time.monotonic()-started},'denoising_pair_audit.json')
    print(f'Appended denoising AE: original weak precision={info["original_weak_agreement"]["precision"]:.4f}; synthetic recall={info["synthetic_heldout_metrics"]["recall"]:.4f}; fit epochs={len(history)}',flush=True)

if __name__=='__main__':main()
