"""Current-observation electrical anomaly study; never claim ground-truth fault labels.

Run --extract-only once, then rerun after sklearn, torch, and xgboost are installed.
Optional --label-file CSV: module(equipment), localtime, label (0/1), exact observations.
Unknown observations stay unlabeled; positive-only lists are never interpreted as all
other observations being normal. Source RTU files are read-only.
"""
from pathlib import Path
import os, sys, argparse, hashlib, json, time
import numpy as np
import pandas as pd
import torch
from torch import nn

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
OUT.mkdir(parents=True,exist_ok=True)
SEED=20261011
SENSORS=['voltageR','voltageS','voltageT','voltageRS','voltageST','voltageTR',
         'currentR','currentS','currentT','activePower','powerFactorR','powerFactorS',
         'powerFactorT','reactivePowerLagging']
TRANSFER=['currentR','currentS','currentT','activePower','reactivePowerLagging']
SAMPLE=OUT/'uniform_raw_sample.csv.gz'

class AE(nn.Module):
    """Shared saved reconstruction model for training and standalone inference."""
    def __init__(self,p):
        super().__init__();self.network=nn.Sequential(nn.Linear(p,32),nn.ReLU(),nn.Linear(32,12),nn.ReLU(),nn.Linear(12,4),nn.ReLU(),nn.Linear(4,12),nn.ReLU(),nn.Linear(12,32),nn.ReLU(),nn.Linear(32,p))
    def forward(self,x):return self.network(x)

def dump(obj,name):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def weak_labels(d):
    voltage=(d[['voltageR','voltageS','voltageT']].lt(205)|d[['voltageR','voltageS','voltageT']].gt(225)).any(axis=1)
    pf=(d[['powerFactorR','powerFactorS','powerFactorT']].lt(80)|d[['powerFactorR','powerFactorS','powerFactorT']].gt(100)).any(axis=1)
    domain=d[['activePower','currentR','currentS','currentT']].lt(0).any(axis=1)|~d.operation.isin([0,1])
    d['voltage_candidate']=voltage.astype(int);d['pf_candidate']=pf.astype(int);d['domain_candidate']=domain.astype(int)
    d['weak_candidate']=(voltage|pf|domain).astype(int)
    return d

def extract_sample():
    source=ROOT/'traing-dataset'/'rtu_data_full.csv'
    parts=[];seen=0;start=time.monotonic();slot_cache={}
    for ci,d in enumerate(pd.read_csv(source,chunksize=300_000)):
        seen+=len(d)
        local=d.localtime.to_numpy(dtype=np.int64)
        hour=local//10000
        slot=((local%10000//100)*60+local%100)//5
        d['_hour']=hour
        take=[]
        for (module,hournum),indices in d.groupby(['module(equipment)','_hour'],sort=False).indices.items():
            if int(hournum)>=2025043000:continue
            key=f'{module}|{hournum}'
            if key not in slot_cache:
                seed=int.from_bytes(hashlib.blake2b(f'{SEED}|{key}'.encode(),digest_size=8).digest(),'little')
                slot_cache[key]=np.random.default_rng(seed).choice(720,4,replace=False)
            take.extend(indices[np.isin(slot[indices],slot_cache[key])].tolist())
        if take:parts.append(d.iloc[take].drop(columns='_hour'))
        if ci%10==0:print(f'extract chunk={ci} source_rows={seen:,} sample_rows={sum(map(len,parts)):,} elapsed_s={time.monotonic()-start:.1f}',flush=True)
    sample=pd.concat(parts,ignore_index=True)
    sample['dt']=pd.to_datetime(sample.localtime.astype(str),format='%Y%m%d%H%M%S')
    sample=sample.sort_values(['dt','module(equipment)']).reset_index(drop=True)
    sample=weak_labels(sample)
    counts=sample.groupby(['module(equipment)',sample.dt.dt.floor('h')]).size()
    assert len(counts)==46800 and counts.eq(4).all(),counts.describe().to_dict()
    assert not sample.duplicated(['module(equipment)','dt']).any()
    sample['split']=np.select([sample.dt.lt('2025-03-01'),sample.dt.lt('2025-04-01')],['train','validation'],default='test')
    sample.to_csv(SAMPLE,index=False,compression='gzip')
    summary={'source':str(source),'source_rows_read':seen,'sample_rows':len(sample),'seed':SEED,
             'sampling':'Each module x complete localtime hour: four unique 5-second slots selected uniformly without replacement with deterministic stratum seed; no candidate oversampling',
             'inclusion_probability_per_full_hour':4/720,'complete_strata':len(counts),
             'partial_2025_04_30_endpoint_excluded':True,'timezone':'Guide localtime; Unix timestamp mismatches audited previously',
             'splits':sample.groupby('split',sort=False).agg(rows=('weak_candidate','size'),candidate_rows=('weak_candidate','sum'),first=('dt','min'),last=('dt','max')).assign(candidate_rate=lambda z:z.candidate_rows/z.rows).reset_index().astype({'first':str,'last':str}).to_dict('records'),
             'candidate_by_module_split':sample.groupby(['split','module(equipment)']).weak_candidate.agg(['size','sum']).reset_index().to_dict('records'),
             'weak_rule':'phase voltage outside [205,225] V; phase PF outside [80,100]%; negative current/power or operation outside {0,1}',
             'label_limit':'Weak electrical-quality candidate, not verified fault or true injected-anomaly label',
             'elapsed_seconds':time.monotonic()-start}
    dump(summary,'sample_summary.json')
    print(f'extraction complete sample={len(sample):,}',flush=True)
    return sample

def load_sample():return pd.read_csv(SAMPLE,parse_dates=['dt']) if SAMPLE.exists() else extract_sample()

def attach_labels(d,path):
    if not path:
        d['target']=d.weak_candidate
        return d,{'type':'weak_electrical_quality_candidate','ground_truth_fault_labels_available':False,
                  'scope':'Current 5-second observation, not future failure prediction'}
    labels=pd.read_csv(path)
    required=['module(equipment)','localtime','label']
    if any(c not in labels for c in required):raise ValueError('Actual label CSV requires '+', '.join(required))
    if not labels.label.dropna().isin([0,1]).all():raise ValueError('Actual labels must be binary 0/1')
    text=labels.localtime.astype(str)
    labels['dt']=pd.to_datetime(text,format='%Y%m%d%H%M%S') if text.str.fullmatch(r'\d{14}').all() else pd.to_datetime(text)
    if labels.duplicated(['module(equipment)','dt']).any():raise ValueError('Actual label keys must be unique')
    d=d.merge(labels[['module(equipment)','dt','label']],on=['module(equipment)','dt'],how='left',validate='one_to_one').rename(columns={'label':'target'})
    return d,{'type':'provided_binary_labels','source':str(Path(path).resolve()),
              'unknown_label_rows_excluded_from_classifier_evaluation':int(d.target.isna().sum()),
              'label_semantics_need_user_confirmation':True}

def run_models(d,label_info,deps):
    for path in deps:sys.path.insert(0,str(Path(path)))
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,precision_recall_fscore_support,confusion_matrix
    import xgboost as xgb
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset,DataLoader
    import pickle
    torch.set_num_threads(4);torch.manual_seed(SEED)
    train=d[d.split=='train'].copy();valid=d[d.split=='validation'].copy();test=d[d.split=='test'].copy()
    for split in [train,valid,test]:
        if split.target.dropna().nunique()!=2:raise ValueError('Every evaluation split needs positive and negative labeled observations; inspect supplied-label coverage')

    def score(y,prob,threshold):
        y=np.asarray(y,dtype=int);prob=np.asarray(prob,float);pred=(prob>=threshold).astype(int)
        p,r,f,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
        cm=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=map(int,cm.ravel())
        return {'rows':len(y),'positive_rows':int(y.sum()),'positive_rate':float(y.mean()),'auroc':float(roc_auc_score(y,prob)),
                'average_precision':float(average_precision_score(y,prob)),'precision':float(p),'recall':float(r),'f1':float(f),
                'threshold':float(threshold),'true_negative':tn,'false_positive':fp,'false_negative':fn,'true_positive':tp,
                'false_positive_rate':float(fp/(tn+fp)) if tn+fp else 0.,'alarm_rate':float(pred.mean())}

    results={};pred=test[['module(equipment)','dt','weak_candidate','voltage_candidate','pf_candidate','target']].copy()
    labelled={name:z[z.target.notna()].copy() for name,z in [('train',train),('validation',valid),('test',test)]}
    for tag,features in [('xgb_rule_distillation',SENSORS),('xgb_rule_sensor_excluded_transfer',TRANSFER)]:
        tr,va,te=[labelled[k] for k in ['train','validation','test']]
        ratio=float((tr.target==0).sum()/(tr.target==1).sum())
        model=xgb.XGBClassifier(n_estimators=400,max_depth=3,learning_rate=.05,subsample=.9,colsample_bytree=.9,
                                min_child_weight=4,reg_lambda=2,scale_pos_weight=ratio,objective='binary:logistic',
                                eval_metric='aucpr',tree_method='hist',n_jobs=4,random_state=SEED,early_stopping_rounds=35)
        model.fit(tr[features],tr.target.astype(int),eval_set=[(va[features],va.target.astype(int))],verbose=False)
        vp=model.predict_proba(va[features])[:,1].astype(np.float64)
        threshold=float(np.nextafter(float(np.quantile(vp[(va.target==0).to_numpy()],.995)),np.inf))
        allprob=model.predict_proba(test[features])[:,1].astype(np.float64)
        model.save_model(str(OUT/f'{tag}.json'))
        results[tag]={'purpose':'Reproduces label-defining sensor rule; high scores are circular and not fault evidence' if tag.endswith('distillation') and label_info['type'].startswith('weak') else 'Transfer test excluding voltage, PF, operation, timestamp, counter and module ID',
                      'features':features,'label_info':label_info,'threshold_selection':'99.5th percentile of validation weak-normal probability scores (strictly greater than the quantile), frozen for April; positive validation labels not used to optimize alarm threshold',
                      'best_iteration':int(model.best_iteration),'scale_pos_weight':ratio,
                      'validation':score(va.target,vp,threshold),'test':score(te.target,allprob[test.target.notna()],threshold),
                      'feature_importance':{f:float(v) for f,v in zip(features,model.feature_importances_)}}
        pred[f'{tag}_probability']=allprob;pred[f'{tag}_flag']=(allprob>=threshold).astype(int)
        print(f'{tag} test={results[tag]["test"]}',flush=True)

    # One wholly unsupervised fit, and a separate normal-proxy semi-supervised fit.
    for tag,normal_proxy in [('ae_unsupervised_all_training',False),('ae_weak_normal_proxy',True)]:
        tr=train[train.target==0] if normal_proxy else train
        va_fit=valid[valid.target==0] if normal_proxy else valid
        scaler=RobustScaler(quantile_range=(25,75));xtr=scaler.fit_transform(tr[SENSORS]).astype(np.float32)
        xva=scaler.transform(valid[SENSORS]).astype(np.float32);xte=scaler.transform(test[SENSORS]).astype(np.float32)
        xvaf=scaler.transform(va_fit[SENSORS]).astype(np.float32)
        torch.manual_seed(SEED);model=AE(len(SENSORS));optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5)
        loader=DataLoader(TensorDataset(torch.from_numpy(xtr)),batch_size=1024,shuffle=True,generator=torch.Generator().manual_seed(SEED))
        best=float('inf');state=None;stale=0;history=[]
        for epoch in range(1,51):
            model.train();total=0.
            for (batch,) in loader:
                optimizer.zero_grad();loss=((model(batch)-batch)**2).mean();loss.backward();optimizer.step();total+=loss.item()*len(batch)
            model.eval()
            with torch.no_grad():validation_loss=float(((model(torch.from_numpy(xvaf))-torch.from_numpy(xvaf))**2).mean())
            history.append({'epoch':epoch,'train_mse':total/len(xtr),'validation_mse':validation_loss})
            if validation_loss<best-1e-6:best=validation_loss;state={k:v.detach().clone() for k,v in model.state_dict().items()};stale=0
            else:stale+=1
            if epoch%10==0:print(f'{tag} epoch={epoch} val_loss={validation_loss:.6f}',flush=True)
            if stale>=8:break
        model.load_state_dict(state);model.eval()
        with torch.no_grad():
            va_err=(model(torch.from_numpy(xva))-torch.from_numpy(xva)).pow(2).numpy()
            te_err=(model(torch.from_numpy(xte))-torch.from_numpy(xte)).pow(2).numpy()
        va_score=va_err.mean(axis=1).astype(np.float64);te_score=te_err.mean(axis=1).astype(np.float64)
        threshold_pool=va_score[(valid.target==0).to_numpy()]
        threshold_note='99.5th percentile of validation weak-normal proxy scores, frozen for April; threshold calibration is semi-supervised, including when the AE fit itself is unsupervised'
        threshold=float(np.quantile(threshold_pool,.995))
        torch.save({'state_dict':state,'features':SENSORS,'architecture':[len(SENSORS),32,12,4,12,32,len(SENSORS)],'threshold':threshold},OUT/f'{tag}.pt')
        with (OUT/f'{tag}_scaler.pkl').open('wb') as f:pickle.dump(scaler,f)
        results[tag]={'learning_type':'semi-supervised training and calibration with weak-normal proxy' if normal_proxy else 'unsupervised AE fit on all training observations; semi-supervised weak-normal threshold calibration',
                      'features':SENSORS,'excluded_features':['timestamp','localtime','accumActiveEnergy','module(equipment)','operation'],
                      'fit_train_rows':len(tr),'scaler_fit':'Training only; RobustScaler median/IQR',
                      'architecture':[len(SENSORS),32,12,4,12,32,len(SENSORS)],'loss':'Mean squared standardized reconstruction error',
                      'best_validation_reconstruction_mse':best,'epochs_run':len(history),'history':history,
                      'threshold_selection':threshold_note,'label_info':label_info,
                      'validation_weak_proxy_agreement':score(valid.loc[valid.target.notna(),'target'],va_score[valid.target.notna()],threshold),
                      'test_weak_proxy_agreement':score(test.loc[test.target.notna(),'target'],te_score[test.target.notna()],threshold),
                      'metric_limit':'Agreement with a weak sensor rule does not estimate verified fault precision/recall',
                      'test_mean_error_by_feature':{f:float(v) for f,v in zip(SENSORS,te_err.mean(axis=0))}}
        pred[f'{tag}_score']=te_score;pred[f'{tag}_flag']=(te_score>=threshold).astype(int)
        for feature,errs in zip(SENSORS,te_err.T):pred[f'{tag}_error_{feature}']=errs
        if not normal_proxy:
            pure_threshold=float(np.quantile(va_score,.995))
            results[tag]['fully_unsupervised_threshold_secondary']={'selection':'99.5th percentile of all validation scores; labels used only for post-hoc agreement assessment',
                 'threshold':pure_threshold,'test_weak_proxy_agreement':score(test.loc[test.target.notna(),'target'],te_score[test.target.notna()],pure_threshold)}
        print(f'{tag} test={results[tag]["test_weak_proxy_agreement"]}',flush=True)
    pred.to_csv(OUT/'test_observation_predictions.csv.gz',index=False,compression='gzip',float_format='%.17g')
    dump(results,'diagnosis_model_metrics.json')
    versions={'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,'xgboost':xgb.__version__}
    import sklearn;versions['sklearn']=sklearn.__version__
    report={'title':'전기품질 이상 후보 분류와 재구성 이상탐지','seed':SEED,'versions':versions,
            'label_info':label_info,'sample_summary':json.loads((OUT/'sample_summary.json').read_text(encoding='utf-8')),
            'protocol':'Train Dec 2024–Feb 2025; validation March; final untouched test April 1–29. Current 5-second observation diagnosis. No May electricity target used.',
            'research_reference_status':'Reference rationale is documented in the root modeling report; the tabular MLP AE is an implementation of reconstruction anomaly scoring, not a claimed state-of-the-art fault system.',
            'results':results,'limitations':['No verified real-fault or injected-anomaly labels supplied','Rule distillation can trivially learn the rule and is not independent diagnostic evidence','Transfer model excludes rule variables; failure to generalize is informative','Balanced per-hour subsample preserves equal inclusion probability but can miss rare transient events','Monthly block splits test time generalization on known modules, not unseen equipment','AE fitting on weak-normal proxy is semi-supervised and can still include unknown faults','Alarm counts concern sampled observations, not full-data incident counts']}
    dump(report,'diagnosis_report.json')
    return report,pred

def figures(report,pred,deps):
    for path in deps:sys.path.insert(0,str(Path(path)))
    sys.path.insert(0,str(ROOT/'output'/'analysis'/'.plot_deps'))
    os.environ['MPLCONFIGDIR']=str(OUT/'tmp'/'mpl')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import FuncFormatter,NullFormatter
    from sklearn.metrics import precision_recall_curve
    font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'figure.facecolor':'#f7f9fc','axes.spines.top':False,'axes.spines.right':False})
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    colors=['#007f86','#dc6b26','#3467b7','#8b5ca6']
    results=report['results'];names=['규칙 모사 XGB','규칙 센서 제외 XGB','전체 학습 AE','약한 정상 proxy AE']
    fig,axs=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    testmetrics=[r.get('test',r.get('test_weak_proxy_agreement')) for r in results.values()]
    xx=np.arange(4)
    axs[0,0].bar(xx-.18,[r['average_precision'] for r in testmetrics],width=.35,label='Average precision',color='#007f86')
    axs[0,0].bar(xx+.18,[r['f1'] for r in testmetrics],width=.35,label='고정 임계값 F1',color='#dc6b26')
    axs[0,0].set_xticks(xx,names,rotation=12);axs[0,0].set_ylim(0,1.07);axs[0,0].legend();axs[0,0].set_title('4월 표본의 약한 라벨 일치도 · 실제 고장 성능 아님')
    for name,(tag,r),color in zip(names,results.items(),colors):
        col=f'{tag}_probability' if tag.startswith('xgb') else f'{tag}_score'
        known=pred.target.notna();p,rec,_=precision_recall_curve(pred.loc[known,'target'],pred.loc[known,col])
        axs[0,1].plot(rec,p,label=name,color=color)
    axs[0,1].axhline(pred.target.dropna().mean(),color='#61758e',ls=':',label='후보 비율 기준')
    axs[0,1].set_xlabel('재현율');axs[0,1].set_ylabel('정밀도');axs[0,1].set_title('Precision–recall · 규칙 포함 모사의 순환성 유의');axs[0,1].legend(fontsize=8)
    for tag,ax in zip(['ae_unsupervised_all_training','ae_weak_normal_proxy'],axs[1]):
        scorecol=f'{tag}_score';met=results[tag]['test_weak_proxy_agreement'];normal=pred.loc[pred.target==0,scorecol];candidate=pred.loc[pred.target==1,scorecol]
        lo=max(min(normal.min(),candidate.min()),1e-9);hi=max(normal.max(),candidate.max());bins=np.geomspace(lo,hi,60)
        ax.hist(normal,bins=bins,density=True,alpha=.55,color='#3467b7',label='규칙 정상');ax.hist(candidate,bins=bins,density=True,histtype='step',lw=2,color='#dc6b26',label='규칙 후보')
        ax.axvline(met['threshold'],color='#172a43',ls='--',label='3월 고정 임계값');ax.set_xscale('log')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'));ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title('전체 학습 AE · 임계값 준지도' if tag.startswith('ae_unsupervised') else '약한 정상 proxy AE · 준지도');ax.set_xlabel('센서 재구성 오차 · 로그 축');ax.set_ylabel('정규화 밀도');ax.legend(fontsize=8)
    fig.suptitle('현재 전기품질 이상 후보 진단 실험',fontsize=18,fontweight='bold')
    fig.supxlabel('모든 관측에서 동일 확률로 표본 추출 · Dec–Feb 학습 / 3월 검증 / 4월 테스트 · 미확인 고장 라벨을 대신한 약한 후보 규칙',fontsize=10)
    fig.savefig(figdir/'diagnosis_overview.png',dpi=150,bbox_inches='tight');fig.savefig(figdir/'diagnosis_overview.svg',bbox_inches='tight');plt.close(fig)

def html_report(report):
    import base64,html
    results=report['results'];names={'xgb_rule_distillation':'규칙 모사 XGB','xgb_rule_sensor_excluded_transfer':'규칙 센서 제외 XGB',
                                   'ae_unsupervised_all_training':'전체 학습 AE','ae_weak_normal_proxy':'약한 정상 proxy AE'}
    image=base64.b64encode((OUT/'figures'/'diagnosis_overview.png').read_bytes()).decode()
    rows=''
    for tag,r in results.items():
        met=r.get('test',r.get('test_weak_proxy_agreement'))
        rows+=f'<tr><td>{names[tag]}</td><td>{met["average_precision"]:.4f}</td><td>{met["auroc"]:.4f}</td><td>{met["precision"]:.4f}</td><td>{met["recall"]:.4f}</td><td>{met["f1"]:.4f}</td><td>{met["false_positive_rate"]*100:.3f}%</td><td>{met["true_positive"]}/{met["positive_rows"]}</td></tr>'
    sample=report['sample_summary'];splits=''.join(f'<tr><td>{r["split"]}</td><td>{r["first"][:10]}–{r["last"][:10]}</td><td>{r["rows"]:,}</td><td>{r["candidate_rows"]}</td><td>{r["candidate_rate"]*100:.3f}%</td></tr>' for r in sample['splits'])
    details=''
    for tag,r in results.items():
        met=r.get('test',r.get('test_weak_proxy_agreement'))
        details+=f'<h3>{names[tag]}</h3><p>특성: {html.escape(", ".join(r["features"]))}</p><p>검증에서 고정한 임계값: {met["threshold"]:.6g}. 테스트 경보 {met["true_positive"]+met["false_positive"]}행 중 약한 규칙 후보 {met["true_positive"]}행, 규칙 정상 {met["false_positive"]}행이며 후보 누락은 {met["false_negative"]}행입니다.</p>'
        if tag.startswith('ae'):
            details+=f'<p>학습 유형: {html.escape(r["learning_type"])}. 학습 {r["fit_train_rows"]:,}행, {r["epochs_run"]}에포크. RobustScaler는 학습 구간으로만 적합했습니다. 재구성 모델은 14→32→12→4→12→32→14 구조입니다.</p>'
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>전기품질 이상 후보 진단 실험</title><style>body{{font-family:"Malgun Gothic",sans-serif;background:#eef3f8;color:#172a43;line-height:1.7;margin:0}}main{{max-width:1160px;margin:auto;padding:30px 24px}}section,header{{background:#fff;border:1px solid #dbe4ee;border-radius:14px;padding:28px;margin:20px 0}}header{{background:#152e47;color:white}}h1{{font-size:28px;margin:0 0 15px}}h2{{font-size:21px;margin:0 0 14px}}h3{{font-size:17px}}.note{{background:#fff4e9;border-left:4px solid #dc6b26;padding:16px 20px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:10px 8px;text-align:right;border-bottom:1px solid #dbe4ee}}td:first-child,th:first-child{{text-align:left}}th{{background:#f0f5fa}}img{{width:100%;height:auto}}.tablewrap{{overflow:auto}}small{{color:#61758e}}code{{background:#eef3f8;padding:2px 5px}}@media(max-width:700px){{main{{padding:15px 10px}}section,header{{padding:20px 15px}}}}</style><main><header><h1>현재 전기품질 이상 후보 진단 실험</h1><p>XGBoost 분류, 규칙 변수 제외 전달 테스트, 재구성 Autoencoder를 시간 순서로 비교했습니다.</p></header>
<section><h2>실제 고장 라벨이 없는 상태의 실험</h2><div class="note"><strong>아래 점수는 고장 진단 정확도가 아닙니다.</strong> 가이드는 삽입 이상치가 있다고 설명하지만 위치·정답 라벨은 제공하지 않습니다. 상전압 205–225V 밖 / 상역률 80–100% 밖 등의 탐색 규칙을 약한 후보 라벨로 사용했습니다. 이는 고장·삽입 이상치의 확정 판정이 아닙니다.</div><p>각 관측의 센서값으로 현재 이상 후보 여부를 판별하며 장래 고장 발생을 예측하는 작업이 아닙니다. 5월 전력량 정답은 이 실험에 사용하지 않았습니다.</p></section>
<section><h2>균등 표본과 시간 순서 검증</h2><p>원시 {sample["source_rows_read"]:,}행을 한 번 읽고, 각 모듈×완전시간의 720개 5초 슬롯에서 무작위 4개를 고정 seed로 뽑았습니다. 총 <strong>{sample["sample_rows"]:,}행</strong>이며 이상 후보를 추가 추출하지 않았습니다. 모든 완전시간 관측의 포함 확률은 4/720으로 동일합니다. 4/30의 부분시간 1행은 제외했고 localtime을 사용했습니다.</p><div class="tablewrap"><table><tr><th>구간</th><th>기간</th><th>표본 행</th><th>약한 후보</th><th>후보 비율</th></tr>{splits}</table></div><p>3월 검증의 약한 정상 관측 점수에서 99.5백분위 임계값을 계산한 뒤 4월 테스트에는 고정했습니다. 테스트 정답으로 임계값을 조정하지 않았습니다. 약한 정상 집합을 이용한 임계값 보정은 준지도 절차입니다.</p></section>
<section><h2>4월 테스트의 약한 규칙 일치도</h2><div class="tablewrap"><table><tr><th>모델</th><th>AP</th><th>AUROC</th><th>정밀도</th><th>재현율</th><th>F1</th><th>규칙 정상 오경보율</th><th>찾은 후보/전체 후보</th></tr>{rows}</table></div><p><strong>규칙 모사 XGB</strong>는 라벨을 만드는 전압·역률을 입력으로 사용하므로 높은 점수가 예상됩니다. 원래 규칙을 재현하는 기준선이며 독립적인 고장 진단 능력을 입증하지 않습니다. <strong>규칙 센서 제외 XGB</strong>는 전압·역률·가동·시간·누적계·설비 ID를 제외하여 다른 센서에서 후보를 전달할 수 있는지 확인합니다.</p><img src="data:image/png;base64,{image}" alt="진단 모델의 4월 약한 규칙 일치도와 AE 재구성 오차 분포"></section>
<section><h2>재구성 오차와 학습 가정</h2><p>전체 학습 AE는 12–2월의 모든 관측을 사용해 scaler와 재구성 모델을 비지도 학습했습니다. 약한 정상 proxy AE는 같은 학습 구간에서 규칙 후보를 제외하고 학습한 준지도 실험입니다. 두 모델의 기본 경보 임계값은 3월의 약한 정상으로 보정하므로 경보 시스템 전체를 완전한 비지도라고 부를 수 없습니다.</p><p>전체 학습 AE에 대해서는 모든 3월 관측으로 임계값을 계산하는 완전 비지도 대안도 JSON에 별도로 기록했습니다. 이는 양성 라벨로 임계값을 선택하지 않는 비교이며 점수 확인 후 모델 선택에 사용하는 값이 아닙니다.</p>{details}</section>
<section><h2>프로젝트 적용 판단</h2><ul><li>규칙 모사의 높은 점수는 규칙을 코드로 직접 구현하는 방식과 함께 평가해야 합니다. 동일 규칙이 목적이라면 복잡한 분류기의 실익이 제한적입니다.</li><li>AE 점수는 약한 센서 규칙과 일치하지 않는 지점도 탐색하게 합니다. 그 점을 곧바로 실제 고장이나 오탐으로 확정하지 말고 원시 센서 시계열과 현장 정보를 검토해야 합니다.</li><li>샘플 경보 행 수는 전체 원시 데이터의 고장 사건 수가 아닙니다. 실제 운영에서는 지속시간·설비·유형별 묶음과 재알림 제한이 필요합니다.</li><li>다음 검증은 점검·교체·정지 이력과 삽입 이상치 위치를 확보하여 실제 고장/정상 라벨을 만들고, 시간 순서 테스트와 미관측 설비 테스트를 분리하는 것입니다.</li></ul><p>표본은 드문 짧은 이상을 놓칠 수 있고, 알려진 13개 설비에 대한 월별 검증은 새로운 설비로의 일반화를 보장하지 않습니다. 약한 정상 proxy에도 미확인 실제 고장이 남아 있을 수 있습니다.</p></section><footer><small>재현 코드: diagnosis_experiment.py · 상세 지표: diagnosis_model_metrics.json · 학습 모델/스케일러와 테스트 관측 예측 CSV 저장 · 원본 데이터 변경 없음</small></footer></main></html>'''
    (OUT/'diagnosis_report.html').write_text(page,encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--extract-only',action='store_true');ap.add_argument('--label-file');ap.add_argument('--deps',action='append',default=[])
    args=ap.parse_args();d=load_sample()
    if not args.extract_only:
        d,label_info=attach_labels(d,args.label_file);report,pred=run_models(d,label_info,args.deps);figures(report,pred,args.deps);html_report(report)
