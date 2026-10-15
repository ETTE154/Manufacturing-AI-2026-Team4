"""Chronological context calibration + IF/AE + separate synthetic classifiers."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import json,time,hashlib,copy
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.ensemble import IsolationForest,RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,average_precision_score,precision_recall_fscore_support
from xgboost import XGBClassifier
from threadpoolctl import threadpool_limits
from detector import OUT,RAW,TYPE_IDS,CLASS_NAMES,FEATURE_NAMES,prepare,fit_baseline,features,rule_scores,AE,ae_scores,Detector
ROOT=OUT.parents[2];SEED=20261014
SAMPLE=ROOT/'output/modeling/diagnosis/uniform_raw_sample.csv.gz'
torch.set_num_threads(4);threadpool_limits(limits=4)

def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
    if isinstance(v,(Path,pd.Timestamp)):return str(v)
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(np.bool_,)):return bool(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    return v
def write(path,v):Path(path).write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def synthesize(frame,n,seed):
    """Independent base-row sampling per chronological split; seven known labels.

    Normal class is unmodified-reference, not a verified healthy machine label.
    """
    rng=np.random.default_rng(seed);base=frame.sample(n=min(n,len(frame)),random_state=seed).copy();base_ids=base.index.to_numpy();parts=[];labels=[];manifests=[]
    for label in range(7):
        f=base.copy();severity=rng.uniform(1.5,2.8,len(f));I=f[['currentR','currentS','currentT']].to_numpy(dtype=float,copy=True)
        if label==1:
            choice=rng.integers(0,3,len(f));oldmean=I.mean(axis=1)
            I[np.arange(len(f)),choice]=np.maximum(I[np.arange(len(f)),choice]*severity,35+8*severity)
            f[['currentR','currentS','currentT']]=I;f['activePower']=f.activePower.to_numpy()*I.mean(axis=1)/oldmean
        elif label==2:
            f['reactivePowerLagging']=np.maximum(f.reactivePowerLagging.to_numpy()*severity*2,1200*severity)
            f[['powerFactorR','powerFactorS','powerFactorT']]*=rng.uniform(.9,1.,(len(f),1))
        elif label==3:
            pf=f[['powerFactorR','powerFactorS','powerFactorT']].to_numpy(dtype=float,copy=True);choice=rng.integers(0,3,len(f))
            pf[np.arange(len(f)),choice]*=rng.uniform(.4,.8,len(f));f[['powerFactorR','powerFactorS','powerFactorT']]=pf
        elif label==4:
            v=f[['voltageR','voltageS','voltageT']].to_numpy(dtype=float,copy=True);choice=rng.integers(0,3,len(f));v[np.arange(len(f)),choice]*=rng.choice([.82,1.18],len(f))
            f[['voltageR','voltageS','voltageT']]=v;f['voltageRS']*=rng.uniform(.75,1.25,len(f));f['activePower']*=rng.uniform(.7,1.3,len(f))
        elif label==5:
            shifted=f.dt.dt.normalize()+pd.to_timedelta(rng.integers(0,6,len(f)),unit='h')+pd.to_timedelta(f.dt.dt.minute,unit='m')+pd.to_timedelta(f.dt.dt.second,unit='s')
            f['localtime']=shifted.dt.strftime('%Y%m%d%H%M%S');f['dt']=shifted
            f['activePower']*=severity;f[['currentR','currentS','currentT']]=I*severity[:,None]
        elif label==6:
            f[['currentR','currentS','currentT']]=I*severity[:,None];f['activePower']*=rng.uniform(.98,1.02,len(f))
        f['synthetic_label']=label;f['source_kind']='synthetic_simulation';parts.append(f);labels.extend([label]*len(f))
        manifests.extend([{'base_sample_index':int(i),'original_time':str(t),'synthetic_label':CLASS_NAMES[label],'seed':seed} for i,t in zip(base_ids,base.dt)])
    return pd.concat(parts,ignore_index=True),np.asarray(labels,dtype=int),manifests

def binary_report(y,score,threshold):
    alarm=np.asarray(score)>threshold;precision,recall,f1,_=precision_recall_fscore_support(y,alarm,average='binary',zero_division=0)
    return {'count':len(y),'positive_count':int(np.sum(y)),'threshold':threshold,'precision':float(precision),'recall':float(recall),'f1':float(f1),
            'roc_auc':float(roc_auc_score(y,score)),'average_precision':float(average_precision_score(y,score)),
            'meaning':'Only explicit six-form synthetic transformations versus unmodified reference rows; not real-fault accuracy.'}

def main():
    OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic();torch.manual_seed(SEED);np.random.seed(SEED)
    frame=prepare(pd.read_csv(SAMPLE,dtype={'module(equipment)':'string','localtime':'string'},float_precision='round_trip'))
    train=frame.loc[frame.dt<pd.Timestamp('2025-01-01')].copy();val=frame.loc[frame.dt.between('2025-01-01','2025-01-31 23:59:59')].copy();test=frame.loc[frame.dt>=pd.Timestamp('2025-02-01')].copy()
    assert max(train.dt)<min(val.dt) and max(val.dt)<min(test.dt)
    baseline=fit_baseline(train);Xt,_=features(train,baseline);Xv,_=features(val,baseline);Xtest,_=features(test,baseline)
    rules_val=rule_scores(Xv);thresholds={'rules':{}}
    for m in val.module_id.unique():
        mask=val.module_id.eq(m).to_numpy();thresholds['rules'][m]={}
        for col,t in enumerate(TYPE_IDS):
            use=mask & (Xv[:,-1].eq(1) if isinstance(Xv[:,-1],pd.Series) else Xv[:,-1]==1) if t=='night_operation' else mask
            thresholds['rules'][m][t]=max(float(np.quantile(rules_val[use,col],.999)),1e-6)
    forest=IsolationForest(n_estimators=192,max_samples=1024,contamination='auto',random_state=SEED,n_jobs=4).fit(Xt)
    iso_val=-forest.score_samples(Xv).astype(np.float64);thresholds['iforest']=float(np.quantile(iso_val,.999))
    model=AE(len(FEATURE_NAMES));optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    t=torch.from_numpy(Xt.astype(np.float32));v=torch.from_numpy(Xv.astype(np.float32));best=float('inf');best_state=None;stale=0;fit_log=[]
    for epoch in range(1,2001):
        model.train();order=torch.randperm(len(t));total=0.
        for begin in range(0,len(t),1024):
            batch=t[order[begin:begin+1024]];optimizer.zero_grad();loss=(model(batch)-batch).pow(2).mean();loss.backward();optimizer.step();total+=float(loss.item())*len(batch)
        model.eval()
        with torch.inference_mode():vl=float((model(v)-v).pow(2).mean())
        fit_log.append({'epoch':epoch,'train_loss':total/len(t),'January_loss':vl})
        if vl<best-1e-4:best=vl;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
        else:stale+=1
        if epoch%25==0:print(f'AEepoch{epoch}, Januaryloss{vl:.6f}, bestepoch{best_epoch}, elapsed{time.monotonic()-started:.1f}s',flush=True)
        if stale>=20:break
    model.load_state_dict(best_state);model.eval();ae_val=ae_scores(model,Xv);thresholds['ae']=float(np.quantile(ae_val,.999))
    print(f'AE earlystopped epoch{epoch}, best{best_epoch}; preparing independent synthetic sets',flush=True)
    syntr,ytr,manifesttr=synthesize(train,3000,SEED);synv,yv,manifestv=synthesize(val,1500,SEED+1);synt,ytest,manifesttest=synthesize(test,2000,SEED+2)
    Xtr,_=features(syntr,baseline);Xsv,_=features(synv,baseline);Xst,_=features(synt,baseline)
    rf=RandomForestClassifier(n_estimators=300,max_depth=14,min_samples_leaf=4,class_weight='balanced',random_state=SEED,n_jobs=4).fit(Xtr,ytr)
    xgb=XGBClassifier(n_estimators=4000,max_depth=4,learning_rate=.05,min_child_weight=10,reg_lambda=10.,subsample=.9,colsample_bytree=.9,
             tree_method='hist',objective='multi:softprob',num_class=7,eval_metric='mlogloss',early_stopping_rounds=60,random_state=SEED,n_jobs=4)
    xgb.fit(Xtr,ytr,eval_set=[(Xsv,yv)],verbose=False)
    classifiers={'random_forest':rf,'xgboost':xgb};classification={}
    for name,obj in classifiers.items():
        vpred=obj.predict(Xsv);tpred=obj.predict(Xst)
        classification[name]={'validation':classification_report(yv,vpred,target_names=CLASS_NAMES,output_dict=True,zero_division=0),
             'test':classification_report(ytest,tpred,target_names=CLASS_NAMES,output_dict=True,zero_division=0),
             'test_confusion_matrix':confusion_matrix(ytest,tpred,labels=np.arange(7)).tolist()}
    selected=max(classification,key=lambda n:classification[n]['validation']['macro avg']['f1-score'])
    saved={'baseline':baseline,'feature_names':FEATURE_NAMES,'thresholds':thresholds,'iforest':forest,'classifiers':classifiers,
           'selected_classifier':selected,'class_names':CLASS_NAMES,'seed':SEED,'threshold_fit_end':str(val.dt.max())}
    torch.save({'state_dict':model.state_dict(),'feature_names':FEATURE_NAMES,'best_epoch':best_epoch},OUT/'ae.pt');joblib.dump(saved,OUT/'models.joblib',compress=3)
    xgb.save_model(OUT/'synthetic_xgb.json')
    write(OUT/'baseline.json',baseline);write(OUT/'thresholds.json',thresholds)
    pd.DataFrame(fit_log).to_csv(OUT/'ae_training_curve.csv',index=False,float_format='%.17g')
    pd.DataFrame(manifesttr+manifestv+manifesttest).to_csv(OUT/'synthetic_manifest.csv.gz',index=False,compression='gzip')
    positive=ytest>0;synrules=rule_scores(Xst)
    syn_thresholds=np.array([[thresholds['rules'][str(m)][t] for t in TYPE_IDS] for m in synt.module_id])
    normalized_rules=np.max(synrules/syn_thresholds,axis=1)
    binary={'rules':binary_report(positive,normalized_rules,1.),'isolation_forest':binary_report(positive,-forest.score_samples(Xst),thresholds['iforest']),
            'autoencoder':binary_report(positive,ae_scores(model,Xst),thresholds['ae'])}
    loaded=Detector();probe=val.iloc[:1000].copy();directX,_=features(probe,baseline);rs=loaded.scores(probe,True)
    ae_diff=float(np.max(np.abs(rs['ae_score']-ae_scores(model,directX))));if_diff=float(np.max(np.abs(rs['iforest_score']+forest.score_samples(directX))))
    class_diff=float(np.max(np.abs(rs['class_probs']-classifiers[selected].predict_proba(directX))));assert max(ae_diff,if_diff,class_diff)<=1e-8
    original_test_scores=loaded.scores(test,True)
    rates={'rule_any_fraction':float(original_test_scores['rule_flags'].any(axis=1).mean()),'AE_alarm_fraction':float(original_test_scores['ae_alarm'].mean()),
           'IF_alarm_fraction':float(original_test_scores['iforest_alarm'].mean()),'meaning':'Alarm rates on unlabeled original sample, not false-positive rates.'}
    result={'status':'trained_pending_full_scan','source_sample':str(SAMPLE),'sample_sha256':sha(SAMPLE),'sample_rows':len(frame),
        'chronological_splits':{'baseline_training':{'start':str(train.dt.min()),'end':str(train.dt.max()),'rows':len(train)},
           'threshold_and_AE_validation':{'start':str(val.dt.min()),'end':str(val.dt.max()),'rows':len(val)},
           'unlabeled_original_and_synthetic_test_bases':{'start':str(test.dt.min()),'end':str(test.dt.max()),'rows':len(test)}},
        'six_types':dict(zip(TYPE_IDS,['부하 불균형','무효전력 급증','역률 급감','비정상 패턴','야간 이상가동','전류만 증가'])),
        'rule_definition':'Module-specific empirical robust deviations plus completed January-tail thresholds; physical ratio is calibrated from December. Night requires unusual load relative to its module/hour baseline, not operation==1.',
        'threshold_design':'99.9th percentile of January original scores per module/rule; night quantile uses night rows only. Unknown fault contamination can influence baselines/thresholds.',
        'autoencoder':{'architecture':[len(FEATURE_NAMES),32,12,32,len(FEATURE_NAMES)],'epochs_run':epoch,'best_epoch':best_epoch,'January_best_loss':best,
             'stop':'January validation plateau patience20,min_delta1e-4; max2000 safety guard','guard_reached':epoch==2000},
        'synthetic_classification':classification,'selected_classifier':selected,'selected_on':'January synthetic validation macroF1 only',
        'synthetic_detection':binary,'synthetic_design':{'train_bases':3000,'validation_bases':1500,'test_bases':2000,'classes':CLASS_NAMES,
             'source_base_split_disjoint':True,'same_base_repeated_across7classes':True,'severity':'1.5..2.8 multiplicative broad transformations; classifier transfer to actual unknown injected forms is unverified.'},
        'original_test_alarm_rates':rates,'model_files':{'bundle':str(OUT/'models.joblib'),'ae':str(OUT/'ae.pt'),'xgb_native':str(OUT/'synthetic_xgb.json')},
        'reload':{'ae_max_absdiff':ae_diff,'IF_max_absdiff':if_diff,'classifier_max_absdiff':class_diff},
        'model_sha256':{n:sha(OUT/n) for n in ['models.joblib','ae.pt','synthetic_xgb.json']},
        'limitations':['No actual fault type/timestamps; natural candidate numbers are not ground truth.','Synthetic precision/recall does not estimate real precision/recall.',
          'Background current imbalance is large; use module-relative context and manual review.','Raw power/Q numeric units and meter hierarchy remain uncertain.',
          'Autoencoder and IF are pointwise multivariate detectors, not a validated temporal fault classifier.','Unmodified reference class is not verified healthy.'],
        'references':[{'title':'IsolationForest API','url':'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html'},
                     {'title':'RandomForestClassifier API','url':'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html'},
                     {'title':'XGBClassifier API','url':'https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier'}],
        'prior_models_policy':'Reviewed existing diagnosis/anomaly code, but their weak electrical-quality binary labels differ from this6form task; kept all old artifacts unchanged.',
        'runtime_seconds':time.monotonic()-started}
    write(OUT/'results.json',result);print(json.dumps({'status':'trained','selected_classifier':selected,'AEbest_epoch':best_epoch,'synthetic_detection':binary,'runtime_seconds':result['runtime_seconds']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
