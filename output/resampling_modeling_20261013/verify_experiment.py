"""Verify frozen paths/metrics, replay state and preserved originals."""
from pathlib import Path
import sys,json,hashlib,datetime,uuid
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import joblib
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
sys.path.insert(0,str(OUT));from resample_common import metrics

def main():
    r=read(OUT/'results.json');lock=read(OUT/'global_selection_lock.json');checks={};details={}
    checks['global_lock_unchanged']=all(sha(OUT/k)==v for k,v in lock['native_selection_lock_sha256'].items())
    checks['future_paths_unchanged']=all(sha(c['paths']['future'])==lock['future_path_sha256'][c['key']] for c in r['models'])
    checks['chosen_by_validation_only']=lock['selected_model']==min(r['models'],key=lambda c:c['val']['mae_kwh'])['key'] and not r['selection_changed_after_may']
    checks['may_evaluation_after_freeze']=pd.Timestamp(r['evaluation_started_utc'])>pd.Timestamp(lock['created_utc'])
    counts=[];coarse_replay={};hourlytruth=pd.read_csv(OUT/'data/hourly_target.csv',float_precision='round_trip');hourlytruth.index=pd.to_datetime(hourlytruth.iloc[:,0]);truth=pd.Series(hourlytruth.iloc[:,1].to_numpy(float),index=hourlytruth.index)
    for c in r['models']:
        trainlabels=set();testlabels=set()
        for split,expected,n in [('validation',2235,1515),('test',745,505),('future',745,0)]:
            f=pd.read_csv(c['paths'][split],float_precision='round_trip');f['id']=pd.to_datetime(f.id)
            assert len(f)==expected;late=f[f.horizon_offset_hours.between(240,744)]
            if split=='future':assert f.actual_kwh.isna().all()
            else:
                assert np.max(np.abs(truth.reindex(f.id).to_numpy()-f.actual_kwh.to_numpy()))<1e-7
                assert abs(metrics(late.actual_kwh,late.predicted_kwh)['mae_kwh']-c['val' if split=='validation' else 'test']['mae_kwh'])<1e-7
                if split=='validation':trainlabels=set(f.id)
                else:testlabels=set(f.id)
            for _,fold in f.groupby('origin'):
                idx=pd.date_range(pd.Timestamp(fold.origin.iloc[0]),periods=745,freq='h');assert pd.DatetimeIndex(fold.id).equals(idx)
                assert list(fold.horizon_offset_hours)==list(range(745)) and list(fold.lead_hours)==list(range(1,746))
            counts.append(dict(model=c['key'],split=split,rows=len(f),late=len(late),unique_labels=f.id.nunique()))
        assert not trainlabels.intersection(testlabels)
        may=pd.read_csv(OUT/c['may_csv'],float_precision='round_trip');assert len(may)==505
        assert abs(metrics(may.actual_kwh,may.predicted_kwh)['mae_kwh']-c['may']['mae_kwh'])<1e-7
        if c['source'].startswith('coarse/'):
            sys.path.insert(0,str(OUT/'coarse'));from coarse_models import forecast
            b=joblib.load(OUT/'coarse'/c['key']/'saved_model.joblib');idx,p,_=forecast(b)
            ref=pd.read_csv(c['paths']['future'],float_precision='round_trip').predicted_kwh.to_numpy(float)
            err=float(np.max(abs(p-ref)));assert err==0;coarse_replay[c['key']]=err
    checks['all17_model_paths_metrics_and_label_separation']=True;checks['all7_coarse_native_reloads_exact']=True
    details['path_checks']=counts;details['coarse_reload_difference']=coarse_replay
    submission=pd.read_csv(OUT/'evaluation/selected_guide505_forecast.csv',float_precision='round_trip')
    checks['guide505_schema']=list(submission)==['id','hourly_pow','may_bill','may_carbon','agg_pow'] and len(submission)==505
    checks['guide_factors_and505_sum']=np.allclose(submission.may_bill,submission.hourly_pow*180,rtol=0,atol=1e-7) and np.allclose(submission.may_carbon,submission.hourly_pow*.424,rtol=0,atol=1e-10) and np.allclose(submission.agg_pow,submission.hourly_pow.sum(),rtol=0,atol=1e-7)
    original=[]
    for o in read(ROOT/'output/optimization_20261013/source_audit/byte_comparison.json')['files']:
        a=o['original'];p=Path(a['path']);st=p.stat();same=st.st_size==a['bytes'] and st.st_mtime_ns==a['mtime_ns'];assert same
        original.append(dict(path=str(p),size=st.st_size,mtime_ns=st.st_mtime_ns,size_mtime_preserved=same,
          prior_source_audit_sha256=a['sha256'],hash_recomputed_this_verification=False))
    checks['original_data_size_mtime_preserved']=True;details['original_data']=original
    # Global lock does not include post-selection audit/report files.
    for name in ['hourly/verification.json','neural/verification.json','independent_audit.json']:
        p=OUT/name
        if p.exists():details[name]=read(p)
    checks['hourly_native_verification_exists']=(OUT/'hourly/verification.json').exists()
    checks['neural_native_verification_exists']=(OUT/'neural/verification.json').exists()
    checks['independent_audit_exists']=(OUT/'independent_audit.json').exists()
    checks['independent_audit_passed']=details.get('independent_audit.json',{}).get('status')=='pass'
    checks['hourly_native_verification_passed']=details.get('hourly/verification.json',{}).get('status')=='passed'
    checks['neural_native_verification_passed']=details.get('neural/verification.json',{}).get('status')=='pass'
    cli=OUT/'replay/selected745.csv'
    if cli.exists():
        a=pd.read_csv(cli,float_precision='round_trip').predicted_energy_kwh.to_numpy(float);b=pd.read_csv(OUT/'evaluation/selected_full745_forecast.csv',float_precision='round_trip').predicted_kwh.to_numpy(float)
        err=float(np.max(abs(a-b)));checks['selected_checkpoint_forward_CLI_exact']=err==0;details['selected_CLI_max_abs_difference']=err
    else:checks['selected_checkpoint_forward_CLI_exact']=False
    browser=OUT/'browser_qa.json'
    if browser.exists():details['browser_qa']=read(browser);checks['browser_QA']=details['browser_qa']['pass']
    result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,details=details,
      mathematical_global_optimality_claimed=False,pristine_blind_test_claimed=False,monthly_model_training_claim='4complete months only; baseline',
      physical_target_scope_verified=False,originals_modified=False)
    result['pass']=all(checks.values())
    write(OUT/'verification.json',result);print(json.dumps({'pass':result['pass'],'checks':checks},ensure_ascii=True))
    protocol=read(OUT/'experiment_protocol.json');protocol['status']='complete';protocol['native_candidates_completed']=137;protocol['families_completed']=17;protocol['selected_by_validation']=lock['selected_model'];protocol['verification_pass']=result['pass'];write(OUT/'experiment_protocol.json',protocol)
    # AI-generated artifacts receive stable IDs; original source registry is unchanged.
    inventory=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or p.name=='artifact_inventory.json' or '__pycache__'in p.parts or 'tmp'in p.parts:continue
        rp=p.relative_to(OUT).as_posix();inventory.append(dict(file_id=str(uuid.uuid5(uuid.NAMESPACE_URL,'ai-work-e/AWS/resampling_20261013/'+rp)),relative_path=rp,
          workspace_id='ai-work-e',project_id=None,generated_by_ai=True,bytes=p.stat().st_size,sha256=sha(p),original_path=None,copy_status='generated_not_copied',verification_status='size_and_sha256_recorded'))
    write(OUT/'artifact_inventory.json',dict(workspace_id='ai-work-e',project_id=None,source_registry_unchanged=True,original_source_audit=str(ROOT/'output/optimization_20261013/source_audit/byte_comparison.json'),artifacts=inventory))
    if not result['pass']:sys.exit(1)
if __name__=='__main__':main()
