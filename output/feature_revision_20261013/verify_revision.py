"""Independent metric/index/lock reconciliation of the completed revision."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(a,p):
    a,p=np.asarray(a,float),np.asarray(p,float);e=p-a
    return {'mae_kwh':float(np.abs(e).mean()),'rmse_kwh':float(np.sqrt((e*e).mean())),
            'mape_percent':float((np.abs(e)/np.abs(a)).mean()*100),'bias_pred_minus_actual_kwh':float(e.mean()),
            'r2':float(1-(e*e).sum()/((a-a.mean())**2).sum())}
def main():
    comp=load(OUT/'comparison_results.json');lock=load(OUT/'comparison_selection_lock.json')
    assert not lock['April_or_May_scores_used_for_selection']
    assert sha(OUT/'data/hourly_features.csv')==lock['feature_sha256']
    target=pd.read_csv(ROOT/'output/analysis/baseline_audit/complete_hour_module_sum.csv')
    target=pd.Series(target.iloc[:,1].values,index=pd.to_datetime(target.iloc[:,0]))
    truth=pd.read_csv(ROOT/'may-dataset/rtu_ground_truth_may.csv');mt=pd.to_datetime(truth.id)
    audits={name:load(OUT/p) for name,p in {
        'data':'data/extraction_verification.json','analysis':'analysis/analysis_verification.json',
        'classical_day':'classical/verification.json','classical_long':'classical/long745/verification.json',
        'neural':'neural/verification.json','independent_model':'independent_model_audit.json'}.items()}
    checks=[]
    for s,scenario in comp['scenarios'].items():
        minimum=min(m['validation_metrics']['mae_kwh'] for m in scenario['models'].values())
        assert abs(scenario['models'][scenario['selected_model']]['validation_metrics']['mae_kwh']-minimum)<1e-6
        assert lock['selected_models'][s]==scenario['selected_model']
        for k,m in scenario['models'].items():
            modes=[('validation','validation_metrics'),('test','test_metrics' if s=='dayahead24' else 'internal_test_metrics')]
            if s=='long745':modes.append(('may','may_metrics'))
            for split,key in modes:
                p=Path(m['paths'][f'{split}_csv']);f=pd.read_csv(p,float_precision='round_trip');t=pd.to_datetime(f.id);o=pd.to_datetime(f.origin)
                expected=1680 if s=='dayahead24' and split=='validation' else 696 if s=='dayahead24' else 505 if split=='may' else 745
                assert len(f)==expected and not f.duplicated(['origin','id']).any()
                assert np.array_equal(((t-o)/pd.Timedelta(hours=1)).to_numpy(int),f.horizon_offset_hours)
                assert np.array_equal(f.lead_hours,f.horizon_offset_hours+1)
                actual=truth.hourly_pow.to_numpy(float) if split=='may' else target.reindex(t).to_numpy(float)
                assert np.allclose(f.actual_kwh,actual,rtol=0,atol=1e-8)
                if split=='may':assert t.reset_index(drop=True).equals(mt.reset_index(drop=True)) and f.lead_hours.min()==241 and f.lead_hours.max()==745
                if s=='dayahead24':assert f.groupby('origin').size().eq(24).all() and f.origin.nunique()==(70 if split=='validation' else 29)
                value=metrics(f.actual_kwh,f.predicted_kwh);reported=m[key]
                diff=max(abs(value[a]-reported[a]) for a in value)
                assert diff<1e-7 and reported['count']==len(f)
                checks.append({'scenario':s,'model':k,'split':split,'rows':len(f),'max_abs_metric_difference':diff,'actual_and_lead_alignment':True})
            future=pd.read_csv(m['paths']['future_csv']);t=pd.to_datetime(future.id);points=24 if s=='dayahead24' else 745
            assert len(future)==points and np.array_equal(future.lead_hours,np.arange(1,points+1))
            if 'actual_kwh' in future:assert future.actual_kwh.isna().all()
            assert t.iloc[0]==pd.Timestamp('2025-04-30') and t.iloc[-1]==pd.Timestamp('2025-04-30 23:00:00' if points==24 else '2025-05-31')
    native_locks={'classical_day':OUT/'classical/selection_lock.json','classical_long':OUT/'classical/long745/selection_lock.json'}
    for s in ['dayahead24','long745']:
        for name in ['itransformer','patchtst']:native_locks[f'{s}/{name}']=OUT/'neural'/s/name/'selection_lock.json'
    assert all(sha(p)==lock['new_model_selection_lock_sha256'][k] for k,p in native_locks.items())
    source=load(ROOT/'output/optimization_20261013/source_audit/byte_comparison.json');sources=[]
    for item in source['files']:
        p=Path(item['original']['path']);stat=p.stat()
        result={'source_key':item['source_key'],'size_unchanged':stat.st_size==item['original']['bytes'],
                'mtime_unchanged_since_full_SHA_verification':stat.st_mtime_ns==item['original']['mtime_ns'],
                'previous_full_sha256':item['original']['sha256'],'previous_official_copy_identical':item['sha256_match']}
        assert result['size_unchanged'] and result['mtime_unchanged_since_full_SHA_verification'] and result['previous_official_copy_identical']
        sources.append(result)
    assert comp['May_not_used_for_reselection']
    for k,f in comp['figures'].items():assert Path(f['png']).is_file() and Path(f['svg']).is_file()
    result={'passed':True,'CSV_evaluation_checks':checks,'config_locks_unchanged':True,'validation_only_global_selection':True,
        'original_sources_preserved':sources,'artifact_verifications':audits,'figure_pairs_in_gallery':len(comp['figures']),
        'limitations':['Replay and numeric checks do not prove that a model predicts future load well.',
            'Existing April/May EDA precludes a pristine prospective test claim.',
            'Source unit and meter hierarchy uncertainty is not resolved by these tests.']}
    (OUT/'verification_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('PASS',len(checks),'CSV evaluations; config lock and timing preserved')
if __name__=='__main__':main()
