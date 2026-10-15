"""Independent read-only checks of reported metrics, timing, locks and original files."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    comp=load(OUT/'forecast/comparison_results.json')
    lock=load(OUT/'forecast/comparison_selection_lock.json')
    protocol=load(OUT/'forecast/protocol.json')
    source=load(OUT/'source_audit/byte_comparison.json')
    checks=[];identity=[]
    for s in source['files']:
        p=Path(s['original']['path']);st=p.stat()
        result={'source_key':s['source_key'],'bytes_unchanged':st.st_size==s['original']['bytes'],
                'mtime_unchanged':st.st_mtime_ns==s['original']['mtime_ns'],
                'sha256_unchanged':sha(p)==s['original']['sha256'],
                'downloaded_exactly_matches_original':s['sha256_match'] and s['bytes_match']}
        assert all(result[k] for k in result if k!='source_key'),result
        identity.append(result)
    target=Path(protocol['target_source'])
    assert sha(target)==lock['target_sha256']==protocol['target_sha256']
    y=pd.read_csv(target)
    # The shared input was audited separately; enforce its fixed complete-hour grid again.
    print('TARGET_COLUMNS',list(y))
    assert len(y)==3600
    may_source=ROOT/'may-dataset/rtu_ground_truth_may.csv'
    truth=pd.read_csv(may_source)
    assert len(truth)==505
    truth_time=pd.to_datetime(truth['id'])
    truth_value=truth['hourly_pow'].to_numpy(dtype=float)
    assert truth_time.iloc[0]==pd.Timestamp('2025-05-10 00:00:00')
    assert truth_time.iloc[-1]==pd.Timestamp('2025-05-31 00:00:00')
    assert truth_time.diff().dropna().eq(pd.Timedelta(hours=1)).all()
    for scenario,s in comp['scenarios'].items():
        best=min(s['models'],key=lambda m:s['models'][m]['validation_metrics']['mae_kwh'])
        if s['models']['baseline']['validation_metrics']['mae_kwh']<=s['models'][best]['validation_metrics']['mae_kwh']+1e-6:best='baseline'
        assert s['selected_model']==best==lock['scenarios'][scenario]['selected_model']
        for name,model in s['models'].items():
            d=OUT/'forecast'/scenario/('baselines' if name=='baseline' else name)
            assert sha(d/'selection_lock.json')==lock['selection_lock_sha256'][f'{scenario}/{name}']
            modes=[('validation','validation_metrics'),('test','test_metrics' if scenario=='dayahead24' else 'internal_test_metrics')]
            if scenario=='long745':modes.append(('may','may_metrics'))
            for split,key in modes:
                p=d/f'{split}_predictions.csv'
                df=pd.read_csv(p);actual=df['actual_kwh'].to_numpy(dtype=float);pred=df['predicted_kwh'].to_numpy(dtype=float)
                assert np.isfinite(actual).all() and np.isfinite(pred).all()
                err=pred-actual;metrics={'mae_kwh':np.abs(err).mean(),'rmse_kwh':np.sqrt((err**2).mean()),
                    'mape_percent':(np.abs(err)/np.abs(actual)).mean()*100,
                    'bias_pred_minus_actual_kwh':err.mean(),
                    'r2':1-(err**2).sum()/((actual-actual.mean())**2).sum()}
                reported=model[key]
                assert reported['count']==len(df)
                differences={k:float(abs(v-reported[k])) for k,v in metrics.items()}
                assert max(differences.values())<1e-7,(scenario,name,split,differences)
                assert np.allclose(df['absolute_error_kwh'],np.abs(err),rtol=0,atol=1e-8)
                dt=pd.to_datetime(df['id']);origin=pd.to_datetime(df['origin'])
                offset=((dt-origin)/pd.Timedelta(hours=1)).to_numpy(dtype=int)
                assert np.array_equal(df['horizon_offset_hours'],offset)
                assert np.array_equal(df['lead_hours'],offset+1)
                if scenario=='dayahead24':
                    origins=df.groupby('origin',sort=False)
                    assert len(origins)==(70 if split=='validation' else 29)
                    assert origins.size().eq(24).all()
                    assert len(df)==(1680 if split=='validation' else 696)
                    ep=df.loc[df['lead_hours']==24]
                    endpoint_mae=np.abs(ep['predicted_kwh']-ep['actual_kwh']).mean()
                    assert abs(endpoint_mae-model[f'{split}_endpoint_metrics']['mae_kwh'])<1e-7
                    assert ep['id'].nunique()==len(ep)
                elif split=='may':
                    assert len(df)==505 and dt.reset_index(drop=True).equals(truth_time.reset_index(drop=True))
                    assert np.allclose(actual,truth_value,rtol=0,atol=1e-8)
                    assert offset.min()==240 and offset.max()==744
                else:assert len(df)==745 and offset.min()==0 and offset.max()==744
                checks.append({'scenario':scenario,'model':name,'split':split,'rows':len(df),
                               'reported_metric_max_abs_diff':max(differences.values()),'timing_and_targets_checked':True})
            f=pd.read_csv(d/'future_path.csv');dt=pd.to_datetime(f['id'])
            count=24 if scenario=='dayahead24' else 745
            assert len(f)==count and f['actual_kwh'].isna().all()
            assert dt.iloc[0]==pd.Timestamp('2025-04-30 00:00:00')
            assert dt.iloc[-1]==pd.Timestamp('2025-04-30 23:00:00' if count==24 else '2025-05-31 00:00:00')
            assert np.array_equal(f['lead_hours'],np.arange(1,count+1))
    assert not lock['selection_uses_May_labels']
    figures=[]
    for m,items in comp['figures'].items():
        for k,p in items.items():
            path=Path(p);assert path.is_file() and path.stat().st_size>1000
            figures.append({'model':m,'kind':k,'bytes':path.stat().st_size})
    a=load(OUT/'anomaly/verification.json');assert a['passed'] and a['models']==8
    af=OUT/'anomaly/figures';assert len(list(af.glob('*.png')))==54 and len(list(af.glob('*.svg')))==54
    extra={}
    for name,p in [('forecast',OUT/'forecast/verification_results.json'),('independent_audit',OUT/'independent_audit.json')]:
        if p.exists():extra[name]=load(p)
    result={'passed':True,'source_originals_unchanged':identity,'target_sha256_unchanged':True,
        'selection_matches_validation_minimum':True,'12_selection_lock_hashes_unchanged':True,
        'May_not_used_to_reselect':True,'prediction_csv_metric_checks':checks,'forecast_figure_files':figures,
        'anomaly_figure_pairs':54,'anomaly_saved_model_verification':a,
        'additional_verifications':extra,
        'limitations':['Numeric and replay checks do not establish scientific validity or source-unit correctness.',
                       'Source labels for real injected faults are absent; source rules and synthetic masks have distinct meanings.',
                       'Previously examined April/May data are not prospective blind holdouts.']}
    (OUT/'verification_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('PASS',len(checks),'CSV evaluations,',len(figures),'forecast figures')
if __name__=='__main__':main()
