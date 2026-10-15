"""Meaningful artifact checks: provenance, masks, splits, flags, inference replay."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from anomaly_study import weak,metrics
from predict_anomaly import load,infer

OUT=Path(__file__).resolve().parent
KEY='module(equipment)'

def main():
    report=json.loads((OUT/'anomaly_metrics.json').read_text(encoding='utf-8'))
    locks=json.loads((OUT/'threshold_locks.json').read_text(encoding='utf-8'))
    manifest=json.loads((OUT/'augmented_sample_manifest.json').read_text(encoding='utf-8'))
    windows=pd.read_csv(OUT/'continuous_source_windows.csv.gz',parse_dates=['dt'])
    scores=pd.read_csv(OUT/'continuous_window_scores.csv.gz',parse_dates=['dt'],float_precision='round_trip')
    original=pd.read_csv(OUT/'original_test_scores.csv.gz',float_precision='round_trip')
    synthetic=pd.read_csv(OUT/'synthetic_test_scores.csv.gz',float_precision='round_trip')
    candidate=pd.read_csv(OUT/'full_weak_candidate_rows.csv.gz')
    assert len(candidate)==63963 and weak(candidate).sum()==len(candidate)
    assert not windows.duplicated(['window_id','localtime']).any()
    assert windows[['window_id','localtime']].equals(scores[['window_id','localtime']])
    for wid,d in windows.groupby('window_id'):
        assert len(d)==360 and d.dt.diff().dropna().eq(pd.Timedelta(seconds=5)).all()
    expected={'train':('2024-12-01','2025-03-01'),'validation':('2025-03-01','2025-04-01'),'test':('2025-04-01','2025-04-30')}
    ids=[];seeds=[]
    for e in manifest['events']:
        ids.append(e['event_id']);seeds.append(e['seed']);a,b=expected[e['split']]
        assert pd.Timestamp(a)<=pd.Timestamp(e['source_base_start'])<pd.Timestamp(e['source_base_end_exclusive'])<=pd.Timestamp(b)
        assert e['mask_rows']*5==e['duration_seconds']
    assert len(ids)==len(set(ids)) and len(seeds)==len(set(seeds))
    byid={e['event_id']:e for e in manifest['events']}
    for eid,d in synthetic.groupby('event_id'):
        e=byid[eid];assert e['split']=='test' and len(d)==360
        ix=np.flatnonzero(d.synthetic_label.to_numpy())
        assert np.array_equal(ix,np.arange(e['start_offset'],e['stop_offset']))
        assert int(d.iloc[ix[0]].localtime)==e['start_localtime']
    for name,lock in locks.items():
        for data in [scores,original,synthetic]:
            p=data[name+'_score'].to_numpy(dtype=np.float64);alarm=data[name+'_alarm'].to_numpy()
            assert np.array_equal(p>=float(lock['threshold']),alarm.astype(bool)),(name,'flag mismatch')
        recomputed=metrics(original.weak_candidate,original[name+'_score'],lock['threshold'])
        reference=report['models'][name]['original_weak_agreement']
        for k in ['true_positive','false_positive','false_negative','true_negative','rows']:
            assert recomputed[k]==reference[k],(name,k)
        sm=metrics(synthetic.synthetic_label,synthetic[name+'_score'],lock['threshold'])
        for k in ['true_positive','false_positive','false_negative','true_negative','rows']:
            assert sm[k]==report['models'][name]['synthetic_heldout_metrics'][k]
    if (OUT/'prior_seven_score_snapshot.npz').exists():
        before=np.load(OUT/'prior_seven_score_snapshot.npz')
        frames={'continuous_window_scores.csv.gz':scores,'original_test_scores.csv.gz':original,
                'synthetic_test_scores.csv.gz':synthetic,
                'full_candidate_scores.csv.gz':pd.read_csv(OUT/'full_candidate_scores.csv.gz',float_precision='round_trip')}
        for key in before.files:
            fn,col=key.split('|',1);assert np.array_equal(frames[fn][col].to_numpy(),before[key]),(fn,col,'prior score changed')
    provenance=json.loads((OUT/'legacy_model_provenance.json').read_text(encoding='utf-8'))
    for model,files in provenance.items():
        for fn,p in files.items():
            assert hashlib.sha256(Path(p['source']).read_bytes()).hexdigest()==p['sha256']
            assert hashlib.sha256((OUT/'models'/fn).read_bytes()).hexdigest()==p['sha256']
    # Replay points from each actual candidate/normal illustration, including extremes.
    subset=windows[windows.window_id.str.startswith(('candidate_','normal_'))].copy()
    replay=infer(subset,load())
    reference=scores[scores.window_id.str.startswith(('candidate_','normal_'))]
    for name in locks:
        a=replay[name+'_score'].to_numpy();b=reference[name+'_score'].to_numpy()
        assert np.allclose(a,b,rtol=3e-7,atol=1e-10),name
        assert np.array_equal(replay[name+'_alarm'].to_numpy(),reference[name+'_alarm'].to_numpy()),name
    summary={'passed':True,'models':len(locks),'source_candidate_rows':len(candidate),'continuous_windows':windows.window_id.nunique(),
             'synthetic_events':len(manifest['events']),'synthetic_test_events':synthetic.event_id.nunique(),
             'all_split_masks_seed_and_flag_checks_passed':True,'unchanged_legacy_native_artifact_hashes':True,
             'saved_inference_replay_observations':len(subset),'saved_inference_replay_alarm_matches':True,
             'previous_seven_score_arrays_exactly_preserved':(OUT/'prior_seven_score_snapshot.npz').exists(),
             'numeric_replay_tolerance':'rtol 3e-7, atol 1e-10 for score CSV round trip; alarms exact'}
    (OUT/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
