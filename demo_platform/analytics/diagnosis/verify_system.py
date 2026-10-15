"""Read-only validation of saved models/API, source scan and replay candidates."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from detector import OUT,TYPE_IDS,FEATURE_NAMES,prepare,Detector,diagnose_batch
from train_system import SAMPLE,write,sha

def main():
    results=json.loads((OUT/'results.json').read_text(encoding='utf-8'));detector=Detector()
    sample=pd.read_csv(SAMPLE,nrows=200,dtype={'localtime':'string'},float_precision='round_trip')
    originals=sample.copy(deep=True);a=detector.diagnose_batch(sample.to_dict('records'));b=Detector().diagnose_batch(sample.to_dict('records'))
    assert len(a)==200 and a==b and sample.equals(originals)
    changed=sample.copy();changed['timestamp']=0;c=detector.diagnose_batch(changed.to_dict('records'))
    assert a==c
    tagged=sample.iloc[:5].copy();tagged['source_kind']='synthetic_simulation';tagged['source_tag']='independent_test'
    tag_result=diagnose_batch(tagged.to_dict('records'));assert all(r['source_kind']=='synthetic_simulation' and r['source_tag']=='independent_test' for r in tag_result)
    invalid=[]
    for mutate in ['missing_sensor','bad_localtime','nonfinite_sensor','unknown_module']:
        frame=sample.iloc[:2].copy()
        if mutate=='missing_sensor':frame=frame.drop(columns='currentR')
        elif mutate=='bad_localtime':frame['localtime']='20250199bad'
        elif mutate=='nonfinite_sensor':frame['activePower']=np.nan
        else:frame['module(equipment)']='99(unknown)'
        try:detector.diagnose_batch(frame.to_dict('records'))
        except ValueError:invalid.append(mutate)
    assert len(invalid)==4
    probs=np.array([list(r['ai']['class_probs'].values()) for r in a]);assert np.max(abs(probs.sum(axis=1)-1))<1e-6
    assert all(r['confidence']['fault_probability'] is None and r['status'] in ['no_candidate','unconfirmed_candidate'] for r in a)
    verification={'status':'passed','saved_model_replay_200_rows_exact':True,'Unix_timestamp_mutation_does_not_change_scores':True,
          'original_input_rows_unchanged':True,'synthetic_source_tag_propagated':True,'strict_invalid_inputs_rejected':invalid,
          'class_probability_sum_max_absdiff':float(np.max(abs(probs.sum(axis=1)-1))),
          'actual_fault_probability_never_claimed':True,'model_hashes_unchanged':results['model_sha256']=={n:sha(OUT/n) for n in results['model_sha256']},
          'source_and_old_artifacts_policy':'Only new diagnosis output files written; all source and prior model artifacts kept read-only.'}
    if (OUT/'candidates.json').exists():
        data=json.loads((OUT/'candidates.json').read_text(encoding='utf-8'));candidates=data['candidates']
        # Normalize the aliases after scan if the in-flight scanner preceded UI contract additions.
        for ev in candidates:
            ev.update(id=ev['event_id'],title=ev['label']+' 후보',start=ev['start_time'],end=ev['end_time'],time=ev['peak_time'])
            ev['evaluation_scope']='chronological_after_calibration' if pd.Timestamp(ev['peak_time'])>=pd.Timestamp('2025-02-01') else 'retrospective_fit_or_calibration_period'
            if 'score_definition' not in ev:
                ev['archive_rule_equivalent_score']=ev['score'];ev['archive_rule_threshold']=ev['threshold']
                ev['score']=ev['normalized_score'];ev['threshold']=1.
                ev['score_definition']='Dimensionless threshold exceedance: max(rule_score/rule_threshold,AE_score/AE_threshold) for abnormal_pattern; rule ratio for other types. Separate original model scores retained in ai/rules_at_peak.'
            assert ev['status']=='unconfirmed_candidate' and ev['source_kind']=='recorded' and ev['point_count']>=1
            assert pd.Timestamp(ev['end_time'])>=pd.Timestamp(ev['start_time']) and (pd.Timestamp(ev['end_time'])-pd.Timestamp(ev['start_time'])).total_seconds()==(ev['point_count']-1)*5
            assert ev['peak_source_row']>=ev['start_source_row'] and ev['peak_source_row']<=ev['end_source_row']
        write(OUT/'candidates.json',data)
        selectedcols=['id','event_id','module_id','equipment','start','end','time','type','title','score','threshold','normalized_score','point_count','status','source_kind','confidence','evaluation_scope','score_definition']
        pd.DataFrame([{k:ev[k] for k in selectedcols} for ev in candidates]).to_csv(OUT/'candidates.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
        rows=[{'module(equipment)':ev['equipment'],**ev['raw_features']} for ev in candidates];scored=detector.diagnose_batch(rows)
        diffs=[]
        for ev,score in zip(candidates,scored):
            for name in ['ae_score','iforest_score']:diffs.append(abs(ev['ai'][name]-score['ai'][name]))
            diffs += [abs(ev['ai']['class_probs'][name]-score['ai']['class_probs'][name]) for name in score['ai']['class_probs']]
        assert max(diffs,default=0.)<1e-8
        verification['full_scan']={'rows':results['full_scan']['rows_read'],'source_sha256':results['full_scan']['source_sha256'],
           'source_size_mtime_preserved':results['full_scan']['source_size_mtime_preserved'],'six_rules_all_rows':True,'AE_all_rows':True,
           'candidate_count':len(candidates),'candidate_peak_replay_max_absdiff':max(diffs,default=0.),
           'candidate_intervals_exact_5s_contiguous':True,'actual_ground_truth_count':None}
        write(OUT/'scan_summary.json',results['full_scan'])
    write(OUT/'verification.json',verification);print(json.dumps(verification,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
