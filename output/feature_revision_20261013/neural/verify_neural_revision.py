"""Verify all fitted prefixes, named channels, instance norms and saved replay."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import json
import numpy as np
import pandas as pd
from neural_revision import (OUT,load_hourly,TARGET,core_provenance,reload,infer,prefix,tensor_checks,predict_saved,dump)

def check_normalizer(saved,full):
    norm=saved['normalizer'];sch=saved['schema'];cols=sch['input_features']
    hist=full.loc[:pd.Timestamp(saved['history_end'])]
    assert norm['fit_end']==saved['history_end'] and norm['fit_rows']==len(hist)==saved['history_rows']
    assert norm['features']==cols and cols[sch['target_index']]==TARGET
    assert 0<=sch['target_index']<len(cols) and len(cols)==saved['architecture']['enc_in']
    x=hist[cols].to_numpy(float);mean=x.mean(axis=0);scale=x.std(axis=0);scale=np.where(scale>1e-8,scale,1.)
    assert np.allclose(mean,norm['mean'],rtol=0,atol=1e-11)
    assert np.allclose(scale,norm['scale'],rtol=0,atol=1e-11)
    assert pd.Timestamp(sch['selection_fit_end'])<pd.Timestamp('2025-04-01')
    return hist

def main():
    full=load_hourly();records=[];seen=set();normalization_cases=[]
    for scenario in ['dayahead24','long745']:
        for family in ['itransformer','patchtst']:
            dest=OUT/scenario/family;path=dest/'results.json'
            if not path.exists():continue
            r=json.loads(path.read_text(encoding='utf-8'));core_provenance(family)
            for c in r['candidate_results']:
                for fold in c['fold_fits']:
                    assert pd.Timestamp(fold['inner']['normalizer']['fit_end'])<pd.Timestamp(fold['inner']['inner_tail_start'])
                    assert fold['inner']['feature_selection_and_scaler_exclude_inner_tail'] is True
                    ck=Path(fold['refit']['path'])
                    if str(ck) in seen:continue
                    seen.add(str(ck));m,s=reload(ck);hist=check_normalizer(s,full)
                    assert pd.Timestamp(s['history_end'])<pd.Timestamp(fold['origin'])
                    checks=tensor_checks(family,m,hist,s['architecture'],s['normalizer'],s['schema'])
                    normalization_cases.append({'checkpoint':str(ck),'family':family,'scenario':scenario,**checks})
            m,s=reload(dest/'saved_model.pt');hist=check_normalizer(s,full)
            assert s['history_end']=='2025-04-29 23:00:00'
            expected=pd.read_csv(r['prediction_csv']['future'],parse_dates=['id'],float_precision='round_trip')
            replay=predict_saved(dest/'saved_model.pt');error=float(np.abs(replay.predicted_kwh-expected.predicted_kwh).max())
            assert error<1e-8 and replay.id.equals(expected.id)
            # The generic wide-history path uses the SAME stored feature names/scaler.
            generic=predict_saved(dest/'saved_model.pt',OUT.parent/'data'/'hourly_features.csv')
            generic_error=float(np.abs(generic.predicted_kwh-replay.predicted_kwh).max());assert generic_error<1e-8
            test=pd.read_csv(r['prediction_csv']['test'],parse_dates=['id','origin'],float_precision='round_trip')
            assert test.id.eq(test.origin+pd.to_timedelta(test.horizon_offset_hours,unit='h')).all()
            assert test.lead_hours.eq(test.horizon_offset_hours+1).all()
            assert np.allclose(test.actual_kwh,full.loc[test.id,TARGET],rtol=0,atol=1e-9)
            assert len(test)==(696 if scenario=='dayahead24' else 745)
            assert test.origin.nunique()==(29 if scenario=='dayahead24' else 1)
            assert r['may_labels_read'] is False
            records.append({'family':family,'scenario':scenario,'test_rows':len(test),'target_index':s['schema']['target_index'],
                            'input_channels':len(s['schema']['input_features']),'default_saved_context_replay_max_absdiff_kwh':error,
                            'generic_named_wide_history_replay_max_absdiff_kwh':generic_error,'all_named_training_scalers_verified':True})
    assert len(records)==4
    enabled=[r for r in normalization_cases if r['input_only_instance_norm_enabled']]
    assert enabled and any(r['family']=='itransformer' for r in enabled) and any(r['family']=='patchtst' for r in enabled)
    result={'passed':True,'models':records,'unique_CV_checkpoints_audited':len(seen),'all_CV_inner_scaler_feature_prefixes_exclude_inner_tail':True,
            'per_channel_instance_norm_enabled_cases_verified':len(enabled),'tensor_checks':normalization_cases,
            'all_test_label_timestamp_and_horizon_alignment_verified':True,'future_poison_guards_passed':True,
            'generic_history_supported':'Consecutive completed-hour wide CSV containing hour and every saved named input feature; feature names and scaler stay fixed',
            'May_labels_read':False,'old_vendor_hashes_unchanged':True}
    dump(OUT/'verification.json',result)
    print(json.dumps({'passed':True,'models':4,'CV_checkpoints':len(seen),'norm_enabled_checks':len(enabled)},ensure_ascii=True))

if __name__=='__main__':main()
