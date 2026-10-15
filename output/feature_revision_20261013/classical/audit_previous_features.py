"""Independent temporal-alignment audit of the preserved previous runner."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import hashlib
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
OLD=OUT.parents[1]/'optimization_20261013/forecast'
sys.path.insert(0,str(OLD))
from classical_opt import origin_features,build_train,build_future,calendar,FEATURE_NAMES,CALENDAR_NAMES

def main():
    idx=pd.date_range('2024-12-01',periods=960,freq='h')
    y=pd.Series(np.arange(len(idx),dtype=float)+39000,index=idx)
    checks=[]
    for fs in ['history','history_calendar']:
        x,a=build_train(y,24,fs)
        row=0
        max_difference=0.
        for pos in range(168,len(y),6):
            leads=min(24,len(y)-pos)
            future_x=build_future(y.iloc[:pos],idx[pos:pos+leads],fs)
            max_difference=max(max_difference,float(np.max(np.abs(x[row:row+leads]-future_x))))
            assert np.array_equal(a[row:row+leads],y.iloc[pos:pos+leads].to_numpy())
            row+=leads
        assert row==len(x) and max_difference==0
        checks.append({'feature_set':fs,'historical_training_vs_inference_max_absdiff':max_difference,
                       'all_training_target_indices_within_prefix':True,'training_rows_checked':row})
    pos=600
    original=origin_features(y.to_numpy(),pos,idx[pos])
    changed=y.to_numpy().copy();changed[pos:]=-999999
    assert original==origin_features(changed,pos,idx[pos])
    x=build_future(y.iloc[:pos],idx[pos:pos+24],'history_calendar')
    assert np.all(x[:,1]==y.iloc[pos-24])
    safe_day=[int(pos+offset-24) for offset in range(24)]
    assert max(safe_day)<pos
    examples=[]
    for lead in [1,6,24]:
        offset=lead-1
        examples.append({'lead_hours':lead,'origin':str(idx[pos]),'last_observed':str(idx[pos-1]),
                         'target_hour':str(idx[pos+offset]),'fixed_origin_lag24_hour':str(idx[pos-24]),
                         'new_target_aligned_daily_lag_hour':str(idx[pos+offset-24]),
                         'new_target_aligned_weekly_lag_hour':str(idx[pos+offset-168])})
    source=OLD/'classical_opt.py'
    result={'status':'passed','temporal_alignment_bug_found':False,
            'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'checks':checks,'future_target_mutation_leaves_origin_features_unchanged':True,
            'fixed_origin_lag24_is_constant_across_24_predictions':True,
            'interpretation':'Constant origin lag24 is an intentional fixed-origin feature, not an indexing bug. New candidates additionally use target-hour-aligned past daily/weekly values.',
            'lead_examples':examples,
            'long745_limit':'Naive target-offset-minus24 sensor/target values leak beyond offset23. The new long runner uses explicit last-observed-week phase wrapping or omits these columns.',
            'feature_names':FEATURE_NAMES+CALENDAR_NAMES,
            'previous_training_notes':'Direct examples used stride6, repeated labels and partial last-origin horizons. Residual center was the mean of those training labels. These are modeling/weighting choices, not future-label leakage.',
            'previous_inputs':'Only past aggregate target and deterministic calendar; no past module or physical-sensor features.',
            'old_outputs_modified':False}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'previous_feature_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','temporal_alignment_bug_found':False,'checks':checks}))

if __name__=='__main__':main()
