"""Independent artifact reload, alignment, score and input boundary checks."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'output/modeling/forecast'))
from common import OUT,load_training,future_index,history_before,VALIDATION_ORIGIN,TEST_ORIGIN,score,write_json
from predict_forecast import run

def main():
    series=load_training();future=future_index(series.index[-1]+pd.Timedelta(hours=1))
    assert len(future)==745 and future[0]==pd.Timestamp('2025-04-30') and future[-1]==pd.Timestamp('2025-05-31')
    assert future[240]==pd.Timestamp('2025-05-10') and len(future[240:])==505
    assert future_index(VALIDATION_ORIGIN)[-1]<TEST_ORIGIN
    paths={'arima':OUT/'arima_may_full_bridge.csv','xgboost':OUT/'xgboost_may_full_bridge.csv',
      'patchtst':OUT/'patchtst/may_full_bridge.csv','itransformer':OUT/'itransformer/may_full_path.csv',
      'chronos2':OUT/'chronos2/may_full_path.csv'}
    models={}
    for name,path in paths.items():
        started=time.perf_counter();index,pred,qs=run(name,series)
        saved=pd.read_csv(path,parse_dates=['id'])
        assert pd.DatetimeIndex(saved.id).equals(index)
        np.testing.assert_allclose(pred,saved.predicted_kwh,rtol=0,atol=1e-8)
        models[name]={'reload_max_absolute_difference_kwh':float(np.max(np.abs(pred-saved.predicted_kwh))),
                      'prediction_count':len(pred),'finite_predictions':bool(np.isfinite(pred).all()),
                      'verification_seconds':time.perf_counter()-started}
        print(name,models[name],flush=True)
    comparison=json.loads((OUT/'comparison_results.json').read_text(encoding='utf-8'))
    rows=pd.read_csv(OUT/'all_predictions.csv',parse_dates=['id'])
    assert not rows.duplicated(['model','split','id']).any()
    for model,info in comparison['models'].items():
        for split,metric in [('validation',info['validation']['P745']),('internal_test',info['internal_test']['P745']),('may_labelled',info['may']['P505'])]:
            part=rows[(rows.model==model)&(rows.split==split)]
            np.testing.assert_allclose(score(part.actual_kwh,part.predicted_kwh)['mae_kwh'],metric['mae_kwh'],rtol=0,atol=1e-7)
            assert len(part)==(505 if split=='may_labelled' else 745)
        for split in ['internal_test','may_labelled']:
            own=rows[(rows.model==model)&(rows.split==split)]
            mean=rows[(rows.model=='baseline_mean')&(rows.split==split)]
            delta=score(own.actual_kwh,own.predicted_kwh)['mae_kwh']-score(mean.actual_kwh,mean.predicted_kwh)['mae_kwh']
            paired=info['paired_error_analysis'][split]
            np.testing.assert_allclose(delta,paired['delta_mae_model_minus_mean_kwh'],rtol=0,atol=1e-8)
            assert paired['count']==len(own)
    # Forecast features must depend only on history and deterministic clock data.
    from arima_xgb import direct_training_examples,direct_future_features
    for origin in [VALIDATION_ORIGIN,TEST_ORIGIN]:
        hist=history_before(series,origin);X,y,meta=direct_training_examples(hist)
        assert pd.Timestamp(meta['example_target_end'])<origin
        assert direct_future_features(hist,future_index(origin)).shape==(745,18)
    lock=json.loads((OUT/'deployment_selection_lock.json').read_text(encoding='utf-8'))
    assert not lock['selection_uses_may_labels'] and not lock['selection_uses_internal_test_labels']
    for path,digest in lock['model_lock_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    original={str(p):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in [ROOT/'프로젝트가이드.pdf',ROOT/'traing-dataset/rtu_data_full.csv',ROOT/'may-dataset/rtu_ground_truth_may.csv']}
    result={'status':'passed','models':models,'metric_reconciliation':'All 8 models across validation/internal test/May reconciled to stored prediction CSV',
       'chronological_boundaries':'Complete future P745, nonoverlapping outer targets, XGB supervised labels before each training cutoff',
       'locked_selection_sha256_verified':True,
       'paired_difference_window_counts_and_metrics':'passed, internal test P745 and May P505 separately',
       'original_read_only_inventory':original}
    write_json(ROOT/'output/modeling/verification_results.json',result)

if __name__=='__main__':main()
