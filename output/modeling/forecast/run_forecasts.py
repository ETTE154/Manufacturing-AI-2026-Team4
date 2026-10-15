"""Select on blocked validation, lock, then compare internal test and May.

External neural models can reuse common.py and emit the identical prediction schema.
"""
import argparse
from pathlib import Path
import time
import json
import platform
import importlib.metadata
import numpy as np
import pandas as pd
from common import (OUT,VALIDATION_ORIGIN,TEST_ORIGIN,MAX_HORIZON,load_training,
                    future_index,history_before,actual_at,protocol,write_json,
                    score,score_prefixes,baseline_predict,prediction_frame,may_labels_after_lock)
from arima_xgb import select_arima,predict_arima,select_xgb,predict_xgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models',nargs='+',default=['arima','xgboost'],choices=['arima','xgboost'])
    parser.add_argument('--n-jobs',type=int,default=2)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--internal-only',action='store_true',help='Freeze selections and evaluate internal test without opening May labels')
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    training = load_training()
    manifest = protocol(training)
    write_json(OUT/'protocol.json',manifest)
    if args.prepare_only:
        print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
        return
    np.random.seed(42)
    version_packages = ['numpy','pandas','statsmodels','xgboost']
    versions = {'python':platform.python_version()}
    for name in version_packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    valid_history = history_before(training,VALIDATION_ORIGIN)
    valid_future = future_index(VALIDATION_ORIGIN)
    valid_actual = actual_at(training,valid_future)
    test_history = history_before(training,TEST_ORIGIN)
    test_future = future_index(TEST_ORIGIN)
    test_actual = actual_at(training,test_future)
    methods = {'arima':(select_arima,predict_arima),'xgboost':(select_xgb,predict_xgb)}
    selected = {}
    results = {'environment':versions,'protocol_file':str(OUT/'protocol.json'),'models':{}}
    frames = []
    for name in args.models:
        select,predict = methods[name]
        kwargs = {'n_jobs':args.n_jobs} if name=='xgboost' else {}
        started = time.monotonic()
        config,evidence,valid_pred = select(valid_history,valid_future,valid_actual.to_numpy(),**kwargs)
        selected[name] = config
        # Persist each concrete choice before any internal test or May label access.
        write_json(OUT/f'{name}_selection.json',{'selection':evidence,'environment':versions,
                   'validated_origin':str(VALIDATION_ORIGIN),'honest_may_EDA_disclosure':manifest['honest_holdout_disclosure']})
        print(f'{name}: blocked-validation config frozen {config}',flush=True)
        test_pred,test_fit = predict(test_history,test_future,config,**kwargs)
        frames.append(prediction_frame(name,'validation',VALIDATION_ORIGIN,valid_future,valid_actual,valid_pred))
        frames.append(prediction_frame(name,'internal_test',TEST_ORIGIN,test_future,test_actual,test_pred))
        result = {'frozen_config':config,'validation':score_prefixes(valid_actual,valid_pred),
                  'internal_test':score_prefixes(test_actual,test_pred),'internal_test_fit':test_fit,
                  'internal_runtime_seconds':time.monotonic()-started}
        results['models'][name] = result
        write_json(OUT/f'{name}_internal_results.json',result)
        print(f'{name}: internal test P745 MAE={result["internal_test"]["P745"]["mae_kwh"]:.6f}',flush=True)
    baseline_names = ['mean','median','last_week_mean','hour_mean','weekday_hour_mean','seasonal_24','seasonal_168']
    for baseline in baseline_names:
        valid_pred = baseline_predict(baseline,valid_history,valid_future)
        test_pred = baseline_predict(baseline,test_history,test_future)
        frames.append(prediction_frame('baseline_'+baseline,'validation',VALIDATION_ORIGIN,valid_future,valid_actual,valid_pred))
        frames.append(prediction_frame('baseline_'+baseline,'internal_test',TEST_ORIGIN,test_future,test_actual,test_pred))
        results['models']['baseline_'+baseline] = {'validation':score_prefixes(valid_actual,valid_pred),
                                                  'internal_test':score_prefixes(test_actual,test_pred)}
    # A single manifest locks every requested model configuration before May is read.
    write_json(OUT/'arima_xgb_selection_lock.json',{'configs':selected,'environment':versions,
               'validation_origin':str(VALIDATION_ORIGIN),'protocol':manifest,
               'selection_uses_may_labels':False,'selection_uses_internal_test_labels':False})
    pd.concat(frames,ignore_index=True).to_csv(OUT/'internal_predictions.csv',index=False,encoding='utf-8-sig')
    if not args.internal_only:
        may_origin = training.index.max()+pd.Timedelta(hours=1)
        may_future = future_index(may_origin)
        may_full = []
        may_pred_map = {}
        for name in args.models:
            _,predict = methods[name]
            kwargs = {'n_jobs':args.n_jobs} if name=='xgboost' else {}
            path,fit_meta = predict(training,may_future,selected[name],**kwargs)
            may_pred_map[name] = path
            results['models'][name]['may_full_history_fit'] = fit_meta
        # Baseline weekday-hour-mean was chosen in the prior analysis without this experiment's May tuning.
        for baseline in ['mean','weekday_hour_mean']:
            may_pred_map['baseline_'+baseline] = baseline_predict(baseline,training,may_future)
        # Predictions are concrete and frozen before opening the May labels.
        for name,path in may_pred_map.items():
            prediction_frame(name,'may_full_bridge',may_origin,may_future,np.repeat(np.nan,len(may_future)),path).to_csv(OUT/f'{name}_may_full_bridge.csv',index=False,encoding='utf-8-sig')
        labels = may_labels_after_lock()
        for name,path in may_pred_map.items():
            predicted = pd.Series(path,index=may_future).reindex(labels.index)
            if predicted.isna().any():
                raise ValueError('The 745-hour fixed path does not cover every May label')
            results['models'][name]['may_505_labelled_hours'] = score(labels.to_numpy(),predicted.to_numpy())
            frame = prediction_frame(name,'may_labelled',may_origin,labels.index,labels.to_numpy(),predicted.to_numpy())
            may_full.append(frame)
        pd.concat(may_full,ignore_index=True).to_csv(OUT/'may_labelled_predictions.csv',index=False,encoding='utf-8-sig')
    write_json(OUT/'arima_xgb_results.json',results)
    print(json.dumps({name:{'internal_test_P745':r['internal_test']['P745'],
                           'may_505':r.get('may_505_labelled_hours')} for name,r in results['models'].items()},ensure_ascii=False,indent=2),flush=True)

if __name__ == '__main__':
    main()
