"""Refit frozen ARIMA/XGB configurations and score May without reselection.

Run after the parent experiment has frozen the deployment choice on validation.
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
from common import (OUT,load_training,protocol,future_index,write_json,
                    baseline_predict,prediction_frame,may_labels_after_lock,score)
from arima_xgb import predict_arima,predict_xgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock',default=str(OUT/'arima_xgb_selection_lock.json'))
    parser.add_argument('--n-jobs',type=int,default=4)
    parser.add_argument('--save-only',action='store_true',help='Save frozen final models and full bridge forecasts without opening May labels')
    args = parser.parse_args()
    locked = json.loads(open(args.lock,encoding='utf-8').read())
    if locked['selection_uses_may_labels'] or locked['selection_uses_internal_test_labels']:
        raise ValueError('Selection lock violates validation-only model choice')
    training = load_training()
    current_protocol = protocol(training)
    for key in ['target_source_sha256','validation_origin','test_origin','may_origin','fixed_forecast_path_rows']:
        if locked['protocol'][key]!=current_protocol[key]:
            raise ValueError(f'Frozen protocol differs from current data: {key}')
    origin = training.index[-1]+pd.Timedelta(hours=1)
    future = future_index(origin)
    results_path = OUT/'arima_xgb_results.json'
    results = json.loads(results_path.read_text(encoding='utf-8'))
    methods = {'arima':predict_arima,'xgboost':predict_xgb}
    paths = {}
    for name,config in locked['configs'].items():
        started = time.monotonic()
        kwargs = {'n_jobs':args.n_jobs} if name=='xgboost' else {}
        extension = '.pkl' if name=='arima' else '.json'
        kwargs['save_path'] = OUT/'saved_models'/f'final_{name}{extension}'
        kwargs['metadata_path'] = OUT/'saved_models'/f'final_{name}_metadata.json'
        paths[name],fit_meta = methods[name](training,future,config,**kwargs)
        results['models'][name]['may_full_history_fit'] = fit_meta
        results['models'][name]['may_refit_runtime_seconds'] = time.monotonic()-started
    for baseline in ['mean','median','weekday_hour_mean']:
        paths['baseline_'+baseline] = baseline_predict(baseline,training,future)
    # Every full bridge prediction is stored before opening May labels.
    for name,path in paths.items():
        if len(path)!=745 or not np.isfinite(path).all():
            raise ValueError('Invalid fixed 745-hour forecast path')
        prediction_frame(name,'may_full_bridge',origin,future,np.repeat(np.nan,745),path).to_csv(
            OUT/f'{name}_may_full_bridge.csv',index=False,encoding='utf-8-sig')
    if args.save_only:
        results['may_evaluation_status'] = 'Final models and bridge paths saved and reload-verified; May labels were not read by this save-only run'
        write_json(results_path,results)
        print(json.dumps({name:results['models'][name]['may_full_history_fit']['saved_model'] for name in locked['configs']},indent=2),flush=True)
        return
    labels = may_labels_after_lock()
    frames = []
    for name,path in paths.items():
        predicted = pd.Series(path,index=future).reindex(labels.index)
        if predicted.isna().any():
            raise ValueError('May labelled timestamps are outside the frozen bridge')
        results['models'][name]['may_505_labelled_hours'] = score(labels.to_numpy(),predicted.to_numpy())
        frames.append(prediction_frame(name,'may_labelled',origin,labels.index,labels.to_numpy(),predicted.to_numpy()))
    pd.concat(frames,ignore_index=True).to_csv(OUT/'may_labelled_predictions.csv',index=False,encoding='utf-8-sig')
    results['may_evaluation'] = {'selection_lock':args.lock,'configs_changed':False,
                               'full_path_rows':745,'labelled_evaluation_rows':505,
                               'forecast_origin':str(origin),'labelled_start':str(labels.index[0]),'labelled_end':str(labels.index[-1]),
                               'first_offset':int((labels.index[0]-origin).total_seconds()/3600),
                               'last_offset':int((labels.index[-1]-origin).total_seconds()/3600),
                               'honest_holdout_disclosure':current_protocol['honest_holdout_disclosure']}
    write_json(results_path,results)
    print(json.dumps({name:results['models'][name]['may_505_labelled_hours'] for name in paths},indent=2),flush=True)

if __name__ == '__main__':
    main()
