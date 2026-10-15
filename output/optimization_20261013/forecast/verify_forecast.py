"""CSV/metric/origin/lock checks and independent saved-model CLI-equivalent QA."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
from opt_common import (OUT,DAY_FOLD_STARTS,APRIL_TEST_ORIGINS,LONG_VALID_ORIGIN,LONG_TEST_ORIGIN,
                        load_target,future,metrics,day_summary,write_json)
from compare_and_plot import MODEL_IDS,location,load_json,verify_locks
from predict_optimized import forecast_saved

def assert_metrics(expected,observed):
    differences={}
    for key in ['mae_kwh','rmse_kwh','mape_percent','bias_pred_minus_actual_kwh','r2']:
        if key not in expected:continue
        difference=abs(float(expected[key])-float(observed[key]));differences[key]=difference
        if difference>1e-7:raise ValueError(f'CSV metric discrepancy {key}: {difference}')
    if int(expected['count'])!=int(observed['count']):raise ValueError('Metric row count differs')
    return max(differences.values(),default=0.)

def checks_frame(path,scenario,split,y):
    f=pd.read_csv(path);f['id']=pd.to_datetime(f['id']);f['origin']=pd.to_datetime(f['origin'])
    if f.duplicated(['origin','id']).any():raise ValueError('Duplicate forecast origin/target')
    if f.actual_kwh.dtype!='float64' or f.predicted_kwh.dtype!='float64' or f.absolute_error_kwh.dtype!='float64':
        raise ValueError('CSV numeric precision is not float64')
    np.testing.assert_allclose(f.absolute_error_kwh,np.abs(f.actual_kwh-f.predicted_kwh),rtol=0,atol=1e-8)
    if scenario=='dayahead24':
        expected=pd.DatetimeIndex(np.concatenate([future(origin,24).values for origin in
            (pd.DatetimeIndex(np.concatenate([pd.date_range(start,periods=14,freq='D').values for start in DAY_FOLD_STARTS])) if split=='validation' else APRIL_TEST_ORIGINS)]))
        if len(f)!=(1680 if split=='validation' else 696):raise ValueError('Wrong day path count')
        if f.groupby('origin').size().nunique()!=1 or f.groupby('origin').size().iloc[0]!=24:raise ValueError('Wrong per-origin day count')
        if f[f.lead_hours==24].shape[0]!=(70 if split=='validation' else 29):raise ValueError('Wrong day endpoint count')
    else:
        expected=future(LONG_VALID_ORIGIN if split=='validation' else LONG_TEST_ORIGIN,745)
        if len(f)!=745:raise ValueError('Wrong long count')
    if not pd.DatetimeIndex(f.sort_values(['origin','id']).id).equals(expected):raise ValueError('Unexpected target clock')
    actual=y.reindex(f.id).to_numpy(dtype=float)
    np.testing.assert_allclose(f.actual_kwh,actual,rtol=0,atol=1e-8)
    if not np.array_equal(f.lead_hours.to_numpy(),f.horizon_offset_hours.to_numpy()+1):raise ValueError('Lead vs offset mismatch')
    return f

def run(reload_models):
    verify_locks();y=load_target();combined=load_json(OUT/'comparison_results.json');records=[];reload_records=[]
    for scenario in ['dayahead24','long745']:
        for model in MODEL_IDS:
            card=combined['scenarios'][scenario]['models'][model];directory=location(scenario,model)
            maximum=0.
            for split in ['validation','test']:
                f=checks_frame(directory/('validation_predictions.csv' if split=='validation' else 'test_predictions.csv'),scenario,split,y)
                if scenario=='dayahead24':
                    stat=day_summary(f);expected=card['validation_metrics'] if split=='validation' else card['test_metrics']
                    maximum=max(maximum,assert_metrics(expected,stat['path_24_values']))
                    endpoint=card['validation_endpoint_metrics'] if split=='validation' else card['test_endpoint_metrics']
                    maximum=max(maximum,assert_metrics(endpoint,stat['endpoint_lead_24']))
                else:
                    expected=card['validation_metrics'] if split=='validation' else card['internal_test_metrics']
                    maximum=max(maximum,assert_metrics(expected,metrics(f.actual_kwh,f.predicted_kwh)))
            unlabelled=pd.read_csv(directory/'future_path.csv')
            if len(unlabelled)!=(24 if scenario=='dayahead24' else 745) or not unlabelled.actual_kwh.isna().all():raise ValueError('Wrong final unlabeled path')
            if scenario=='long745':
                may=pd.read_csv(directory/'may_predictions.csv')
                if len(may)!=505 or may.horizon_offset_hours.min()!=240 or may.horizon_offset_hours.max()!=744:raise ValueError('Wrong May slice')
                maximum=max(maximum,assert_metrics(card['may_metrics'],metrics(may.actual_kwh,may.predicted_kwh)))
            records.append({'scenario':scenario,'model':model,'CSV_metric_max_difference':maximum,'status':'pass'})
            if reload_models:
                idx,p,verification=forecast_saved(scenario,model)
                if verification['reload_max_abs_diff_kwh']>1e-8:raise ValueError('Saved reload differs')
                reload_records.append(verification)
    for model,paths in combined['figures'].items():
        for key,path in paths.items():
            if not Path(path).is_absolute() or not Path(path).exists() or Path(path).stat().st_size<1000:raise ValueError('Figure missing')
    result={'status':'pass','CSV_checks':records,'saved_model_reload_checks':reload_records,
            'expected_day_validation_origins':70,'expected_day_test_origins':29,'expected_long_rows':745,'expected_May_rows':505,
            'metrics_tolerance':1e-7,'reload_tolerance':1e-8,'all_original_data_and_old_modeling_outputs_untouched':True,
            'May_model_selection_changed':False,'comparison_lock_sha256':hashlib.sha256((OUT/'comparison_selection_lock.json').read_bytes()).hexdigest()}
    write_json(OUT/'verification_results.json',result)
    print(json.dumps({'status':'pass','CSV_metric_max_difference':max(x['CSV_metric_max_difference'] for x in records),
                      'reload_models_checked':len(reload_records),'reload_max_difference':max([x['reload_max_abs_diff_kwh'] for x in reload_records],default=None)},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reload-models',action='store_true');args=p.parse_args();run(args.reload_models)
