"""Small secondary745-hour context-only XGB/Ridge experiment; never reads May."""
from pathlib import Path
import sys,json,argparse,time
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from retrain_classical import (OUT,REV,load_hourly,prefix,select_sensor_features,fit_model,predict,
                              save_and_check,idx_future,metrics,prediction_frame,write_json,sha,TARGET)
LONG=OUT/'long745'
VALID=pd.Timestamp('2025-02-25');TEST=pd.Timestamp('2025-03-29')
CONFIGS=[{'model':'xgboost','group':group,'objective':'reg:absoluteerror','max_depth':2,'n_estimators':80,
          'learning_rate':.02,'min_child_weight':100,'reg_lambda':100.,'residual_shrinkage':.5}
         for group in ['target_aligned_calendar','selected_sensor_calendar']]+[
         {'model':'ridge','group':group,'alpha':alpha} for group,alpha in
         [('target_aligned_calendar',1e5),('selected_sensor_calendar',1e6)]]

def stat(a,p):return {'path_P745':metrics(a,p),'late_P505':metrics(np.asarray(a)[240:],np.asarray(p)[240:])}
def main():
    frame=load_hourly();LONG.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    h=prefix(frame,VALID);selected,audit=select_sensor_features(h,max_features=12)
    write_json(LONG/'feature_selection_validation.json',audit)
    write_json(LONG/'search_plan.json',{'candidates':CONFIGS,'count':4,'forecast_points':745,
          'selection':'SingleFeb25P745MAE only; March29 internal and May excluded.',
          'sensor_availability':'All inputs are static past context plus known future calendar. Daily/weekly analogues phase-wrap the last observed day/week, not future raw lags.',
          'sparse_training_leads':'Offsets0..24 and48..744 by24; terminal origins may have partial labels, all strictly inside fitting prefix.',
          'data_sha256':sha(REV/'data/hourly_features.csv')})
    records=[];best={};cache={};idx=idx_future(h,745);actual=frame.loc[idx,TARGET].to_numpy(dtype=float)
    for candidate,cfg in enumerate(CONFIGS,1):
        model=fit_model(h,745,cfg,selected,cache);p=predict(model,h,745);stats=stat(actual,p);value=stats['path_P745']['mae_kwh']
        record={'candidate':candidate,'config':cfg,'validation_metrics':stats,'objective_mae_kwh':value};records.append(record)
        if cfg['model'] not in best or value<best[cfg['model']]['objective_mae_kwh']:
            best[cfg['model']]=record;dest=LONG/cfg['model'];dest.mkdir(parents=True,exist_ok=True)
            prediction_frame(cfg['model'],'long745','validation',0,VALID,idx,actual,p).to_csv(dest/'validation_predictions.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
        print(f'long candidate{candidate}/4 {cfg["model"]}/{cfg["group"]}: {value:.9f}',flush=True)
    lock={'models':best,'records':records,'validation_origin':str(VALID),'selected_model':min(best,key=lambda x:best[x]['objective_mae_kwh']),
          'feature_selection_validation':str(LONG/'feature_selection_validation.json'),'configuration_count':4,
          'data_sha256':sha(REV/'data/hourly_features.csv'),'selector_sha256':sha(REV/'feature_common.py'),
          'May_labels_read':False,'internal_test_used_for_selection':False}
    write_json(LONG/'selection_lock.json',lock)
    testh=prefix(frame,TEST);test_selected,testaudit=select_sensor_features(testh,max_features=12)
    write_json(LONG/'feature_selection_internal_test.json',testaudit)
    final_selected,finalaudit=select_sensor_features(prefix(frame,'2025-04-01'),max_features=12)
    write_json(LONG/'feature_selection_final_preApril.json',finalaudit)
    models={}
    for family,chosen in best.items():
        cfg=chosen['config'];dest=LONG/family;fit=fit_model(testh,745,cfg,test_selected,cache)
        testidx=idx_future(testh,745);a=frame.loc[testidx,TARGET].to_numpy(dtype=float);p=predict(fit,testh,745)
        test_stats=stat(a,p)
        prediction_frame(family,'long745','internal_test',0,TEST,testidx,a,p).to_csv(dest/'test_predictions.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
        final=fit_model(frame,745,cfg,final_selected,cache);future,info=save_and_check(final,frame,745,dest/'saved_model')
        futidx=idx_future(frame,745)
        prediction_frame(family,'long745','future_unlabelled',0,futidx[0],futidx,np.full(745,np.nan),future).to_csv(dest/'future_path.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
        models[family]={'selected_config':cfg,'trials_count':2,'validation_metrics':chosen['validation_metrics'],
              'internal_test_metrics':test_stats,'saved_model':info,'paths':{'validation_csv':str(dest/'validation_predictions.csv'),
              'test_csv':str(dest/'test_predictions.csv'),'future_csv':str(dest/'future_path.csv')},
              'input_scope':'Aggregate/module history, optional training-screened sensor history; known calendar. Historical analogues phase-wrap at fixed origin; no future raw observations.',
              'feature_selection_paths':{'validation':str(LONG/'feature_selection_validation.json'),'internal_test':str(LONG/'feature_selection_internal_test.json'),
              'final_frozen_preApril':str(LONG/'feature_selection_final_preApril.json')},'final_sensor_columns':final_selected}
        write_json(dest/'results.json',models[family]);print(f'{family} internal MAE {test_stats["path_P745"]["mae_kwh"]:.9f}',flush=True)
    result={'scenario':'long745','models':models,'selected_model':lock['selected_model'],'selection_lock':str(LONG/'selection_lock.json'),
            'selection_lock_sha256':sha(LONG/'selection_lock.json'),'May_labels_read':False,'May_score_status':'awaiting_parent_global_lock',
            'runtime_seconds':time.monotonic()-started,'limitations':['Only one validation and one internal-test origin; no many-origin reliability evidence.',
                    'The lastobserved day/week phase analogue is an explicit feature assumption and differs from true future24/168h lags.',
                    'May745path has505 labels at offsets240..744; future physical sensors are unavailable. No rollingMay or label feedback.']}
    write_json(LONG/'results.json',result)
    day=json.loads((OUT/'results.json').read_text(encoding='utf-8'));day['secondary_long745']=result
    write_json(OUT/'results.json',day)
    print('Long context models frozen, saved and reloaded; May not read.',flush=True)
if __name__=='__main__':main()
