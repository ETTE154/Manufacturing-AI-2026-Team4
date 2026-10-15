"""One predeclared loss-only companion to the already selected30-trial XGB.

All prior intermediate files are preserved. Only objective changes; comparison
uses the original validation protocol, not the already visible April test.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import argparse
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from opt_common import (OUT,DAY_FOLD_STARTS,APRIL_TEST_ORIGINS,LONG_VALID_ORIGIN,LONG_TEST_ORIGIN,
                        require_source_approval,load_target,history,future,actual,metrics,
                        day_summary,prediction_frame,write_json)
from classical_opt import fit_model,predict,save_model,evaluate_day

def extend(scenario):
    require_source_approval();y=load_target();directory=OUT/scenario/'xgboost'
    prior=json.loads((directory/'selection_lock.json').read_text(encoding='utf-8'))
    if prior['configuration_count']!=30:raise ValueError('Extension requires exactly30 previously fixed trials')
    archive=directory/'intermediate_30_trials'
    if archive.exists():raise ValueError('Original30-trial archive already exists; do not duplicate extension')
    archive.mkdir()
    for filename in ['selection_lock.json','results.json','validation_predictions.csv','test_predictions.csv','future_path.csv']:
        shutil.copy2(directory/filename,archive/filename)
    shutil.copytree(directory/'saved_model',archive/'saved_model')
    companion=dict(prior['selected_config']);companion['objective']='reg:absoluteerror'
    declaration={'candidate':30,'total_attempted_configs':31,'previous_selected_config':prior['selected_config'],
                 'companion_config':companion,'only_changed_parameter':'objective=reg:absoluteerror',
                 'previous_selection_lock_sha256':hashlib.sha256((archive/'selection_lock.json').read_bytes()).hexdigest(),
                 'selection_uses_April_test':False,'selection_uses_May_labels':False,
                 'reason':'Parent request to test loss-objective alignment with MAE; no other hyperparameter changes based on intermediate April metrics',
                 'disclosure':'Intermediate30-trial April results already existed; this is a declared goal-alignment extension with validation-only selection, not a pristine never-viewed test'}
    write_json(directory/'loss_extension_declaration.json',declaration)
    cache={}
    if scenario=='dayahead24':
        frames=[];folds=[]
        for fold,start in enumerate(DAY_FOLD_STARTS,1):
            frame,stat=evaluate_day('xgboost',companion,y,pd.date_range(start,periods=14,freq='D'),'validation',fold,start,cache)
            frames.append(frame);folds.append(stat)
        companion_frame=pd.concat(frames,ignore_index=True);stats=day_summary(companion_frame);objective=stats['mean_origin_path_mae_kwh']
    else:
        h=history(y,LONG_VALID_ORIGIN);idx=future(LONG_VALID_ORIGIN,745);model=fit_model('xgboost',h,745,companion,cache)
        pred=predict(model,h,idx);a=actual(y,idx);companion_frame=prediction_frame('xgboost',scenario,'validation',0,LONG_VALID_ORIGIN,idx,a,pred)
        stats={'path_P745':metrics(a,pred),'late_P505':metrics(a[240:745],pred[240:745])};objective=stats['path_P745']['mae_kwh'];folds=[]
    companion_frame.to_csv(directory/'loss_companion_validation_predictions.csv',index=False,encoding='utf-8-sig')
    record={'candidate':30,'config':companion,'objective_mae_kwh':objective,'validation':stats,'fold_validation':folds,
            'loss_alignment_extension':True}
    records=prior['search_records']+[record]
    chosen=min((r for r in records if 'objective_mae_kwh' in r),key=lambda r:r['objective_mae_kwh'])
    cfg=chosen['config'];companion_won=chosen['candidate']==30
    locked={**prior,'selected_config':cfg,'objective':chosen['objective_mae_kwh'],'selected_candidate':chosen['candidate'],
            'configuration_count':31,'search_records':records,'loss_extension_declaration':str(directory/'loss_extension_declaration.json'),
            'companion_won_validation':companion_won,'previous30_preserved':str(archive)}
    write_json(directory/'selection_lock.json',locked)
    selected_frame=companion_frame if companion_won else pd.read_csv(archive/'validation_predictions.csv')
    selected_frame.to_csv(directory/'validation_predictions.csv',index=False,encoding='utf-8-sig')
    if scenario=='dayahead24':
        test,test_stats=evaluate_day('xgboost',cfg,y,APRIL_TEST_ORIGINS,'April_test',0,pd.Timestamp('2025-04-01'),cache)
        points=24
    else:
        h=history(y,LONG_TEST_ORIGIN);idx=future(LONG_TEST_ORIGIN,745);model=fit_model('xgboost',h,745,cfg,cache)
        pred=predict(model,h,idx);a=actual(y,idx);test=prediction_frame('xgboost',scenario,'internal_test',0,LONG_TEST_ORIGIN,idx,a,pred)
        test_stats={'path_P745':metrics(a,pred),'late_P505':metrics(a[240:745],pred[240:745])};points=745
    test.to_csv(directory/'test_predictions.csv',index=False,encoding='utf-8-sig')
    final=fit_model('xgboost',y,points,cfg,cache);idx=future(y.index[-1]+pd.Timedelta(hours=1),points);pred=predict(final,y,idx)
    info=save_model(final,y,points,directory/'saved_model')
    prediction_frame('xgboost',scenario,'future_unlabelled',0,idx[0],idx,np.repeat(np.nan,points),pred).to_csv(directory/'future_path.csv',index=False,encoding='utf-8-sig')
    result=json.loads((archive/'results.json').read_text(encoding='utf-8'))
    result.update({'selected_config':cfg,'validation':chosen['validation'],'test':test_stats,'saved_model':info,
                   'trials_count':31,'companion_won_validation':companion_won,
                   'loss_extension_declaration':str(directory/'loss_extension_declaration.json'),'May_labels_read':False})
    write_json(directory/'results.json',result)
    print(json.dumps({'scenario':scenario,'previous_validation':prior['objective'],'companion_validation':objective,
                      'companion_won':companion_won,'final_config':cfg,'test':test_stats},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scenario',choices=['dayahead24','long745'],required=True);args=p.parse_args();extend(args.scenario)
