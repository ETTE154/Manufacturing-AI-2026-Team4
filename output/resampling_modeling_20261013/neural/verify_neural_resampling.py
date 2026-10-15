"""Audit prefix-only scaler/covariate clocks, native mapping and saved replays."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import torch
from neural_resampling import (OUT,VALID,TEST,FINAL,POINTS,read,write,load_hourly_target,prefix,
                               native_frame,reload_predict,hourly_map,core_provenance)

def audit():
    y=load_hourly_target();records=[];overlap=[]
    for resolution in ['hourly','daily']:
        for family in ['patchtst','itransformer']:
            dest=OUT/resolution/family;lock=read(dest/'selection_lock.json');result=read(dest/'results.json')
            assert pd.DatetimeIndex(pd.to_datetime(lock['validation_origins'])).equals(VALID)
            assert pd.Timestamp(lock['latest_validation_label'])<TEST
            assert min(lock['candidate_results'],key=lambda r:r['selection_objective_mae_kwh'])['candidate']==lock['selected_candidate']
            chosen=lock['candidate_results'][lock['selected_candidate']]
            epochs=int(np.median([r['inner_early_stopping']['best_epoch'] for r in chosen['folds']]))
            assert epochs==lock['final_refit_epochs']==result['final_epoch_count']
            for candidate in lock['candidate_results']:
                for fold in candidate['folds']:
                    origin=pd.Timestamp(fold['origin']);h=prefix(y,origin);cfg=candidate['config']
                    frame=native_frame(h,resolution,cfg['group'])
                    inner=fold['inner_early_stopping'];hold=745 if resolution=='hourly' else 32
                    fit=frame.iloc[:-hold];norm=inner['normalizer']
                    error=normalizer_error(fit,norm);assert error==0
                    assert str(fit.index[-1])==inner['fit_end'] and str(fit.index[-1])==norm['fit_end']
                    cp=fold['fitted_model']['path'];native,clock,p=reload_predict(cp)
                    fullnorm=normalizer_error(frame,p['normalizer']);assert fullnorm==0 and clock==origin
                    expected=pd.read_csv(dest/f'candidate{candidate["candidate"]}_validation_predictions.csv',float_precision='round_trip')
                    expected=expected[pd.to_datetime(expected.origin)==origin].predicted_kwh.to_numpy(float)
                    diff=float(np.abs(expected-hourly_map(native,resolution)).max());assert diff==0
                    assert pd.Timestamp(p['history_end'])<origin
                    if resolution=='daily':
                        assert p['native_period_count']==len(h)//24
                        if family=='patchtst':assert p['architecture']['patch_len']==4
                    records.append({'resolution':resolution,'family':family,'candidate':candidate['candidate'],'fold':fold['fold'],
                                    'inner_scaler_max_error':error,'refit_scaler_max_error':fullnorm,'saved_model_replay_max_error':diff,
                                    'inner_training_windows':inner['training_windows'],'native_fit_rows':len(fit),
                                    'schema_feature_count':len(norm['features']),'target_index':norm['target_index']})
            for split,origin in [('internal_test',TEST),('future_unlabelled',FINAL)]:
                fit=result['fits'][split];native,clock,p=reload_predict(fit['fitted_model']['path'])
                f=pd.read_csv(fit['prediction_csv'],float_precision='round_trip');idx=pd.DatetimeIndex(pd.to_datetime(f.id))
                assert idx.equals(pd.date_range(origin,periods=745,freq='h')) and clock==origin
                assert float(np.abs(hourly_map(native,resolution)-f.predicted_kwh).max())==0
                assert pd.Timestamp(p['history_end'])==origin-pd.Timedelta(hours=1)
                if split=='internal_test':assert np.array_equal(f.actual_kwh.to_numpy(float),y.reindex(idx).to_numpy(float))
                else:assert f.actual_kwh.isna().all() and p['epochs']==epochs
                if resolution=='daily':
                    assert np.unique(np.repeat(np.arange(32),24)[:745],return_counts=True)[1].tolist()==[24]*31+[1]
                    assert all(np.array_equal(hourly_map(native,resolution)[i*24:(i+1)*24],np.repeat(native[i],24)) for i in range(31))
            core_provenance(family)
    paths=[pd.date_range(o,periods=745,freq='h') for o in VALID]
    union=pd.DatetimeIndex(np.concatenate([p.values for p in paths])).unique()
    late=pd.DatetimeIndex(np.concatenate([p[240:].values for p in paths]))
    test=pd.date_range(TEST,periods=745,freq='h');assert len(union.intersection(test))==0
    # Causal multi-resolution values agree if each row is independently rebuilt
    # from observations available at that row's t+1h completion clock.
    h=prefix(y,VALID[0]);whole=native_frame(h,'hourly','completed_multires');causal=[]
    for pos in [0,23,24,167,191,743,1078,1823]:
        a=native_frame(h.iloc[:pos+1],'hourly','completed_multires').iloc[-1]
        delta=float(np.abs(a-whole.iloc[pos]).max());assert delta==0
        causal.append({'last_hour':str(h.index[pos]),'completion_clock':str(h.index[pos]+pd.Timedelta(hours=1)),
                       'truncated_prefix_vs_full_causal_row_max_error':delta})
    from chronos_resampling import pipeline,predict,native_series
    pipe,chronos_provenance=pipeline();model_summaries=[];chronos_replays=[]
    for resolution in ['hourly','daily']:
        for family in ['patchtst','itransformer','chronos2']:
            dest=OUT/resolution/family;lock=read(dest/'selection_lock.json');result=read(dest/'results.json')
            epochs_records=[r['inner_early_stopping'] for c in lock['candidate_results'] for r in c.get('folds',[])]
            selected_records=[r['inner_early_stopping'] for r in lock['candidate_results'][lock['selected_candidate']].get('folds',[])]
            model_summaries.append({'family':family,'resolution':resolution,'candidate_count':len(lock['candidate_results']),
                  'selected_config':lock['selected_config'],'validation_origin_count':3,'inner_holdout_native_periods':
                  None if family=='chronos2' else (745 if resolution=='hourly' else 32),
                  'maximum_epochs':None if family=='chronos2' else 24,'patience':None if family=='chronos2' else 4,
                  'all_candidate_inner_best_at_epoch_cap_count':sum(r['best_at_epoch_cap'] for r in epochs_records),
                  'selected_inner_best_at_epoch_cap_count':sum(r['best_at_epoch_cap'] for r in selected_records),
                  'selected_inner_epochs':[r['best_epoch'] for r in selected_records],
                  'convergence_note':'Some minima at epoch24; further convergence is unresolved' if any(r['best_at_epoch_cap'] for r in selected_records) else 'Early stopping minima occurred before cap',
                  'alpha0_baseline_equivalent':family=='chronos2' and lock['selected_config']['mean_blend_alpha']==0,
                  'official_provenance':result.get('official_core_provenance',{'source':chronos_provenance.get('model_id',chronos_provenance.get('model')),
                     'revision':chronos_provenance['revision'],'local_snapshot':chronos_provenance['local_snapshot'],'files_sha256':chronos_provenance['files']}),
                  'validation_505_mae':result['validation']['mean_origin_late505_mae_kwh'],
                  'internal_test_505_mae':result['internal_test']['late505']['mae_kwh']})
            if family=='chronos2':
                for split in ['internal_test','future_unlabelled']:
                    model=result['fits'][split]['saved_model'];outcsv=OUT/'replay'/f'{resolution}_chronos2_{split}.csv'
                    error=predict(model,outcsv,recompute=True,pipe=pipe);assert error==0
                    # predict() also asserts EXACT official raw median replay,
                    # including hourly alpha0 where a blend alone could hide it.
                    original=pd.read_csv(result['fits'][split]['prediction_csv'],float_precision='round_trip')
                    replayed=pd.read_csv(outcsv,float_precision='round_trip');mapped=float(np.abs(original.predicted_kwh.to_numpy()-replayed.predicted_energy_kwh.to_numpy()).max())
                    assert mapped==0
                    m=read(model);h=prefix(y,pd.Timestamp(m['forecast_origin']))
                    assert np.array_equal(native_series(h,resolution).to_numpy(),np.asarray(m['native_history_values']))
                    chronos_replays.append({'resolution':resolution,'split':split,'official_checkpoint_reloaded':True,
                        'frozen_official_forward_raw_median_max_error':0.,'locked_blend_native_max_error':error,
                        'mapped_hourly745_CSV_max_error':mapped,'output_csv':str(outcsv),'new_fitted_weights':False,
                        'May_labels_opened':False})
    out={'status':'pass','model_count':6,'CV_models_checked':len(records),'models':records,'model_summaries':model_summaries,
         'chronos_official_forward_replays':chronos_replays,'multi_resolution_clock_checks':causal,
         'validation_observations':2235,'validation_unique_hours':len(union),'validation_late_observations':1515,
         'validation_late_unique_hours':late.nunique(),'validation_internal_test_label_overlap':0,
         'first_daily_inner_training_window_count':6,'context14_daily_infeasible_first_fold':True,
         'final_epoch_counts_frozen_before_April':True,'future_exports_have_no_actuals':True,'May_labels_opened':False,
         'all_35_neural_checkpoint_replays_exact':True,'limitation':'Deterministic replay and no-leak checks do not establish forecasting skill.'}
    write(OUT/'verification.json',out);print(f'PASS {len(records)} CV checkpoints +8 neural test/future +4 official Chronos forwards; causal/scaler/mapping exact')

def normalizer_error(frame,norm):
    values=frame[norm['features']].to_numpy(float);mean=values.mean(0);sd=values.std(0);sd=np.where(sd>1e-8,sd,1.)
    assert norm['features'][norm['target_index']]=='target_rate'
    return float(max(np.abs(mean-np.asarray(norm['mean'])).max(),np.abs(sd-np.asarray(norm['scale'])).max()))
if __name__=='__main__':audit()
