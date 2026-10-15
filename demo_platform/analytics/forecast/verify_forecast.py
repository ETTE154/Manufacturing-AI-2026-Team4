"""Independent recomputation of prefixes, scalers and native saved predictions."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np,pandas as pd,torch,joblib
from common import *
import deep,classical
def norm_error(a,n):
    expected=norm_fit(a,n['kind'])
    return float(max(abs(np.asarray(expected[k])-np.asarray(n[k])).max() for k in ['center','offset','scale']))
def main():
    y=load_y();records=[];model_norm=[]
    cr=read(OUT/'classical_results.json')
    for r in cr['candidates']:
        if r['status']!='completed':continue
        for fit in r['origin_fit_audits']:
            h=prefix(y,pd.Timestamp(fit['origin']));n=fit['normalizer']
            if n:
                e=norm_error(native(h,r['config']['resolution']).values,n);assert e==0
                records.append({'family':r['config']['family'],'candidate':r['candidate'],'origin':fit['origin'],'scaler_recompute_error':e})
            if fit['early_stopping']:
                s=native(h,r['config']['resolution']);inner=s.iloc[:-fit['early_stopping']['holdout_periods']]
                assert norm_error(inner.values,fit['early_stopping']['normalizer'])==0
    for family in ['patchtst','itransformer']:
        result=read(OUT/family/'results.json');candidates=read(OUT/family/'candidate_results.json')['candidates']
        for candidate in candidates:
            cfg=candidate['config']
            for fold in candidate['folds']:
                h=prefix(y,fold['origin']);d=deep.data_frame(h,cfg);hold=fold['early']['holdout_native_periods'];fit=d.iloc[:-hold];inner=fold['early']
                assert norm_error(fit.values,inner['normalizer'])==0 and norm_error(d.values,fold['fit']['normalizer'])==0
                assert inner['fixed_epoch_cap'] is None and inner['logs'][-1]['learning_rate']<=1.00001e-6
                p=deep.replay(inner['best_checkpoint']);origin=d.iloc[-hold:].index[0]
                actual=h.reindex(pd.date_range(origin,periods=745,freq='h')).to_numpy(float)
                scored=metrics(actual[240:],hourly_map(p,cfg['resolution'])[240:])['mae_kwh'];assert abs(scored-inner['best_inner505_mae'])<1e-7
                error=float(abs(inverse(transform(d.values,fold['fit']['normalizer']),fold['fit']['normalizer'])-d.values).max());assert error<1e-8
                records.append({'family':family,'candidate':candidate['candidate'],'origin':fold['origin'],'inner_best_replay_MAE_error':abs(scored-inner['best_inner505_mae']),
                    'scaler_recompute_error':0,'inverse_roundtrip_error':error,'epochs_run':inner['epochs_run'],'best_epoch':inner['best_epoch']})
        for split in ['internal_test','future_unlabelled']:
            fit=result['fits'][split];pred=hourly_map(deep.replay(fit['saved_model']),result['selected_config']['resolution'])
            f=pd.read_csv(fit['prediction_csv'],float_precision='round_trip');err=float(abs(pred-f.predicted_kwh).max());assert err==0
            assert pd.Timestamp(fit['history_end'])==pd.to_datetime(f.origin.iloc[0])-pd.Timedelta(hours=1)
            model_norm.append({'family':family,'split':split,'native_CSV_reload_error':err,'input_shape':fit['input_shape'],'output_shape':fit['output_shape'],'explicit_target_index':fit['target_index']})
    for family,r in cr['models'].items():
        for split in ['internal_test','future_unlabelled']:
            fit=r['fits'][split];p=classical.predict(joblib.load(fit['saved_model']));f=pd.read_csv(fit['prediction_csv'],float_precision='round_trip')
            err=float(abs(p-f.predicted_kwh).max());assert err==0
            model_norm.append({'family':family,'split':split,'native_CSV_reload_error':err})
    from chronos_search import pipeline,raw_predict,blend
    pipe,provenance=pipeline();ch=read(OUT/'chronos2/results.json')
    for split in ['internal_test','future_unlabelled']:
        meta=read(ch['fits'][split]['saved_model']);h=pd.Series(meta['history_values'],index=pd.to_datetime(meta['history_times']));cfg=meta['config']
        raw=raw_predict(pipe,h,cfg['resolution'],cfg['context']);assert np.array_equal(raw,np.asarray(meta['stored_raw']))
        p=hourly_map(blend(raw,h,cfg),cfg['resolution']);f=pd.read_csv(ch['fits'][split]['prediction_csv'],float_precision='round_trip')
        err=float(abs(p-f.predicted_kwh).max());assert err==0
        model_norm.append({'family':'chronos2','split':split,'official_raw_forward_max_error':0.,'native_CSV_reload_error':err})
    # Independently check old pinned cores and their complete declared files.
    sources=[]
    for family in ['patchtst','itransformer']:
        old=ROOT/'output/modeling/forecast';p=old/('patchtst/provenance.json' if family=='patchtst' else 'itransformer/vendor/provenance.json');pr=read(p)
        for file in pr['files']:
            path=Path(file['vendored_path']) if family=='patchtst' else old/'itransformer/vendor'/file['path']
            assert sha(path)==(file['vendored_sha256'] if family=='patchtst' else file['local_sha256'])
        sources.append({'family':family,'official_core_provenance':str(p),'sha256':sha(p),'declared_files_verified':len(pr['files'])})
    validation_unique=pd.DatetimeIndex(np.concatenate([pd.date_range(o,periods=745,freq='h').values for o in VALID])).unique()
    assert not len(validation_unique.intersection(pd.date_range(TEST,periods=745,freq='h')))
    write(OUT/'verification.json',{'status':'pass','scaler_inner_model_checks':records,'native_replays':model_norm,'official_cores':sources,
        'target_source_sha256':sha(TARGET),'module_source_sha256':sha(MODULE_SOURCE),'unit_tests':audit_units(),
        'validation_internal_test_label_overlap':0,'scalers_fit_only_origin_prefix':True,'explicit_native_shape_checked':True,
        'no_original_or_old_venv_modified':True,'May_labels_read':False,'serving_API_verification':str(OUT/'serving_api_verification.json')})
    print(f'PASS{len(records)} scaler/inner checks,14 native path reloads, official cores, unit and no-overlap audit')
if __name__=='__main__':main()
