"""Replay any frozen experiment model without reading evaluation labels.

This CLI reproduces the saved Apr30 forecast origin; it is not a rolling
production retraining service. Chronos recomputes official weights by default.
"""
from pathlib import Path
import sys,json,argparse,importlib.util
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import joblib
OUT=Path(__file__).resolve().parent

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',default='selected');parser.add_argument('--output',required=True)
    parser.add_argument('--cached-chronos',action='store_true',help='Replay cached raw official output instead of another checkpoint forward')
    args=parser.parse_args();dest=Path(args.output).resolve()
    if OUT not in dest.parents:raise ValueError('Output must be inside this experiment directory')
    dest.parent.mkdir(parents=True,exist_ok=True)
    lock=json.loads((OUT/'global_selection_lock.json').read_text(encoding='utf8'));key=lock['selected_model'] if args.model=='selected' else args.model
    if key not in [r['model']for r in lock['rankings']]:raise ValueError('Unknown frozen model')
    replay_dir=OUT/'replay';replay_dir.mkdir(exist_ok=True)
    if key in ['hourly_xgboost','hourly_ridge','hourly_arima','hourly_baseline']:
        sys.path.insert(0,str(OUT/'hourly'));m=module('saved_hourly',OUT/'hourly/predict_hourly.py');f=m.forecast_saved(key.removeprefix('hourly_'));idx=pd.to_datetime(f.id);pred=f.predicted_kwh.to_numpy(float)
    elif key.endswith(('_patchtst','_itransformer')):
        sys.path.insert(0,str(OUT));sys.path.insert(0,str(OUT/'neural'));m=module('saved_neural',OUT/'neural/neural_resampling.py')
        grain,family=key.split('_',1);native,origin,meta=m.reload_predict(OUT/'neural'/grain/family/'saved_models/future_unlabelled.pt')
        idx=pd.date_range(origin,periods=745,freq='h');pred=m.hourly_map(native,meta['resolution'])
    elif key.endswith('_chronos2'):
        sys.path.insert(0,str(OUT));sys.path.insert(0,str(OUT/'neural'));m=module('saved_chronos',OUT/'neural/chronos_resampling.py')
        grain=key.split('_')[0];raw=replay_dir/(key+'_native_replay.csv');m.predict(OUT/'neural'/grain/'chronos2/saved_models/future_unlabelled.json',raw,recompute=not args.cached_chronos)
        f=pd.read_csv(raw,float_precision='round_trip');idx=pd.to_datetime(f.timestamp);pred=f.predicted_energy_kwh.to_numpy(float)
    else:
        m=module('saved_coarse',OUT/'coarse/coarse_models.py');bundle=joblib.load(OUT/'coarse'/key/'saved_model.joblib');idx,pred,_=m.forecast(bundle)
    f=pd.DataFrame(dict(id=idx,predicted_energy_kwh=pred,predicted_bill_won=np.asarray(pred)*180,predicted_carbon_kg=np.asarray(pred)*.424))
    assert len(f)==745 and np.isfinite(f.predicted_energy_kwh).all()
    f.to_csv(dest,index=False,encoding='utf-8-sig',float_format='%.17g')
    g=f[f.id.between('2025-05-10','2025-05-31 00:00')].copy();assert len(g)==505
    g=g.rename(columns={'predicted_energy_kwh':'hourly_pow','predicted_bill_won':'may_bill','predicted_carbon_kg':'may_carbon'});g['agg_pow']=g.hourly_pow.sum()
    guide=dest.with_name(dest.stem+'_guide505.csv');g.to_csv(guide,index=False,encoding='utf-8-sig',float_format='%.17g')
    print(json.dumps(dict(model=key,forecast_points=745,guide_points=505,output=str(dest),guide_output=str(guide),May_labels_read=False,origin='2025-04-30'),ensure_ascii=True))
if __name__=='__main__':main()
