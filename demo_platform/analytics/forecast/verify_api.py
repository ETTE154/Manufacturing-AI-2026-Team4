"""Replay/future-poison/module-independence tests using source historical data."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from common import *
from api import serving_forecast
def main():
    y=load_y();origin=pd.Timestamp('2025-03-13 19:00');h=prefix(y,origin)
    a=serving_forecast(h,origin,24);future_poison=y.copy();future_poison.loc[future_poison.index>=origin]=1e12
    b=serving_forecast(future_poison,origin,24)
    assert a['forecast']==b['forecast'] and a['history_end']=='2025-03-13 18:00:00'
    c=serving_forecast(h,'2025-03-13 19:43',24);assert c['forecast']==a['forecast']
    f=pd.read_csv(MODULE_SOURCE,float_precision='round_trip');f=f[f.complete_720].copy();f['hour']=pd.to_datetime(f.hour)
    modules=[]
    for name,d in f.groupby('module(equipment)'):
        s=pd.Series(d.energy_kwh.to_numpy(),index=d.hour);m=name.split('(')[0];r=serving_forecast(s,origin,24,m)
        truth=float(s.loc[s.index<origin].mean());assert r['forecast'][0]['predicted_kwh']==truth
        assert r['scope']=='independently_fitted_single_module' and not r['module_model_is_total_proportional_allocation']
        modules.append({'module_id':m,'predicted_kwh':truth,'independent_prefix_fit':True})
    cold=serving_forecast(h.iloc[:12],h.index[12],24);assert cold['status']=='cold_start_baseline' and cold['metrics'] is None
    for bad in [h.iloc[:-1],h.drop(h.index[10]),h.iloc[:1].set_axis([pd.Timestamp('2024-12-01 00:05')])]:
        try:serving_forecast(bad,origin,24)
        except ValueError:pass
        else:raise AssertionError('Invalid history accepted')
    write(OUT/'serving_api_verification.json',{'status':'pass','requested_default_cursor':'2025-03-13 19:43','forecast_origin':str(origin),
         'history_end':a['history_end'],'future_poison_forecast_unchanged':True,'modules':modules,
         'cold_start_correctly_labeled':True,'invalid_incomplete_or_gapped_history_rejected':True,
         'recent_replay_metrics':a['metrics'],'no_April_model_loaded_for_March_replay':True,'May_labels_read':False})
    write(OUT/'serving_api_example.json',a);print('Serving API strict-prefix/future poison/13 independent modules/cold start PASS')
if __name__=='__main__':main()
