"""Reconcile all secondary long CSVs and standalone saved-context inference."""
import sys,json
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from retrain_classical import OUT,metrics,write_json,sha
from predict_classical import forecast_saved
def main():
    base=OUT/'long745';results=json.loads((base/'results.json').read_text(encoding='utf-8'));checks=[]
    for name,m in results['models'].items():
        for split,key in [('validation','validation_metrics'),('test','internal_test_metrics')]:
            f=pd.read_csv(m['paths'][split+'_csv'],float_precision='round_trip');assert len(f)==745
            assert f.origin.nunique()==1 and np.array_equal(f.lead_hours,np.arange(1,746))
            diff=[]
            for label,sub in [('path_P745',f),('late_P505',f.iloc[240:])]:
                actual=metrics(sub.actual_kwh,sub.predicted_kwh)
                diff += [abs(v-m[key][label][k]) for k,v in actual.items() if v is not None]
            error=max(diff);assert error<1e-7
            checks.append({'model':name,'split':split,'points':745,'metric_max_absdiff':error})
        saved=pd.read_csv(m['paths']['future_csv'],float_precision='round_trip');fresh=forecast_saved(name,'long745')
        error=float(np.max(np.abs(saved.predicted_kwh-fresh.predicted_kwh)));assert error<=1e-8
        assert saved.actual_kwh.isna().all() and saved.id.iloc[0]=='2025-04-30 00:00:00' and saved.id.iloc[-1]=='2025-05-31 00:00:00'
        checks.append({'model':name,'saved_context_reload_max_absdiff':error,'future_points':745,'future_actual_all_nan':True})
    write_json(base/'verification.json',{'status':'passed','checks':checks,'selection_lock_sha256':sha(base/'selection_lock.json'),'May_labels_read':False})
    print(json.dumps({'status':'passed','verification':str(base/'verification.json')}))
if __name__=='__main__':main()
