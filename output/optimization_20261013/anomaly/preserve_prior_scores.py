"""Restore exact pre-extension score arrays after CSV round-trip and check metrics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent

def main():
    snapshot=np.load(OUT/'prior_seven_score_snapshot.npz')
    changes={}
    for fn in ['original_test_scores.csv.gz','continuous_window_scores.csv.gz','full_candidate_scores.csv.gz','synthetic_test_scores.csv.gz']:
        d=pd.read_csv(OUT/fn,float_precision='round_trip');changed=0
        for key in snapshot.files:
            source,col=key.split('|',1)
            if source!=fn:continue
            old=snapshot[key];assert len(d)==len(old)
            changed+=int(np.count_nonzero(d[col].to_numpy()!=old));d[col]=old
        if changed:d.to_csv(OUT/fn,index=False,compression='gzip',float_format='%.17g')
        changes[fn]=changed
    report=json.loads((OUT/'anomaly_metrics.json').read_text(encoding='utf-8'))
    report['models']['ae_synthetic_denoising']['display_name']='합성 paired-target denoising AE'
    prior=json.loads((OUT/'prior_seven_metrics_snapshot.json').read_text(encoding='utf-8'))
    for name,info in report['models'].items():
        if name=='ae_synthetic_denoising':continue
        for field in ['original_weak_agreement','synthetic_heldout_metrics','threshold']:
            assert info[field]==prior['models'][name][field],(name,field)
    report['protocol']['original_seven_results_preserved_after_denoising_extension']=True
    (OUT/'anomaly_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (OUT/'prior_models_preservation.json').write_text(json.dumps({'pre_extension_seven_score_arrays_exactly_restored':True,'roundtrip_values_corrected':changes,
       'seven_original_and_synthetic_metrics_and_thresholds_unchanged':True,'new_model_only':'ae_synthetic_denoising'},indent=2),encoding='utf-8')
    print(changes)

if __name__=='__main__':main()
