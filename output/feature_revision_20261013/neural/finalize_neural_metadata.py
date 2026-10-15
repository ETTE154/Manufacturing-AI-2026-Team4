"""Final read-only CLI check and exact budget-limit disclosures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent

def main():
    records=[]
    for scenario in ['dayahead24','long745']:
        for family in ['itransformer','patchtst']:
            dest=OUT/scenario/family;r=json.loads((dest/'results.json').read_text(encoding='utf-8'))
            actual_cli=pd.read_csv(OUT/'cli_verification'/f'{scenario}_{family}.csv',parse_dates=['id'],float_precision='round_trip')
            expected=pd.read_csv(r['prediction_csv']['future'],parse_dates=['id'],float_precision='round_trip')
            error=float(np.abs(actual_cli.predicted_kwh-expected.predicted_kwh).max())
            assert len(actual_cli)==len(expected) and actual_cli.id.equals(expected.id) and error<1e-8
            chosen=next(c for c in r['candidate_results'] if c['candidate']==r['selected_candidate'])
            cap_count=sum(int(f['inner']['best_epoch']==16) for f in chosen['fold_fits'])
            r['convergence_budget']={'max_epochs':16,'chosen_config_CV_folds_at_epoch_cap':cap_count,'CV_folds':len(chosen['fold_fits']),
                  'limit':'Small declared local budget, not a convergence/global-optimum guarantee. At-cap minima can still improve with more training; no test-driven budget expansion performed.'}
            r['reload_evidence']['actual_cli_output_rows']=len(actual_cli);r['reload_evidence']['actual_cli_max_absdiff_kwh']=error
            r['reload_evidence']['actual_cli_path']=str(OUT/'cli_verification'/f'{scenario}_{family}.csv')
            (dest/'results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            plan=json.loads((dest/'search_plan.json').read_text(encoding='utf-8'))
            plan['feature_selection']='Independent inner/outer TRAIN prefixes; pretest schema (March31 day, March28 long) reused for final April29 weight/scaler fit'
            (dest/'search_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            records.append({'family':family,'scenario':scenario,'rows':len(actual_cli),'max_absdiff_kwh':error,'chosen_CV_folds_at_cap':cap_count})
    (OUT/'cli_verification.json').write_text(json.dumps({'passed':True,'actual_CLI_runs':records,'May_labels_read':False},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(records,ensure_ascii=True))

if __name__=='__main__':main()
