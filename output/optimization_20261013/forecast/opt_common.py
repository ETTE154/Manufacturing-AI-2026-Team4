"""New optimization experiment; read old artifacts but never modify them."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TARGET_SOURCE = ROOT/'output/analysis/baseline_audit/complete_hour_module_sum.csv'
TIME_AUDIT = ROOT/'output/analysis/time_anomaly_audit.json'
MAY_SOURCE = ROOT/'may-dataset/rtu_ground_truth_may.csv'
SOURCE_APPROVAL = OUT/'source_approval.json'
DAY_POINTS = 24
LONG_POINTS = 745
LONG_VALID_ORIGIN = pd.Timestamp('2025-02-25 00:00:00')
LONG_TEST_ORIGIN = pd.Timestamp('2025-03-29 00:00:00')
DAY_FOLD_STARTS = pd.to_datetime(['2025-01-10','2025-01-24','2025-02-07','2025-02-21','2025-03-07'])
APRIL_TEST_ORIGINS = pd.date_range('2025-04-01','2025-04-29',freq='D')
SEED = 20261013
THREADS = 4

def clean_json(obj):
    if isinstance(obj,dict): return {str(k):clean_json(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [clean_json(v) for v in obj]
    if isinstance(obj,np.ndarray): return clean_json(obj.tolist())
    if isinstance(obj,(pd.Timestamp,Path)): return str(obj)
    if isinstance(obj,np.integer): return int(obj)
    if isinstance(obj,(float,np.floating)): return float(obj) if np.isfinite(obj) else None
    return obj

def write_json(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean_json(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def load_target():
    audit = json.loads(TIME_AUDIT.read_text(encoding='utf-8'))
    for checks in audit['time_key_checks'].values():
        for key in ['local_duplicate_rows','local_negative_diffs','local_non_5s_diffs']:
            if checks[key]!=0: raise ValueError('Localtime grid failed exhaustive source audit')
    raw = pd.read_csv(TARGET_SOURCE)
    y = pd.Series(raw['provisional_sum_energy_kwh'].to_numpy(dtype=float),index=pd.to_datetime(raw.iloc[:,0]),name='energy_kwh').sort_index()
    if len(y)!=3600 or not y.index.is_unique or not np.isfinite(y.to_numpy()).all():
        raise ValueError('Unexpected aggregate target shape')
    if not y.index.equals(pd.date_range('2024-12-01 00:00:00','2025-04-29 23:00:00',freq='h')):
        raise ValueError('Hourly target is not the declared continuous interval')
    return y

def require_source_approval():
    if not SOURCE_APPROVAL.exists():
        raise RuntimeError('Preparation complete; training awaits root source-audit approval')
    approved = json.loads(SOURCE_APPROVAL.read_text(encoding='utf-8'))
    if not approved.get('approved_for_training'): raise RuntimeError('Source approval is not affirmative')
    if approved.get('target_sha256')!=hashlib.sha256(TARGET_SOURCE.read_bytes()).hexdigest():
        raise ValueError('Approved target hash differs from current aggregate')
    return approved

def future(origin,points):
    return pd.date_range(pd.Timestamp(origin),periods=int(points),freq='h')

def history(y,origin):
    h = y[y.index<pd.Timestamp(origin)]
    if h.empty or h.index[-1]+pd.Timedelta(hours=1)!=pd.Timestamp(origin):
        raise ValueError('Origin does not immediately follow known history')
    return h

def actual(y,idx):
    a = y.reindex(idx)
    if a.isna().any(): raise ValueError('Target horizon extends past available labels')
    return a.to_numpy(dtype=float)

def metrics(a,p):
    a,p = np.asarray(a,dtype=float),np.asarray(p,dtype=float)
    if a.shape!=p.shape or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError('Unaligned/nonfinite arrays')
    error = p-a
    variance = np.sum((a-a.mean())**2)
    return {'count':len(a),'mae_kwh':float(np.mean(np.abs(error))),
            'rmse_kwh':float(np.sqrt(np.mean(error**2))),
            'mape_percent':float(np.mean(np.abs(error)/np.abs(a))*100),
            'bias_pred_minus_actual_kwh':float(error.mean()),
            'r2':float(1-np.sum(error**2)/variance) if variance>0 else None}

def prediction_frame(model,scenario,split,fold,origin,idx,a,p):
    offset = ((pd.DatetimeIndex(idx)-pd.Timestamp(origin)).total_seconds()/3600).astype(int)
    actual_values = np.asarray(a,dtype=np.float64)
    predicted_values = np.asarray(p,dtype=np.float64)
    return pd.DataFrame({'model':model,'scenario':scenario,'split':split,'fold':fold,
                         'origin':origin,'id':idx,'horizon_offset_hours':offset,'lead_hours':offset+1,
                         'actual_kwh':actual_values,'predicted_kwh':predicted_values,
                         'absolute_error_kwh':np.abs(actual_values-predicted_values).astype(np.float64)})

def day_summary(frame):
    ordered = frame.sort_values(['origin','id'])
    if ordered.duplicated(['origin','id']).any(): raise ValueError('Duplicate origin/target')
    counts = ordered.groupby('origin').size()
    if not counts.eq(24).all(): raise ValueError('Not all day-ahead paths have exactly 24 values')
    endpoint = ordered[ordered['lead_hours'].eq(24)]
    by_origin = ordered.groupby('origin')['absolute_error_kwh'].mean()
    return {'path_24_values':metrics(ordered.actual_kwh,ordered.predicted_kwh),
            'endpoint_lead_24':metrics(endpoint.actual_kwh,endpoint.predicted_kwh),
            'origin_count':len(counts),'mean_origin_path_mae_kwh':float(by_origin.mean()),
            'origin_path_mae_std_kwh':float(by_origin.std(ddof=1)) if len(by_origin)>1 else None,
            'origin_start':str(pd.to_datetime(ordered.origin).min()),'origin_end':str(pd.to_datetime(ordered.origin).max())}

def protocol_manifest(y):
    folds = []
    all_targets = []
    for i,start in enumerate(DAY_FOLD_STARTS,1):
        origins = pd.date_range(start,periods=14,freq='D')
        targets = [future(origin,DAY_POINTS) for origin in origins]
        flat = pd.DatetimeIndex(np.concatenate([idx.values for idx in targets]))
        if not flat.is_unique: raise ValueError('Day fold contains overlapping target periods')
        all_targets.extend(flat.tolist())
        for idx in targets: actual(y,idx)
        folds.append({'fold':i,'fit_origin':str(start),'history_hours':len(history(y,start)),
                      'origins':[str(t) for t in origins],'target_start':str(flat[0]),'target_end':str(flat[-1]),
                      'target_rows':len(flat),'origins_count':len(origins)})
    if len(set(all_targets))!=len(all_targets): raise ValueError('Day folds overlap target periods')
    for origin in APRIL_TEST_ORIGINS: actual(y,future(origin,24))
    valid = future(LONG_VALID_ORIGIN,LONG_POINTS)
    test = future(LONG_TEST_ORIGIN,LONG_POINTS)
    actual(y,valid);actual(y,test)
    return {'status':'source_approved_training_enabled' if SOURCE_APPROVAL.exists() else 'prepared_source_approval_pending','assumed_presentation_scenario':'24-value hourly day-ahead curve, plus lead24 endpoint; user clarification was pending and 24h is the declared assumption',
            'target_source':str(TARGET_SOURCE),'target_sha256':hashlib.sha256(TARGET_SOURCE.read_bytes()).hexdigest(),
            'old_outputs_policy':'Read existing analysis/modeling; all new code/models/metrics/predictions/figures only under this forecast directory',
            'target':'13-module provisional hourly energy sum, kWh; unknown meter hierarchy can double count; not certified facility consumption',
            'source_resolution_seconds':5,'model_step_hours':1,
            'aggregation':'sum(activePower_kW)*5/3600 over 720 unique five-second values per module; all 13 modules must be complete at the same hour',
            'data_start':str(y.index[0]),'data_end':str(y.index[-1]),'target_hours':len(y),
            'time_key':'Recorded localtime: continuous audited grid; supplied Unix key has March9 offset transition inconsistent with Korean-time guide',
            'dayahead':{'points':24,'offsets_from_origin':'0..23','leads_from_last_observed':'1..24',
                        'endpoint_definition':'lead24 is origin+23h, not origin+24h; relative to last observed origin-1h it is exactly 24h later',
                        'validation_folds':folds,'validation_total_origins':70,'validation_target_rows':1680,
                        'training_updates':'Expand fitted history at each fold start. Weights fixed within each 14-day block; input context updates daily using only observations before that origin. ARIMA filtering/state may update on newly past observations without refitting coefficients.',
                        'selection':'Lowest mean 70-origin path MAE; 24h endpoint MAE is separately reported, not used opportunistically for winner selection',
                        'test_fit_origin':'2025-04-01 00:00:00','test_origins':[str(t) for t in APRIL_TEST_ORIGINS],
                        'test_origin_count':29,'test_target_rows':696,
                        'test_target_start':'2025-04-01 00:00:00','test_target_end':'2025-04-29 23:00:00',
                        'test_usage':'April labels excluded from new HPO. Existing project EDA/evaluations already saw April, so this is post-EDA locking, not a pristine prospective holdout.'},
            'long':{'points':745,'offsets_from_origin':'0..744','leads_from_last_observed':'1..745',
                    'validation_origin':str(LONG_VALID_ORIGIN),'validation_history_hours':len(history(y,LONG_VALID_ORIGIN)),
                    'validation_end':str(valid[-1]),'test_origin':str(LONG_TEST_ORIGIN),
                    'test_history_hours':len(history(y,LONG_TEST_ORIGIN)),'test_end':str(test[-1]),
                    'selection':'Blocked Feb25 fixed-origin P745 validation only; March29 internal test does not tune',
                    'independent_origin_limitation':'Only one long validation origin and one long test origin; 150 days of data cannot support many nonoverlapping 31-day horizons with sufficient expanding history.',
                    'may_origin':'2025-04-30 00:00:00','may_path_end':'2025-05-31 00:00:00',
                    'may_score_start':'2025-05-10 00:00:00','may_score_end':'2025-05-31 00:00:00',
                    'may_score_rows':505,'may_score_offsets':'240..744','may_score_leads':'241..745',
                    'may_no_tuning':'May values were seen in prior EDA/evaluation. No additional HPO, calibration, preprocessing, or model selection may use May labels.',
                    'may_day_ahead_limitation':'May sensor/context observations are unavailable. Public May505 labels are not used as rolling model inputs. April demonstrates day-ahead; May remains the separate single-origin long-forecast task.'},
            'neural_input_policy':'All newly optimized neural models are aggregate univariate; iTransformer consequently has one variate token and cannot demonstrate its multivariate cross-channel attention advantage. XGBoost/calendar Ridge additionally use deterministically known calendar.',
            'optimization_claim':'Budgeted local search and validation selection; no guarantee of global optimum or superiority over strong simple baselines',
            'planned_search_budget':{'xgboost_configs':31,'xgboost_loss_extension':'30original configurations plus1 declared loss-only MAE companion; original intermediate results preserved','arima_small_grid':True,'patchtst_configs_per_scenario':7,'itransformer_configs_per_scenario':7,'neural_loss_extension':'6MSE configurations plus1 declared L1 candidate','day_neural_contexts':[24,72,168,336,720],'long_neural_contexts':[24,72,168,336,512],'long_context_constraint':'Inner fitting history has1319points after holding out745; context720+output745 exceeds it, so context512 is the largest predeclared feasible candidate.','chronos_contexts':[168,720,'full'],'cpu_threads':4},
            'seed':SEED}

if __name__=='__main__':
    y=load_target();manifest=protocol_manifest(y);write_json(OUT/'protocol.json',manifest)
    print(json.dumps({'status':manifest['status'],'day_folds':len(DAY_FOLD_STARTS),'day_validation_origins':70,
                      'April_test_origins':29,'April_test_target_rows':696,'long_path_rows':745,
                      'source_sha256':manifest['target_sha256']},indent=2))
