"""Read-only forecast audit; paired bootstrap uncertainty, never model reselection.

Only output is paired_uncertainty.json in this research directory. April's unit is
the 24-hour mean absolute error at each of 29 daily origins. May's unit is a
calendar-day block on a single fixed-origin path, including the final one-hour
partial day. Unequal May blocks are pooled by row count rather than giving the
single-hour partial day the same point weight as a complete day.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
FORECAST=OUT.parent/'forecast'
COMPARISON=FORECAST/'comparison_results.json'
SEED=20261013
RESAMPLES=10000

def readjson(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_csv(row,filename,expected_scenario,expected_split):
    # Result directories are explicitly referenced by comparison_results.json.
    result_path=Path(row['result_path']);path=result_path.parent/filename
    if not path.is_file():raise FileNotFoundError(path)
    d=pd.read_csv(path,float_precision='round_trip',parse_dates=['origin','id'])
    required=['origin','id','horizon_offset_hours','lead_hours','actual_kwh','predicted_kwh','absolute_error_kwh']
    if any(k not in d for k in required):raise ValueError(f'Missing prediction columns: {path}')
    assert d.scenario.eq(expected_scenario).all(),path
    allowed={'April_test','test'} if expected_split=='April_test' else {expected_split}
    assert d.split.isin(allowed).all(),path
    assert not d.duplicated(['origin','id']).any(),path
    assert np.isfinite(d[['actual_kwh','predicted_kwh','absolute_error_kwh']].to_numpy()).all(),path
    assert np.array_equal(d.lead_hours.to_numpy(),d.horizon_offset_hours.to_numpy()+1),path
    assert d.id.eq(d.origin+pd.to_timedelta(d.horizon_offset_hours,unit='h')).all(),path
    recomputed=np.abs(d.predicted_kwh.to_numpy()-d.actual_kwh.to_numpy())
    assert np.allclose(recomputed,d.absolute_error_kwh.to_numpy(),rtol=0,atol=1e-8),path
    d['error']=recomputed
    metric=row['test_metrics'] if filename=='test_predictions.csv' else row['may_metrics']
    assert len(d)==metric['count'] and abs(float(d.error.mean())-metric['mae_kwh'])<1e-8,(path,'JSON metric mismatch')
    return d.sort_values(['origin','id']).reset_index(drop=True),{
        'prediction_csv':str(path),'prediction_sha256':sha(path),
        'referenced_result_json':str(result_path),'json_metric_mae_kwh':metric['mae_kwh'],
        'source_csv_split_labels':sorted(d.split.unique().tolist())}

def pair(model,baseline):
    cols=['origin','id','horizon_offset_hours','lead_hours']
    d=model.merge(baseline[cols+['actual_kwh','error']],on=cols,how='inner',validate='one_to_one',suffixes=('_model','_baseline'))
    assert len(d)==len(model)==len(baseline),'Prediction cohorts differ'
    assert np.array_equal(d.actual_kwh_model.to_numpy(),d.actual_kwh_baseline.to_numpy()),'Actual values differ'
    return d

def bootstrap(units,counts):
    """Same sampled block indices for both paired errors, weighted by observations."""
    units=np.asarray(units,dtype=np.float64);counts=np.asarray(counts,dtype=np.int64)
    assert units.shape==counts.shape and (counts>0).all()
    rng=np.random.default_rng(SEED);idx=rng.integers(0,len(units),size=(RESAMPLES,len(units)))
    sampled_rows=counts[idx].sum(axis=1)
    draws=units[idx].sum(axis=1)/sampled_rows
    lo,hi=np.quantile(draws,[.025,.975],method='linear')
    return {'resamples':RESAMPLES,'seed':SEED,'method':'Paired cluster/block resampling with replacement, percentile 2.5/97.5 interval',
            'confidence_level':.95,'paired_bootstrap_95_interval_kwh':[float(lo),float(hi)],
            'interval_excludes_zero':bool(lo>0 or hi<0),
            'nominal_interval_direction':'baseline_minus_model',
            'resampled_observation_counts':{'minimum':int(sampled_rows.min()),'maximum':int(sampled_rows.max()),
                                            'variable_due_to_partial_block':bool(sampled_rows.min()!=sampled_rows.max())},
            'bootstrap_standard_error_kwh':float(draws.std(ddof=1)),
            'quantile_method':'numpy linear percentile; no point-adjustment, multiple-test correction, or model reselection'}

def summarize(d,model_key,model_row,baseline_row,model_source,baseline_source,unit_col,unit_type):
    unitrows=[];sums=[];counts=[]
    for unit,x in d.groupby(unit_col,sort=True):
        n=len(x);baseline_mae=float(x.error_baseline.mean());model_mae=float(x.error_model.mean())
        delta_sum=float((x.error_baseline-x.error_model).sum())
        sums.append(delta_sum);counts.append(n)
        unitrows.append({'unit':str(unit),'observations':n,'first_target':str(x.id.min()),'last_target':str(x.id.max()),
                         'baseline_mae_kwh':baseline_mae,'model_mae_kwh':model_mae,
                         'baseline_minus_model_mae_kwh':delta_sum/n})
    delta=float((d.error_baseline-d.error_model).mean());baseline_mae=float(d.error_baseline.mean());model_mae=float(d.error_model.mean())
    assert np.isclose(delta,np.sum(sums)/np.sum(counts),rtol=0,atol=1e-12)
    result={'model':model_key,'model_display_name':model_row['display_name'],
            'model_config_locked':model_row['selected_config'],'baseline_config_locked':baseline_row['selected_config'],
            'observations':len(d),'paired_unit_type':unit_type,'paired_units':len(unitrows),
            'target_start':str(d.id.min()),'target_end':str(d.id.max()),
            'baseline_mae_kwh':baseline_mae,'model_mae_kwh':model_mae,
            'baseline_minus_model_mae_kwh':delta,'relative_mae_improvement_percent':100*delta/baseline_mae,
            'positive_direction':'Positive baseline-minus-model means lower MAE / improvement by the model; negative means worse',
            'source_paths':{'model':model_source,'baseline':baseline_source},'units':unitrows,
            'uncertainty':bootstrap(sums,counts)}
    return result

def main():
    comparison=readjson(COMPARISON);selection_path=Path(comparison['selection_lock']);selection_before=sha(selection_path)
    lock=readjson(selection_path);scenarios=comparison['scenarios'];day=scenarios['dayahead24'];long=scenarios['long745']
    assert day['selected_model']==long['selected_model']=='chronos2'
    assert lock['scenarios']['dayahead24']['selected_model']==day['selected_model']
    assert lock['scenarios']['long745']['selected_model']==long['selected_model']
    assert lock['selection_uses_May_labels'] is False
    day_frames={};may_frames={};day_sources={};may_sources={};mae_audit={}
    # Audit all locked model rows without using their test scores to select a model.
    for key,row in day['models'].items():
        day_frames[key],day_sources[key]=load_csv(row,'test_predictions.csv','dayahead24','April_test')
    for key,row in long['models'].items():
        may_frames[key],may_sources[key]=load_csv(row,'may_predictions.csv','long745','May_P505')
    for scope,frames in [('april_24h',day_frames),('may_505h',may_frames)]:
        base=frames['baseline'];mae_audit[scope]={}
        for key,d in frames.items():
            aligned=pair(d,base);mae_audit[scope][key]={'rows':len(aligned),'mae_kwh':float(d.error.mean()),'actuals_keys_identical_to_baseline':True}
    april=pair(day_frames['chronos2'],day_frames['baseline'])
    expected=pd.date_range('2025-04-01','2025-04-29',freq='D')
    assert len(april)==696 and april.origin.nunique()==29
    assert april.groupby('origin').size().eq(24).all()
    assert np.array_equal(np.sort(april.origin.unique()),expected.to_numpy())
    assert april.id.nunique()==696 and april.id.sort_values().diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert april.groupby('origin').horizon_offset_hours.apply(lambda s:np.array_equal(np.sort(s),np.arange(24))).all()
    a=summarize(april,'chronos2',day['models']['chronos2'],day['models']['baseline'],day_sources['chronos2'],day_sources['baseline'],
                'origin','One complete 24-hour path MAE per fixed daily April origin; 29 units, equal 24-hour weights')
    a['evaluation_role']='CV-selected day-ahead blend, locked before April scoring; uncertainty does not reselect a model'
    observed=long['may_observed_best_not_selected'];assert observed=='xgboost'
    # Check the documented observation; keep CV winner unchanged regardless of result.
    assert min(mae_audit['may_505h'],key=lambda k:mae_audit['may_505h'][k]['mae_kwh'])==observed
    may_comparisons={}
    for key in ['chronos2',observed]:
        d=pair(may_frames[key],may_frames['baseline'])
        assert len(d)==505 and d.origin.nunique()==1
        assert d.origin.iloc[0]==pd.Timestamp('2025-04-30')
        assert np.array_equal(d.horizon_offset_hours.to_numpy(),np.arange(240,745))
        assert d.id.min()==pd.Timestamp('2025-05-10') and d.id.max()==pd.Timestamp('2025-05-31')
        assert d.id.diff().dropna().eq(pd.Timedelta(hours=1)).all()
        d['day_block']=d.id.dt.floor('D');blockcounts=d.groupby('day_block').size()
        assert len(blockcounts)==22 and blockcounts.iloc[:-1].eq(24).all() and blockcounts.iloc[-1]==1
        r=summarize(d,key,long['models'][key],long['models']['baseline'],may_sources[key],may_sources['baseline'],
                    'day_block','Nonoverlapping calendar-day blocks on a single long forecast path: 21 full 24-hour blocks + 1 final one-hour partial block')
        r['partial_block_policy']='Retain and resample the final May31 00:00 one-hour block in the same 22-block pool. Compute total paired error difference / total sampled hours; never average 22 block means equally.'
        r['evaluation_role']='Validation-selected long model, selection lock unchanged' if key=='chronos2' else 'Descriptive May observed best only, not deployment-selected; nominal CI conditional on fixed XGB and not corrected for choosing a May winner'
        may_comparisons[key]=r
    assert sha(selection_path)==selection_before,'Selection lock was modified'
    result={'status':'complete','seed':SEED,'resamples':RESAMPLES,'direction':'baseline_minus_model; positive = MAE improvement',
            'scope':'Observed paired-error uncertainty only. No new fitting, HPO, calibration, model ranking decision, or deployment selection.',
            'comparisons':{'april_24h_cv_selected_vs_baseline':a,
                           'may_505h_locked_selected_vs_baseline':may_comparisons['chronos2'],
                           'may_505h_observed_best_xgb_vs_baseline':may_comparisons[observed]},
            'all_models_csv_mae_audit':mae_audit,
            'provenance':{'comparison_json':str(COMPARISON),'comparison_sha256':sha(COMPARISON),
                          'selection_lock':str(selection_path),'selection_lock_sha256_before_and_after':selection_before,
                          'selection_unchanged_verified':True,'runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__}},
            'limitations_ko':['이 구간·저장 예측·모델 fit을 고정한 조건부 percentile bootstrap이며 학습·하이퍼파라미터 탐색·모델 선택의 불확실성을 포함하지 않습니다.',
                              'April의 29개 원점은 같은 공장의 한 달에서 나온 값입니다. 24시간 안의 상관은 보존하지만 인접 날짜 간 상관·겹치는 과거 입력·고정 모델 파라미터의 의존성은 완전히 반영하지 못합니다.',
                              'May의 22개 블록은 서로 독립적인 forecast origin이 아니라 한 745시간 예측 경로의 505개 라벨 구간입니다. 월·설비·다른 공장으로 일반화하는 신뢰구간이 아닙니다.',
                              '마지막 1시간 블록도 유지하며, resampling마다 전체 관측 수가 달라질 수 있습니다. 시간 수로 가중한 MAE 차이를 계산해 부분 블록을 24시간 블록과 같은 무게로 과대평가하지 않습니다.',
                              'May 관찰 최저 XGB 비교는 사후 관찰입니다. CI는 여러 모델 중 최저를 관찰한 선택 편향이나 다중 비교를 보정하지 않으며, 배포 모델을 바꾸는 근거로 사용하지 않습니다.',
                              'April·May는 기존 EDA/평가에서 이미 노출됐습니다. 완전히 미노출된 전향 평가로 해석할 수 없습니다.',
                              'May에는 새 센서 입력이 없어 일일 24h rolling 평가를 할 수 없습니다. April의 next-day 결과와 May의 단일 장기 경로는 다른 과제입니다.',
                              '명목 95% 구간이 0을 제외하더라도 이 데이터 조건의 관측 차이를 나타낼 뿐 공학적 가치·다른 월의 우위·실제 절감 효과를 보장하지 않습니다.']}
    target=OUT/'paired_uncertainty.json';target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:{'baseline_minus_model_mae_kwh':v['baseline_minus_model_mae_kwh'],
                         'paired_bootstrap_95_interval_kwh':v['uncertainty']['paired_bootstrap_95_interval_kwh'],
                         'units':v['paired_units'],'rows':v['observations']} for k,v in result['comparisons'].items()},ensure_ascii=True))

if __name__=='__main__':main()
