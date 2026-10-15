"""Summarize verified EDA products, with explicit units and scope."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output'/'analysis'
e=pd.read_csv(OUT/'equipment_summary.csv')
h=pd.read_csv(OUT/'hourly_by_equipment.csv',parse_dates=['hour'])
p=json.loads((OUT/'training_profile.json').read_text(encoding='utf-8'))
a=json.loads((OUT/'time_anomaly_audit.json').read_text(encoding='utf-8'))
may=pd.read_csv(ROOT/'may-dataset'/'rtu_ground_truth_may.csv',parse_dates=['id'])
base=json.loads((OUT/'baseline_audit'/'baseline_summary.json').read_text(encoding='utf-8'))
forecast=pd.read_csv(OUT/'baseline_audit'/'may_selected_baseline_predictions.csv',parse_dates=['id'])
forecast=forecast.merge(may[['id','may_bill','may_carbon']],on='id',how='left',validate='one_to_one')
forecast['forecast_cost_krw']=forecast.forecast_provisional_sum_kwh*180
forecast['forecast_carbon_kgco2']=forecast.forecast_provisional_sum_kwh*0.424
forecast['absolute_cost_error_krw']=abs(forecast.forecast_cost_krw-forecast.may_bill)
forecast['absolute_carbon_error_kgco2']=abs(forecast.forecast_carbon_kgco2-forecast.may_carbon)
forecast.to_csv(OUT/'baseline_audit'/'may_forecast_energy_cost_carbon.csv',index=False,encoding='utf-8-sig')
base['derived_evaluation']={'cost_mae_krw':float(forecast.absolute_cost_error_krw.mean()),
    'carbon_mae_kgco2':float(forecast.absolute_carbon_error_kgco2.mean()),
    'method':'same fixed energy forecast multiplied by guide factors 180 and 0.424; no separate models'}
hc=h[h.records.eq(720)].copy()
g=hc.groupby('hour').agg(equipment_count=('module(equipment)','nunique'),energy_kwh=('energy_kwh','sum'))
g=g[g.equipment_count.eq(len(e))]
g['mean_power_kw']=g.energy_kwh
g['cost_krw']=g.energy_kwh*180
g['carbon_kgco2']=g.energy_kwh*0.424
g.to_csv(OUT/'hourly_module_sum.csv',encoding='utf-8-sig')
daily=g.groupby(g.index.date).agg(hours=('energy_kwh','size'),energy_kwh=('energy_kwh','sum'),mean_hourly_energy_kwh=('energy_kwh','mean'),cost_krw=('cost_krw','sum'),carbon_kgco2=('carbon_kgco2','sum'))
daily.index.name='date';daily.to_csv(OUT/'daily_summary.csv',encoding='utf-8-sig')
monthly=g.groupby(g.index.to_period('M')).agg(hours=('energy_kwh','size'),energy_kwh=('energy_kwh','sum'),mean_hourly_energy_kwh=('energy_kwh','mean'),cost_krw=('cost_krw','sum'),carbon_kgco2=('carbon_kgco2','sum'))
monthly.index=monthly.index.astype(str);monthly.index.name='month'
monthly['covered_complete_days']=monthly.hours/24
monthly.to_csv(OUT/'monthly_summary.csv',encoding='utf-8-sig')
hour=g.groupby(g.index.hour).energy_kwh.agg(['count','mean','std'])
hour.index.name='hour_of_day';hour.to_csv(OUT/'hour_of_day_summary.csv',encoding='utf-8-sig')
wd=g.groupby(g.index.dayofweek).energy_kwh.agg(['count','mean','std'])
wd.index.name='weekday_monday_zero';wd.to_csv(OUT/'weekday_summary.csv',encoding='utf-8-sig')
s=pd.read_csv(OUT/'sampled_sensor_rows.csv')
ca=pd.read_csv(OUT/'anomaly_candidate_rows.csv',parse_dates=['dt'])
anomaly_by_equipment=ca.groupby('module(equipment)').agg(candidate_rows=('activePower','size'),voltage_candidate_rows=('voltage_candidate','sum'),pf_candidate_rows=('pf_candidate','sum'),first=('dt','min'),last=('dt','max')).reset_index()
anomaly_by_equipment['rate_of_equipment_percent']=anomaly_by_equipment.candidate_rows.map(lambda n:100*n/2592001)
anomaly_by_equipment['first']=anomaly_by_equipment['first'].astype(str)
anomaly_by_equipment['last']=anomaly_by_equipment['last'].astype(str)
anomaly_by_equipment.to_csv(OUT/'anomaly_by_equipment.csv',index=False,encoding='utf-8-sig')
sc=s[['activePower','currentR','currentS','currentT','voltageR','voltageS','voltageT','powerFactorR','powerFactorS','powerFactorT','reactivePowerLagging']].corr()
sc.to_csv(OUT/'sample_correlations.csv',encoding='utf-8-sig')
e['share_observed_module_sum']=e.energy_kwh/e.energy_kwh.sum()
e.to_csv(OUT/'equipment_summary.csv',index=False,encoding='utf-8-sig')
metrics={
 'analysis_date':'2026-10-11',
 'guide_goal':'전기요금·탄소배출량 예측, 삽입 이상치 탐지, 분석 에이전트 및 Streamlit 대시보드',
 'training':{'rows':p['records'],'equipment_count':len(e),'columns':len(p['columns']),
   'localtime_first':e['first'].min(),'localtime_last':e['last'].max(),'complete_module_hours':len(hc),
   'complete_all_module_hours':len(g),'complete_days':len(daily[daily.hours.eq(24)]),
   'partial_end_hour':'2025-04-30 00:00:00','partial_end_records_per_equipment':1,
   'observed_energy_kwh_including_partial_end':float(e.energy_kwh.sum()),
   'complete_hour_energy_kwh':float(g.energy_kwh.sum()),
   'hourly_energy_mean_kwh':float(g.energy_kwh.mean()),'hourly_energy_std_kwh':float(g.energy_kwh.std()),
   'hourly_cv_percent':float(100*g.energy_kwh.std()/g.energy_kwh.mean()),
   'operation_rate':1.0,'equipment_mean_power_spread_percent':float(100*(e.activePower_mean.max()-e.activePower_mean.min())/e.activePower_mean.mean()),
   'equipment_observed_energy_share_min_percent':float(e.share_observed_module_sum.min()*100),
   'equipment_observed_energy_share_max_percent':float(e.share_observed_module_sum.max()*100),
   'hour_of_day_peak_to_trough_percent':float((hour['mean'].max()-hour['mean'].min())/g.energy_kwh.mean()*100),
   'weekday_peak_to_trough_percent':float((wd['mean'].max()-wd['mean'].min())/g.energy_kwh.mean()*100)},
 'may':{'rows':len(may),'time_first':str(may.id.min()),'time_last':str(may.id.max()),
   'mean_hourly_target':float(may.hourly_pow.mean()),'std_hourly_target':float(may.hourly_pow.std()),
   'cv_percent':float(100*may.hourly_pow.std()/may.hourly_pow.mean()),
   'provided_period_energy_assuming_kwh':float(may.hourly_pow.sum()),
   'provided_period_cost_krw':float(may.may_bill.sum()),'provided_period_carbon_kgco2':float(may.may_carbon.sum()),
   'mean_change_vs_training_percent':float((may.hourly_pow.mean()/g.energy_kwh.mean()-1)*100),
   'complete_days':21,'partial_last_day_hours':1,'is_full_calendar_month':False,
   'derived_columns':'may_bill=hourly_pow*180; may_carbon=hourly_pow*0.424; agg_pow=sum(hourly_pow) repeated on all rows'},
 'quality':{'null_cells':sum(p['missing'].values()),
   'localtime_duplicate_rows':sum(x['local_duplicate_rows'] for x in a['time_key_checks'].values()),
   'localtime_non_5s_diffs':sum(x['local_non_5s_diffs'] for x in a['time_key_checks'].values()),
   'unix_duplicate_extra_rows':sum(x['unix_duplicate_rows'] for x in a['time_key_checks'].values()),
   'unix_reverse_transitions':int(e.negative_time_diffs.sum()),
   'offsets_local_naive_minus_unix_ms':a['offset_local_naive_minus_unix_ms'],
   'offset_transition_first':a['offset_transitions'][0] if a['offset_transitions'] else None,
   'counter_unit_ratio_min':float(e.counter_unit_ratio.min()),'counter_unit_ratio_max':float(e.counter_unit_ratio.max()),
   'counter_negative_diffs':int(e.counter_negative_diffs.sum())},
 'anomalies':{'candidate_rows':a['candidate_rows_unique'],'candidate_rate_percent':100*a['candidate_rows_unique']/p['records'],
   'consecutive_candidate_run_count':a['event_count'],'by_equipment':anomaly_by_equipment.to_dict('records'),'rule_counts':a['rule_counts'],
   'rule_definition':'상전압 <205V 또는 >225V; 상역률 <80% 또는 >100%; 음수 전력·전류 또는 operation 값 오류',
   'limitations':'전형적인 학습 분포를 참고한 탐색적 후보. 현장 정상범위·정답 라벨을 검증하지 않았으며 확정 고장/삽입 이상치 판정이 아니다.'},
 'sensor_sample':{'rows':len(s),'method':p['sample_method'],
   'corr_power_total_current':float(s.activePower.corr(s[['currentR','currentS','currentT']].sum(axis=1))),
   'limitations':'상관관계는 16,849행 체계적 표본 기준. 전수 결과와 구분. 미래 실측 전류는 예측 시점에 미리 알려지지 않아 예측 특성으로 사용 불가.'},
 'baseline':base,
 'assumptions':{'power_unit':'activePower=kW (가이드)',
   'energy_formula':'sum(activePower)*5/3600 kWh','cost_krw_per_kwh':180,'carbon_kgco2_per_kwh':0.424,
   'may_unit':'5월 CSV 자체 단위 미기재. 가이드 계산식과 학습 값 규모에 따라 kWh 동일 스케일 비교를 가정',
   'meter_hierarchy':'계측 계층 미확인. 전 설비 합계는 잠정 모듈 합계이며 실제 공장 총량으로 확정할 수 없음',
   'time_axis':'전수 확인된 localtime 사용. Unix timestamp는 가이드 한국시간과 불일치하여 재정의 필요'}
}
(OUT/'analysis_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'rows':metrics['training']['rows'],'equipment':len(e),
    'candidate_rows':metrics['anomalies']['candidate_rows'],'localtime_key_issues':metrics['quality']['localtime_duplicate_rows']+metrics['quality']['localtime_non_5s_diffs'],
    'energy_mae_kwh':base['may_selected_baseline']['scores']['mae'],'derived_evaluation':base['derived_evaluation']},ensure_ascii=False),flush=True)
