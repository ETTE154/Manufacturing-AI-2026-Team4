from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(r'E:\AI work\AWS_프로젝트')
SOURCE = ROOT / 'may-dataset' / 'rtu_ground_truth_may.csv'
DEST = ROOT / 'output' / 'analysis' / 'may_audit'
DEST.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(SOURCE)
original_shape = list(df.shape)
df['timestamp'] = pd.to_datetime(df['id'], errors='coerce')
ordered = df.sort_values('timestamp').copy()
ordered['date'] = ordered['timestamp'].dt.strftime('%Y-%m-%d')
ordered['hour'] = ordered['timestamp'].dt.hour
ordered['weekday'] = ordered['timestamp'].dt.dayofweek
ordered['weekend'] = ordered['weekday'].ge(5)
powcol = ordered['hourly_pow']
daily = ordered.groupby('date')['hourly_pow'].agg(['count','sum','mean','std','min','max'])
hourly = ordered.groupby('hour')['hourly_pow'].agg(['count','mean','std','min','max'])
weekday = ordered.groupby('weekday')['hourly_pow'].agg(['count','mean','std','min','max'])
weekend = ordered.groupby('weekend')['hourly_pow'].agg(['count','mean','std','min','max'])
expected = pd.date_range(ordered['timestamp'].min(), ordered['timestamp'].max(), freq='h')
summary = {
  'source': str(SOURCE),
  'shape': original_shape,
  'original_columns': [c for c in df.columns if c != 'timestamp'],
  'time_min': str(ordered['timestamp'].min()),
  'time_max': str(ordered['timestamp'].max()),
  'timestamp_parse_failures': int(df['timestamp'].isna().sum()),
  'is_source_chronological': bool(df['timestamp'].is_monotonic_increasing),
  'duplicate_rows': int(df.duplicated().sum()),
  'duplicate_timestamps': int(df['timestamp'].duplicated().sum()),
  'missing_values': df.isna().sum().to_dict(),
  'missing_hours_between_minmax': [str(x) for x in expected.difference(ordered['timestamp'])],
  'timestamp_delta_hours_counts': (ordered['timestamp'].diff().dt.total_seconds().div(3600).dropna().value_counts().to_dict()),
  'numeric_describe': df.select_dtypes('number').describe().to_dict(),
  'unique_counts': df.nunique().to_dict(),
  'coefficient_of_variation_hourly_pow': float(powcol.std()/powcol.mean()),
  'total_hourly_pow': float(powcol.sum()),
  'agg_pow_first': float(ordered['agg_pow'].iloc[0]),
  'agg_pow_minus_sum': float(ordered['agg_pow'].iloc[0]-powcol.sum()),
  'correlation_matrix': df.select_dtypes('number').corr().to_dict(),
  'derived_relationships': {},
  'daily': json.loads(daily.reset_index().to_json(orient='records')),
  'hour_of_day': json.loads(hourly.reset_index().to_json(orient='records')),
  'weekday_monday_zero': json.loads(weekday.reset_index().to_json(orient='records')),
  'weekend': json.loads(weekend.reset_index().to_json(orient='records')),
  'lag_correlations': {str(lag): float(powcol.autocorr(lag)) for lag in [1,2,3,6,12,24,48,168]},
}
for col, hypothesized_coef in [('may_bill',180.0),('may_carbon',0.424)]:
  ratio = ordered[col] / powcol
  residual = ordered[col] - hypothesized_coef*powcol
  summary['derived_relationships'][col] = {
    'ratio_min': float(ratio.min()),'ratio_max': float(ratio.max()),'ratio_mean': float(ratio.mean()),
    'ratio_std': float(ratio.std()),'hypothesized_coefficient': hypothesized_coef,
    'max_abs_residual': float(residual.abs().max()),
    'sum': float(ordered[col].sum()),
  }
delta = powcol.diff()
summary['hour_to_hour_changes'] = {'std': float(delta.std()), 'min': float(delta.min()), 'max': float(delta.max()), 'max_abs': float(delta.abs().max())}
summary['extremes'] = {
  'lowest_hours': ordered.nsmallest(10,'hourly_pow')[['id','hourly_pow']].to_dict('records'),
  'highest_hours': ordered.nlargest(10,'hourly_pow')[['id','hourly_pow']].to_dict('records'),
}
summary['hourly_amplitude'] = {'max_mean':float(hourly['mean'].max()), 'min_mean':float(hourly['mean'].min()),'max_mean_hour':int(hourly['mean'].idxmax()),'min_mean_hour':int(hourly['mean'].idxmin()),'peak_to_trough_fraction':float((hourly['mean'].max()-hourly['mean'].min())/powcol.mean())}
summary['daily_amplitude'] = {'max_mean':float(daily['mean'].max()), 'min_mean':float(daily['mean'].min()),'max_mean_date':str(daily['mean'].idxmax()),'min_mean_date':str(daily['mean'].idxmin()),'peak_to_trough_fraction':float((daily['mean'].max()-daily['mean'].min())/powcol.mean())}
full_daily = daily.loc[daily['count'].eq(24)]
summary['complete_day_amplitude'] = {'number_complete_days':int(len(full_daily)), 'max_mean':float(full_daily['mean'].max()), 'min_mean':float(full_daily['mean'].min()),'max_mean_date':str(full_daily['mean'].idxmax()),'min_mean_date':str(full_daily['mean'].idxmin()),'peak_to_trough_fraction':float((full_daily['mean'].max()-full_daily['mean'].min())/powcol.mean())}
summary['partial_days'] = json.loads(daily.loc[daily['count'].ne(24)].reset_index().to_json(orient='records'))
ordered['time_index'] = np.arange(len(ordered))
slope, intercept = np.polyfit(ordered['time_index'], powcol, 1)
summary['linear_time_trend'] = {'slope_per_hour':float(slope),'intercept':float(intercept),'predicted_total_period_change':float(slope*(len(ordered)-1)), 'correlation':float(ordered['time_index'].corr(powcol))}
summary['first_vs_last_half'] = {'first_mean':float(powcol.iloc[:len(ordered)//2].mean()),'last_mean':float(powcol.iloc[len(ordered)//2:].mean())}
summary['repeated_hourly_pow_rows'] = ordered.loc[ordered['hourly_pow'].duplicated(keep=False), ['id','hourly_pow']].to_dict('records')
split_index = len(ordered)-168
train = ordered.iloc[:split_index]
test = ordered.iloc[split_index:]
hour_mapping = train.groupby('hour')['hourly_pow'].mean()
predictions = {
  'training_mean': np.repeat(train['hourly_pow'].mean(), len(test)),
  'previous_hour_rolling': ordered['hourly_pow'].shift(1).iloc[split_index:].to_numpy(),
  'previous_day_rolling': ordered['hourly_pow'].shift(24).iloc[split_index:].to_numpy(),
  'training_hour_means': test['hour'].map(hour_mapping).to_numpy(),
}
summary['diagnostic_baseline_last_168_hours'] = {'train_start':str(train['timestamp'].min()),'train_end':str(train['timestamp'].max()),'test_start':str(test['timestamp'].min()),'test_end':str(test['timestamp'].max()),'train_count':int(len(train)), 'test_count':int(len(test)), 'scores': {}}
actual = test['hourly_pow'].to_numpy()
for name,pred in predictions.items():
  summary['diagnostic_baseline_last_168_hours']['scores'][name] = {'mae':float(np.abs(actual-pred).mean()),'rmse':float(np.sqrt(((actual-pred)**2).mean())), 'mape_percent':float((np.abs(actual-pred)/np.abs(actual)).mean()*100)}
# Isolate time-pattern explanatory power without using target-derived columns.
grand_mean = powcol.mean()
sst = ((powcol-grand_mean)**2).sum()
summary['group_explained_variance'] = {}
for grouping in ['hour','weekday','date']:
  group_mean = ordered.groupby(grouping)['hourly_pow'].transform('mean')
  summary['group_explained_variance'][grouping] = float(1-((powcol-group_mean)**2).sum()/sst)
def clean_json(value):
  if isinstance(value, dict):
    return {str(k):clean_json(v) for k,v in value.items()}
  if isinstance(value, list):
    return [clean_json(v) for v in value]
  if isinstance(value, float) and not np.isfinite(value):
    return None
  return value
summary = clean_json(summary)
with (DEST/'audit_summary.json').open('w',encoding='utf-8') as handle:
  json.dump(summary,handle,ensure_ascii=False,indent=2,allow_nan=False)
for name,table in [('daily',daily),('hour_of_day',hourly),('weekday',weekday),('weekend',weekend)]:
  table.to_csv(DEST/f'{name}.csv',encoding='utf-8-sig')
print(json.dumps({k:summary[k] for k in ['shape','time_min','time_max','duplicate_rows','duplicate_timestamps','missing_values','missing_hours_between_minmax','unique_counts','coefficient_of_variation_hourly_pow','total_hourly_pow','agg_pow_first','agg_pow_minus_sum','derived_relationships','hourly_amplitude','daily_amplitude','linear_time_trend','lag_correlations','group_explained_variance']},ensure_ascii=False,indent=2))
