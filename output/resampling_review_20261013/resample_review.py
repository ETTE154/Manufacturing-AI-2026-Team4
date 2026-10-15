"""Resample audited complete-hour energy; preserve partial-period flags."""
from pathlib import Path
import hashlib,json,uuid
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
SOURCE=ROOT/'output/analysis/baseline_audit/complete_hour_module_sum.csv'
def main():
    raw=pd.read_csv(SOURCE,float_precision='round_trip')
    y=pd.Series(raw.iloc[:,1].to_numpy(float),index=pd.to_datetime(raw.iloc[:,0])).sort_index()
    assert len(y)==3600 and y.index.equals(pd.date_range('2024-12-01','2025-04-29 23:00',freq='h'))
    datasets={};inventory=[]
    for name,freq in [('daily','D'),('weekly','W-SUN'),('monthly','MS')]:
        g=y.resample(freq).agg(['sum','mean','count'])
        starts=g.index if name!='weekly' else g.index-pd.Timedelta(days=6)
        ends=starts+pd.Timedelta(days=1 if name=='daily' else 7) if name!='monthly' else starts+pd.offsets.MonthBegin(1)
        expected=((ends-starts)/pd.Timedelta(hours=1)).to_numpy(int)
        frame=pd.DataFrame({'period_label':g.index,'period_start':starts,'period_end_exclusive':ends,
            'energy_kwh_provisional':g['sum'].values,'mean_hourly_energy_kwh_provisional':g['mean'].values,
            'observed_hours':g['count'].values,'expected_hours':expected,'coverage':g['count'].values/expected,
            'complete':g['count'].values==expected,'full_period_available_at':ends})
        path=OUT/f'{name}_dataset.csv';frame.to_csv(path,index=False,float_format='%.17g',encoding='utf-8-sig')
        full=frame[frame.complete];pre=full[pd.to_datetime(full.full_period_available_at)<=pd.Timestamp('2025-04-01')]
        means=full.mean_hourly_energy_kwh_provisional
        datasets[name]={'file':str(path),'frequency':freq,'bucket_count':len(frame),'complete_periods':len(full),
            'pre_April_complete_periods':len(pre),'partial_periods':frame[~frame.complete].to_dict(orient='records'),
            'mean_hourly_rate_cv_percent':float(means.std(ddof=1)/means.mean()*100),
            'period_energy_cv_percent':float(full.energy_kwh_provisional.std(ddof=1)/full.energy_kwh_provisional.mean()*100),
            'pre_April_rate_lag1_correlation':float(pre.mean_hourly_energy_kwh_provisional.autocorr(1)) if len(pre)>5 else None,
            'pre_April_rate_lag7_correlation':float(pre.mean_hourly_energy_kwh_provisional.autocorr(7)) if len(pre)>14 else None,
            'partial_periods_excluded_from_statistics':True}
        inventory.append({'file_id':str(uuid.uuid5(uuid.NAMESPACE_URL,'aws-resampling-review/'+path.name)),
            'project_id':None,'relative_path':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'original_path':str(SOURCE),'copy_status':'new_derived_dataset','ai_generated':True,'verified':True})
        assert int(frame.observed_hours.sum())==3600
        assert abs(frame.energy_kwh_provisional.sum()-y.sum())<1e-6
    result={'status':'verified','source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'source_resolution_hours':1,'datasets':datasets,'source_hour_count':3600,
        'hourly_rate_cv_percent':float(y.std(ddof=1)/y.mean()*100),'guide_evaluation':'May10-May31 hourly505 targets; coarser predictions must be mapped to hours using only information available before each origin',
        'limitations':['Monthly total differences reflect calendar month lengths as well as load; examine normalized hourly rate.',
            'Smoothing and lower variation do not establish forecast skill; do not compare MAE across aggregation units.',
            'No future incomplete day/week/month is used as an observed complete-period feature.',
            'No forecast model fitted in this resampling feasibility review; existing post-EDA limitations remain.'],
        'copied_original_files':0,'copy_failures':0,'originals_preserved':True,'files':inventory}
    (OUT/'resampling_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    print(json.dumps({k:{'complete':v['complete_periods'],'pre_April':v['pre_April_complete_periods'],'mean_rate_cv_percent':v['mean_hourly_rate_cv_percent']} for k,v in datasets.items()},ensure_ascii=True))
if __name__=='__main__':main()
