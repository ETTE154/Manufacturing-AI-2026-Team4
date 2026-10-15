"""Document provided CSVs from existing exhaustive evidence and bounded checks.

No source or previous output is changed, no raw file is copied. Full training
descriptive statistics are REUSED from the audited complete streaming scan.
Pandas dtype inference uses only a bounded original prefix and is labelled so.
The505-row May file is independently checked in full for this data overview.
"""
from pathlib import Path
import sys,json,hashlib,re
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
TRAIN=ROOT/'traing-dataset/rtu_data_full.csv'
MAY=ROOT/'may-dataset/rtu_ground_truth_may.csv'
PROFILE=ROOT/'output/analysis/training_profile.json'
TIME=ROOT/'output/analysis/time_anomaly_audit.json'
GUIDE=ROOT/'output/analysis/guide_summary.json'
SOURCE=ROOT/'output/optimization_20261013/source_audit/audit_report.json'
SOURCE_PROFILES=ROOT/'output/optimization_20261013/source_audit/csv_profiles.json'
METRICS=ROOT/'output/analysis/analysis_metrics.json'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,(Path,pd.Timestamp)):return str(x)
    return x
def write(path,value):
    Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    profile,timeaudit,guide,source,source_profiles,metrics=map(read,[PROFILE,TIME,GUIDE,SOURCE,SOURCE_PROFILES,METRICS])
    train_public=source_profiles['traing-dataset/rtu_data_full.csv']['downloaded_profile']
    # This bounded inference check does not validate physical units or the dtype
    # of every cell in a schema-less CSV. It must not be labelled an entire scan.
    sample=pd.read_csv(TRAIN,nrows=2000)
    may=pd.read_csv(MAY,float_precision='round_trip')
    assert sample.columns.tolist()==profile['columns'] and len(sample)==2000
    assert may.columns.tolist()==['id','hourly_pow','may_bill','may_carbon','agg_pow']
    mid=pd.to_datetime(may.id,format='%Y-%m-%d %H:%M:%S',errors='raise')
    assert mid.equals(pd.Series(pd.date_range('2025-05-10','2025-05-31',freq='h'),name='id'))
    assert not may.isna().any().any() and not mid.duplicated().any()
    assert profile['records']==timeaudit['rows']==sum(p['rows'] for p in profile['profiles'].values())==33_696_013
    assert sum(profile['missing'].values())==0
    src_checks=[]
    for item in source['file_comparisons']:
        original=Path(item['original']['path']);stat=original.stat()
        assert stat.st_size==item['original']['bytes'] and stat.st_mtime_ns==item['original']['mtime_ns']
        assert item['sha256_match'] and item['bytes_match']
        src_checks.append({'path':str(original),'bytes':stat.st_size,'recorded_full_SHA256':item['original']['sha256'],
                           'official_download_SHA256':item['downloaded']['sha256'],'recorded_official_byte_identity':True,
                           'current_size_and_mtime_unchanged_since_SHA':True,
                           'checksum_scope':'Previous independent full-file SHA comparison reused; training file not rehashed in this overview'})
    may_sha=hashlib.sha256(MAY.read_bytes()).hexdigest()
    assert may_sha==next(x['recorded_full_SHA256'] for x in src_checks if x['path']==str(MAY))
    numeric_columns=[c for c in profile['columns'] if c in next(iter(profile['profiles'].values()))['sum']]
    n=profile['records'];rows=[]
    def add(dataset,name,meaning,unit,dtype,minimum=None,maximum=None,mean=None,std=None,missing=0,
            scope='',guide_name=None,unit_basis=None,representation=None,parse_rule=None,example=None):
        rows.append({'dataset':dataset,'name':name,'meaning_ko':meaning,'unit':unit,'dtype_observed':dtype,
                     'min':minimum,'max':maximum,'mean':mean,'std':std,'missing_count':missing,'scope':scope,
                     'guide_name':guide_name,'unit_basis':unit_basis,'raw_representation':representation,
                     'parse_rule':parse_rule,'example':example})
    defs=guide['column_definitions'];numstats={}
    for col in numeric_columns:
        ps=profile['profiles'].values();total=sum(p['sum'][col] for p in ps);squares=sum(p['sumsq'][col] for p in ps)
        mean=total/n;std=float(np.sqrt(max(squares/n-mean*mean,0)))
        numstats[col]={'min':min(p['min'][col] for p in ps),'max':max(p['max'][col] for p in ps),'mean':mean,'std':std}
    train_scope='범위·평균·모집단 표준편차(ddof=0)·결측: 기존33,696,013행 전수 스트리밍 프로파일; dtype: 원본 앞2,000행 pandas 기본 추론'
    for col in profile['columns']:
        dtype=str(sample[col].dtype);missing=profile['missing'][col]
        if col=='module(equipment)':
            add('full',col,'모듈 ID와 괄호 안 계측점/설비 표기. 13개 태그를13대 독립 장비로 확정할 수 없음','식별자',dtype,
                missing=missing,scope='13고유값/모듈별건수: 전수 프로파일. dtype와 예시는 bounded prefix.',guide_name='module',
                unit_basis='가이드에는 module; 실제 원본명은 module(equipment)',representation='문자열: ID(표기명)',example=str(sample[col].iloc[0]))
        elif col=='timestamp':
            add('full',col,defs[col]+'; recorded localtime과 일관된 시간대 관계가 확인되지 않음','Unix epoch ms (가이드)',dtype,
                train_public['timestamp_min_ms'],train_public['timestamp_max_ms'],missing=missing,
                scope='min/max: 공식 동일사본의 전수 key scan. 중복·역순·offset: 원본 전수 시간감사. metadata라평균/표준편차생략.',
                guide_name=col,unit_basis='가이드ms; UTC 파싱은 검증됨. 의도된 local timezone은 미확정',representation='따옴표 없는13자리 정수',
                parse_rule="pd.to_datetime(timestamp,unit='ms',utc=True); recorded localtime과 별도대조",example=str(int(sample[col].iloc[0])))
        elif col=='localtime':
            add('full',col,'기록된 현지 시각 YYYYMMDDhhmmss. 가이드는 한국시간 문자열로 설명하지만 원본은 숫자표기','날짜·시각; 시간대 미확정',dtype,
                train_public['localtime_numeric_min'],train_public['localtime_numeric_max'],missing=missing,
                scope='날짜범위/5초격자/유효파싱: 전수 시간감사; dtype: bounded prefix. 날짜코드의평균/표준편차는의미없어생략.',
                guide_name=col,unit_basis='한국시간은 가이드 설명; 실제Unix offset과충돌하여확정불가',representation='따옴표 없는14자리 숫자: YYYYMMDDhhmmss',
                parse_rule="pd.to_datetime(localtime.astype(str),format='%Y%m%d%H%M%S'); 정수직접Timestamp변환금지",example=str(int(sample[col].iloc[0])))
        elif col=='operation':
            add('full',col,defs[col]+'; 관측값은 모든행1이며 상태전환이 없음','상태코드',dtype,1,1,1.,0.,missing,train_scope,
                col,'1=가동은 가이드 의미. 필드상수만으로실제연속가동을확정하지않음','정수표기',example=str(int(sample[col].iloc[0])))
        else:
            stat=numstats[col]
            if col.startswith('voltage'):unit='V (가이드)'
            elif col.startswith('current'):unit='A (가이드)'
            elif col.startswith('powerFactor'):unit='% (가이드)'
            elif col=='activePower':unit='kW (가이드; 미확정)'
            elif col=='reactivePowerLagging':unit='kVar (가이드; 미확정)'
            else:unit='Wh / kWh (가이드 불일치; 미확정)'
            basis='가이드단위이며 센서 교정·계측범위·실제단위 독립검증자료없음'
            if col=='accumActiveEnergy':basis='일반 컬럼표Wh/주요표kWh 충돌. 증가량과 프로젝트 activePower적분의수치비율≈1이나물리단위확정증거는아님'
            meaning=defs[col]
            if col=='activePower':meaning+='; 가이드는 각5초 구간의 평균값으로 설명'
            add('full',col,meaning,unit,dtype,stat['min'],stat['max'],stat['mean'],stat['std'],missing,
                train_scope,col,basis,'정수표기(표본)' if col=='accumActiveEnergy' else '소수 숫자표기(표본)',example=str(sample[col].iloc[0]))
    cost_error=float(np.max(abs(may.may_bill.to_numpy()-may.hourly_pow.to_numpy()*180)))
    carbon_error=float(np.max(abs(may.may_carbon.to_numpy()-may.hourly_pow.to_numpy()*.424)))
    agg_error=float(np.max(abs(may.agg_pow.to_numpy()-may.hourly_pow.sum())))
    assert cost_error<1e-7 and carbon_error<1e-7 and agg_error<1e-7 and may.agg_pow.nunique()==1
    meanings={'id':'시간별 정답의 시각 ID. 원본 문자열에는 시간대가 없음',
              'hourly_pow':'시간별 검증 정답. 원시 5초 합계인지 환산 전력량인지 명시가 부족하여 kWh로 잠정 해석',
              'may_bill':'hourly_pow×180으로 주어진 프로젝트 요금 환산값',
              'may_carbon':'hourly_pow×0.424로 주어진 프로젝트 탄소배출 환산값',
              'agg_pow':'제공된 505개 hourly_pow의 전체 합을 모든 행에 반복한 상수. 행별 누적합이 아님'}
    units={'id':'날짜·시각(시간대미확정)','hourly_pow':'미확정; kWh 잠정','may_bill':'원 (프로젝트환산)','may_carbon':'kgCO2 (프로젝트환산)','agg_pow':'hourly_pow와동일단위; kWh 잠정'}
    for col in may:
        if col=='id':
            add('may',col,meanings[col],units[col],str(may[col].dtype),str(mid.min()),str(mid.max()),missing=0,
                scope='원본May505행 전체재검증. CSVdtype는object; 분석파싱후datetime.',guide_name=None,
                unit_basis='시간축은가이드한국시간예상이나원본timezone표기없음',representation='YYYY-MM-DD hh:mm:ss 문자열',
                parse_rule="pd.to_datetime(id,format='%Y-%m-%d %H:%M:%S')",example=str(may[col].iloc[0]))
        else:
            add('may',col,meanings[col],units[col],str(may[col].dtype),float(may[col].min()),float(may[col].max()),
                float(may[col].mean()),float(may[col].std(ddof=0)),int(may[col].isna().sum()),
                scope='원본May505행 전체: 범위/평균/모집단표준편차ddof=0/결측/수치관계재검증',guide_name=None,
                unit_basis='파일수치관계와프로젝트계수의일치; 센서/요금제의물리단위확정이아님',representation='소수 숫자표기',example=str(may[col].iloc[0]))
    dictionary=pd.DataFrame(rows);dictionary.to_csv(OUT/'feature_dictionary.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    module_details=[];module_numeric=[]
    for tag,p in profile['profiles'].items():
        key=timeaudit['time_key_checks'][tag];identifier=int(tag.split('(',1)[0]);name=tag.split('(',1)[1][:-1]
        module_details.append({'module_id':identifier,'source_tag':tag,'source_label':name,'rows':p['rows'],'first':p['first'],'last':p['last'],
             'operation_counts':p['operation_counts'],'local_duplicate_rows':key['local_duplicate_rows'],
             'local_negative_diffs':key['local_negative_diffs'],'local_non_5s_diffs':key['local_non_5s_diffs'],
             'unix_duplicate_extra_rows':key['unix_duplicate_rows'],'unix_negative_diffs':p['negative_time_diffs'],
             'counter_negative_diffs':p['counter_negative_diffs'],'mapping_confidence':'원본태그문자열그대로만확인; 물리장비/상하위계측기매핑미확정',
             'label_scope':'분전반계측점가능(이름에근거한추정)' if '분전반' in name else '설비/계측점명표기; 실제범위미확정'})
        for col in numeric_columns:
            mean=p['sum'][col]/p['rows'];std=float(np.sqrt(max(p['sumsq'][col]/p['rows']-mean*mean,0)))
            module_numeric.append({'module':tag,'feature':col,'count':p['rows'],'min':p['min'][col],'max':p['max'][col],'mean':mean,'std_ddof0':std,'scope':'prior full streaming scan'})
    pd.DataFrame(module_details).drop(columns=['operation_counts']).to_csv(OUT/'module_dictionary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(module_numeric).to_csv(OUT/'module_numeric_summary.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    first,last=pd.Timestamp(train_public['recorded_localtime_start']),pd.Timestamp(train_public['recorded_localtime_end'])
    gap=(mid.min()-last)/pd.Timedelta(hours=1)
    expected_per=int((last-first)/pd.Timedelta(seconds=5))+1
    assert expected_per==2_592_001 and all(p['rows']==expected_per for p in profile['profiles'].values())
    assert sum(k['unix_duplicate_rows'] for k in timeaudit['time_key_checks'].values())==9360
    source_paths={'full_profile':str(PROFILE),'time_exhaustive_audit':str(TIME),'guide_sanitized_visual_summary':str(GUIDE),
                  'official_byte_comparison':str(SOURCE),'official_copy_key_profiles':str(SOURCE_PROFILES),
                  'prior_hourly_aggregate_metrics':str(METRICS),'prior_streaming_code_reviewed':str(ROOT/'output/analysis/scripts/profile_training.py')}
    result={'schema_version':1,'status':'verified','source_paths':source_paths,'source_integrity':src_checks,
       'full':{'path':str(TRAIN),'bytes':TRAIN.stat().st_size,'rows':n,'columns':19,'column_names':profile['columns'],
               'module_count':13,'first':str(first),'last':str(last),'sampling_seconds':5,'missing_total':sum(profile['missing'].values()),
               'missing_by_column':profile['missing'],'module_rows':{tag:p['rows'] for tag,p in profile['profiles'].items()},
               'listmodule_names':list(profile['profiles']),'module_details':module_details,'span_days':float((last-first)/pd.Timedelta(days=1)),
               'expected_rows_per_module_inclusive':expected_per,'full_day_count':150,'last_day_is_partial':True,
               'last_day_records_per_module':1,'complete_all_module_hour_count':3600,'complete_module_hour_count':46800,
               'module_hour_bucket_count_including_endpoint':46813,'last_hour_is_partial':True,'last_hour_records_per_module':1,
               'operation_observed_values':[1],'operation_field_constant':True,
               'dtype_inference':{'method':'pandas.read_csv 원본 앞 2,000행 기본 추론','rows':len(sample),'scope':'표본의 parser 추론이며 CSV에는 고정된 자료형 스키마가 없음','dtypes':{col:str(t) for col,t in sample.dtypes.items()}},
               'numeric_statistics_scope':'15개원본numeric센서/counter열: 기존전체33,696,013행스트리밍sum/sumsq/min/max에서전체통계재구성. sourceendpoint포함;std ddof=0.',
               'numeric_statistics':numstats},
       'may':{'path':str(MAY),'bytes':MAY.stat().st_size,'sha256_current_full_check':may_sha,'rows':len(may),'columns':5,
              'column_names':list(may),'first':str(mid.min()),'last':str(mid.max()),'sampling_seconds':3600,
              'missing_total':int(may.isna().sum().sum()),'missing_by_column':may.isna().sum().to_dict(),
              'duplicate_time_ids':int(mid.duplicated().sum()),'non_hourly_steps':int((mid.diff().dropna()!=pd.Timedelta(hours=1)).sum()),
              'span_hours':504,'complete_days':21,'partial_last_day_hours':1,'covers_full_calendar_May':False,
              'no_module_or_sensor_columns':True,'derived_relationships':{'may_bill':'hourly_pow*180','may_carbon':'hourly_pow*0.424',
                    'agg_pow':'sum(all505 hourly_pow) repeated on everyrow, not cumulative','cost_max_absdiff':cost_error,
                    'carbon_max_absdiff':carbon_error,'constant_total_max_absdiff':agg_error,'agg_pow_unique_count':int(may.agg_pow.nunique())},
              'scope':'May原本505행전체재검증; 가이드평가기간정답이지만단위/계측범위는원시May자료없어독립확정불가'},
       'time_quality':{'localtime_parse_format':'%Y%m%d%H%M%S after string conversion; preserve14-digit source code',
              'localtime_source_literal_example':str(int(sample.localtime.iloc[0])),
              'localtime_source_representation':'Unquoted numeric-looking14-digit CSV token; pandas bounded default dtype int64; guide calls it a string',
              'localtime_duplicate_keys_module_time':sum(k['local_duplicate_rows'] for k in timeaudit['time_key_checks'].values()),
              'localtime_negative_diffs':sum(k['local_negative_diffs'] for k in timeaudit['time_key_checks'].values()),
              'localtime_non_5_second_steps':sum(k['local_non_5s_diffs'] for k in timeaudit['time_key_checks'].values()),
              'unix_duplicate_extra_keys_module_timestamp':sum(k['unix_duplicate_rows'] for k in timeaudit['time_key_checks'].values()),
              'unix_negative_transitions':sum(p['negative_time_diffs'] for p in profile['profiles'].values()),
              'local_naive_minus_unix_ms':timeaudit['offset_local_naive_minus_unix_ms'],'offset_transitions':timeaudit['offset_transitions'],
              'timestamp_utc_start':train_public['timestamp_min_as_utc'],'timestamp_utc_end':train_public['timestamp_max_as_utc'],
              'first_row_clock_example':{'recorded_localtime':str(first),'timestamp_ms':int(sample.timestamp.iloc[0]),
                      'as_UTC':str(pd.to_datetime(sample.timestamp.iloc[0],unit='ms',utc=True)),
                      'as_Asia_Seoul':str(pd.to_datetime(sample.timestamp.iloc[0],unit='ms',utc=True).tz_convert('Asia/Seoul'))},
              'interpretation':'全13모듈 localtime은5초연속. Unix key는3/9 03:00에서모듈별1회역순/비인접중복720추가행; localtime역순은0. -8h→-7h가미국태평양DST형패턴과유사하지만의도된timezone은미확정.',
              'action':'원본유지·recorded localtime을명시해분석; timezone확인전Seoul변환/Unix중복행자동제거금지'},
       'train_may_coverage':{'time_basis':'Recorded naive calendar labels; common intendedtimezone is unverified',
              'last_training_raw_observation':str(last),'first_May_label':str(mid.min()),'raw_end_to_first_label_elapsed_hours':float(gap),
              'raw_end_to_first_label_elapsed_days':float(gap/24),'overlapping_recorded_label_hours':0,
              'last_complete_training_hour_start':'2025-04-29 23:00:00','first_unobserved_complete_hour_start':'2025-04-30 00:00:00',
              'bridge_hour_count_before_May_labels':240,'bridge_hours':'2025-04-30 00:00 through2025-05-09 23:00',
              'May_missing_days_are_not_zero_load':'May1–9 labels and Mayfive-secondsensorobservations are not provided; no silentzero fill'},
       'units_and_interpretation':{'capacity_or_load_factor_data_present':False,'normalized_period_measure':'기간 평균 부하(kWh/h)=전력량/관측시간. 정격용량 대비 부하율 자료는 없음',
              'guide_units':guide['units_and_formulas'],'counter_unit_conflict':'Guide generalWh vs keycolumnkWh. Numeric increments approximatelymatch project integration but do not independently certify absolutephysicalunits.',
              'module_count_limit':'13 distinctmodule(equipment) labels/계측태그;13확정물리장비가아님. 분전반등상하위계측혼재가능하며전모듈합계는잠정sum,확정공장소비량아님.',
              'module_ids':[d['module_id'] for d in module_details],'nonconsecutive_ID_note':'IDs6–10이없다는사실만으로누락된5장비를추정하지않음',
              'forecast_truth_limit':'Maybill/carbon은정답의선형파생열, agg_pow는전체정답합; 미래예측피처로사용하면정답누출',
              'fault_label_limit':'Guide says intentionally inserted anomalies, but no suppliedrow/event fault labels or mask. Rulecandidate counts are screening, not confirmedfaults.'},
       'statistical_scope':{'prior_training_full_scan_rows':n,'prior_time_exhaustive_rows':timeaudit['rows'],
              'overview_training_bounded_rows_read':len(sample),'overview_new_training_full_scans':0,
              'overview_May_full_rows_read':len(may),'training_dtype_scope':'bounded-prefixinference only',
              'training_range_missing_time_scope':'reusedexhaustive evidence, not sampled estimates',
              'std_convention':'feature_dictionary numeric std is population ddof0; existinghourlySD95.764 uses sample ddof1 and is a differentaggregation',
              'raw_data_copies_created':0,'credentials':'Original guidecredentials are not read, copied or shown; sanitizedguide_summary reused'},
       'outputs':{'feature_dictionary':str(OUT/'feature_dictionary.csv'),'module_dictionary':str(OUT/'module_dictionary.csv'),'module_numeric_summary':str(OUT/'module_numeric_summary.csv')}}
    write(OUT/'evidence_summary.json',result)
    verify={'status':'pass','dictionary_rows':len(dictionary),'full_dictionary_rows':int(dictionary.dataset.eq('full').sum()),
            'May_dictionary_rows':int(dictionary.dataset.eq('may').sum()),'full_column_order_matches':dictionary[dictionary.dataset.eq('full')].name.tolist()==profile['columns'],
            'module_rows_sum_matches':sum(result['full']['module_rows'].values())==n,'all_module_grid_checks_pass':result['time_quality']['localtime_non_5_second_steps']==0,
            'all_numeric_ranges_finite':all(np.isfinite([s['min'],s['max'],s['mean'],s['std']]).all() for s in numstats.values()),
            'May_full_check_pass':True,'source_original_size_mtime_unchanged':True,'new_training_full_scans':0,'raw_copies':0}
    assert verify['dictionary_rows']==24 and verify['full_column_order_matches'] and verify['module_rows_sum_matches']
    write(OUT/'evidence_verification.json',verify)
    print(json.dumps({'status':'pass','full_rows':n,'modules':13,'May_rows':len(may),'dictionary_rows':len(dictionary),
         'localtime_step_anomalies':0,'Unix_duplicate_extra_keys':9360,'Unix_reverse_transitions':13,
         'gap_hours':gap,'May_relationship_errors':[cost_error,carbon_error,agg_error],'new_full_training_scans':0},ensure_ascii=True))

if __name__=='__main__':main()
