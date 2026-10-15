"""Meaningful local integration checks. All report/chat calls prohibit external IO."""
from pathlib import Path
import urllib.request,urllib.error,json,time,datetime,sys
OUT=Path(__file__).resolve().parent
BASE='http://127.0.0.1:8099/api';checks=[]
def call(path,body=None,headers=None):
    req=urllib.request.Request(BASE+path,data=json.dumps(body).encode()if body is not None else None,headers={'Content-Type':'application/json',**(headers or{})},method='POST'if body is not None else'GET')
    with urllib.request.urlopen(req,timeout=20)as r:return json.load(r)
def check(name,condition,details=None):
    checks.append({'name':name,'passed':bool(condition),'details':details});assert condition,name
def expect_error(name,path,body,status=400):
    try:call(path,body)
    except urllib.error.HTTPError as e:check(name,e.code==status);return
    check(name,False)
def test():
    health=call('/health');check('data_ai_forecast_ready',health['data_ready']and health['diagnosis_ready']and health['forecast_ready'])
    call('/replay',{'action':'scenario','scenario_id':'recorded_pf'});meta=call('/bootstrap');check('13_real_modules',len(meta['modules'])==13)
    snapshot=meta['snapshot'];raw=snapshot['points'];check('instantaneous_power_is_sensor_sum',abs(snapshot['kpis']['power_kw']-sum(r['activePower']for r in raw))<1e-8)
    check('source_and_received_clocks_separate',snapshot['timestamp'].startswith('2025-')and snapshot['received_at'].endswith('+00:00'))
    aggregated=call('/series?module_id=all&sensor=energy_kwh&grain=hour');check('no_incomplete_future_hour',all(r['timestamp']<'2025-03-13 19:00:00'for r in aggregated['series']))
    raw_series=call('/series?module_id=all&sensor=activePower&grain=raw');check('raw_default_end_current_observed_tick',raw_series['series'][-1]['timestamp']==snapshot['timestamp'])
    prediction=call('/forecast?module_id=17&horizon=24');check('prefix_only_forecast',prediction['uses_only_origin_history']and prediction['training_cutoff']<prediction['origin']and prediction['origin']=='2025-03-13 19:00:00')
    check('module_fit_not_aggregate_allocation',prediction['module_model_is_total_proportional_allocation']is False and len(prediction['forecast'])==24)
    frozen=call('/forecast/evaluation');check('7_models_505_labels',len(frozen['models'])==7 and len(frozen['series'])==505)
    check('negative_r2_not_hidden',any((m.get('metrics',{}).get('r2')or 0)<0 for m in frozen['models']))
    diagnosis=call('/diagnosis');check('4_sustained_unconfirmed_events',len(diagnosis['candidates'])==4 and all(c['point_count']>=3 for c in diagnosis['candidates']))
    detail=call('/diagnosis/abnormal_pattern_0093');check('complete_model_specific_score_paths',len(detail['series'])==72 and len(detail['ai'])==72 and all(len(a['rules'])==6 for a in detail['algorithm']))
    check('source_fault_label_not_invented',all(c.get('status')!='confirmed_fault'for c in diagnosis['candidates']))
    before=call('/logs')['total'];call('/replay',{'action':'seek','timestamp':'2025-03-13 19:45:05'});call('/replay',{'action':'speed','speed':20});call('/replay',{'action':'play'});time.sleep(1.3);paused=call('/replay',{'action':'pause'});after=call('/logs')['total']
    check('durable_13_rows_per_source_tick',after>before and(after-before)%13==0,{'new_records':after-before})
    check('real_17_three_tick_alarm',any(a['module_id']=='17'and a['time']=='2025-03-13 19:45:15'and a['alarm_policy']=='3_consecutive_ticks'for a in call('/alarms')['rows']))
    time.sleep(.4);check('pause_stops_new_logs',call('/logs')['total']==after)
    expect_error('report_rejects_future_asof','/reports/generate',{'end':'2025-04-29 23:00:00','local_only':True})
    expect_error('chat_rejects_future_context','/chat',{'question':'전력','context':{'cursor_time':'2025-04-29 23:00:00'},'local_only':True})
    report=call('/reports/generate',{'module_id':'17','local_only':True});check('local_report_truthful_168_complete_hours',report['status']=='local_summary'and report['model']=='local_statistics'and report['facts']['observed_hours']==168)
    old_report=call('/reports/generate',{'module_id':'17','end':'2025-02-18 13:00:00','local_only':True});check('historical_report_matches_requested_forecast_origin',old_report['facts']['forecast_origin']=='2025-02-18 13:00:00')
    old_chat=call('/chat',{'question':'지난 일주일 이상 알람','context':{'module_id':'all','cursor_time':'2025-02-18 13:00:00'},'local_only':True});check('fault_chat_has_no_future_events',all('2025-02-11 13:00:00'<=r['time']<'2025-02-18 13:00:00'for r in old_chat['rows']))
    check('weekly_actual_report_excludes_synthetic_alarms',all(a['source_kind']=='recorded'for a in report['facts']['logged_alarm_candidates']))
    chat=call('/chat',{'question':'설비 목록','context':{'module_id':'all'},'local_only':True});check('real_allowlisted_local_data_search',chat['source']=='local_data_search'and len(chat['rows'])==13)
    saved=call('/reports/schedule',{'enabled':False,'weekday':0,'time':'09:25','module_id':'17'});check('schedule_persists_module_and_minutes',saved['module_id']=='17'and saved['minute']==25 and call('/reports')['schedule']['time']=='09:25')
    expect_error('webhook_ssrf_rejected','/integrations/slack',{'webhook_url':'http://127.0.0.1:8099/api/health','enabled':True})
    try:call('/replay',{'action':'play'},{'Origin':'https://untrusted.example'})
    except urllib.error.HTTPError as e:check('cross_site_mutation_rejected',e.code==403)
    else:check('cross_site_mutation_rejected',False)
    types=['load_imbalance','reactive_power_surge','power_factor_drop','abnormal_pattern','night_operation','current_only_increase']
    for kind in types:
        result=call('/replay',{'action':'scenario','scenario_id':'synthetic_'+kind});start=datetime.datetime.fromisoformat(result['replay']['start']);t=start+datetime.timedelta(seconds=150)
        frame=call('/replay',{'action':'seek','timestamp':str(t)})['snapshot'];r=next(r for r in frame['points']if r['module_id']=='17')
        check('explicit_synthetic_source_'+kind,r['source_kind']=='synthetic'and r['augmentation_type']==kind and r['diagnosis']['source_kind']=='synthetic')
        if kind=='power_factor_drop':
            call('/replay',{'action':'play'});time.sleep(1.1);call('/replay',{'action':'pause'});alarms=[a for a in call('/alarms')['rows']if a.get('scenario_id')=='synthetic_'+kind]
            check('synthetic_alarm_retains_scenario_identity',bool(alarms))
            event=call('/diagnosis/'+alarms[0]['id']);original=call('/series?module_id=17&sensor=powerFactorR&grain=raw&start=2025-03-14%2013:02:30&end=2025-03-14%2013:02:35')['series'][0]['value'];transformed=next(r for r in event['series']if r['time']=='2025-03-14 13:02:30')['powerFactorR']
            check('synthetic_detail_reproduces_augmented_values',event['source_kind']=='synthetic'and abs(transformed-original*.55)<1e-8)
    # Restore reviewable demo: original case, paused at the anomaly, daily9:00 disabled.
    call('/replay',{'action':'scenario','scenario_id':'recorded_pf'});call('/reports/schedule',{'enabled':False,'weekday':0,'time':'09:00','module_id':'all'})
    integrations=call('/integrations');check('credentials_absent_from_client_response','sk-'not in json.dumps(integrations)and 'webhook_url'not in integrations.get('slack',{}))
if __name__=='__main__':
    status='passed'
    try:test()
    except Exception as e:status='failed';checks.append({'exception_type':type(e).__name__,'message':str(e)})
    result={'status':status,'checks':checks,'external_sensor_transmission':False,'real_slack_delivery':'not_configured','real_gpt_generation':'not_verified_authentication_failed_or_unconfigured'}
    (OUT/'backend_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({'status':status,'checks':len(checks)},ensure_ascii=False));sys.exit(0 if status=='passed'else 1)
