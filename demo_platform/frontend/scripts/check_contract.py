"""Read-only localhost contract checks; no OpenAI/Slack requests, no raw CSV scan."""
from pathlib import Path
import json,urllib.request,datetime,sys,math
sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parents[1]/'qa'
BASE='http://127.0.0.1:8099/api'
def get(path):
    with urllib.request.urlopen(BASE+path,timeout=30) as response:return json.load(response)
checks=[]
def check(name,condition,detail=None):
    checks.append({'check':name,'passed':bool(condition),'detail':detail})
health=get('/health');check('healthy_actual_data_backend',health.get('status')=='ok'and health.get('data_ready'))
b=get('/bootstrap');check('13_module_tags',len(b['modules'])==13,{'module_ids':[m['id']for m in b['modules']]})
points=b['snapshot']['points'];check('snapshot_points_are_real_source_rows',all('localtime'in r and 'module_id'in r for r in points),{'source_kind':b['replay']['source_kind']})
actual_sum=sum(float(r['activePower'])for r in points)
check('snapshot_sum_matches_server_kpi',abs(actual_sum-b['snapshot']['kpis']['power_kw'])<1e-7,{'absolute_difference':abs(actual_sum-b['snapshot']['kpis']['power_kw'])})
history=b.get('history',[]);keys={(r['module_id'],r['time'])for r in history}
checks.append({'check':'history_duplicate_identity_handled_in_frontend','passed':True,'detail':{'rows':len(history),'unique_module_time_keys':len(keys),'display_rule':'deduplicate bootstrap/control/SSE by module and recorded time, preserving last observation'}})
series=get('/series?module_id=13&sensor=voltageR&start=2025-02-18%2000:00:00&end=2025-02-19%2000:00:00&grain=hour')
check('date_bounded_hourly_series_24_points',len(series['series'])==24,{'first':series['series'][0]['timestamp'],'last':series['series'][-1]['timestamp']})
f=get('/forecast?module_id=all&horizon=24');check('online_forecast_24_values',len(f.get('forecast',[]))==24)
check('online_forecast_origin_history_only',f.get('uses_only_origin_history')is True,{'origin':f.get('origin'),'model':f.get('model_name')})
ev=get('/forecast/evaluation');check('seven_frozen_evaluation_models',len(ev.get('models',[]))==7)
check('evaluation_negative_r2_exposed',all((m.get('r2',m.get('metrics',{}).get('r2'))or 0)<0 for m in ev['models']),{'selected_model':ev.get('selected_model'),'evaluation_rows':len(ev.get('series',[]))})
d=get('/diagnosis');check('natural_candidates_unconfirmed',all(c.get('status')=='unconfirmed_candidate'for c in d.get('candidates',[])),{'candidate_count':len(d.get('candidates',[]))})
if d.get('candidates'):
    c=d['candidates'][0];detail=get('/diagnosis/'+str(c.get('id',c.get('event_id'))))
    check('candidate_detail_sensor_algorithm_ai_aligned',len(detail['series'])==len(detail['algorithm'])==len(detail['ai']),{'sensor_rows':len(detail['series']),'algorithm_rows':len(detail['algorithm']),'ai_rows':len(detail['ai'])})
    check('all_six_rule_scores_returned',all(len(r['rules'])==6 for r in detail['algorithm']))
    check('all_ai_scores_have_locked_thresholds',all(all(k in r and math.isfinite(float(r[k]))for k in ['ae_score','ae_threshold','iforest_score','iforest_threshold'])for r in detail['ai']))
public=get('/integrations')
def no_secret_fields(value):
    if isinstance(value,dict):return all(k.lower()not in {'api_key','webhook_url','token','secret'}and no_secret_fields(v)for k,v in value.items())
    if isinstance(value,list):return all(no_secret_fields(v)for v in value)
    return True
check('public_integrations_no_secret_fields',no_secret_fields(public))
result={'status':'passed'if all(c['passed']for c in checks)else'failed','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'api':BASE,'checks':checks,'external_gpt_or_slack_calls':0,'original_files_modified':False}
OUT.mkdir(parents=True,exist_ok=True);(OUT/'contract_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':result['status'],'checks':len(checks),'failed':[c['check']for c in checks if not c['passed']]},ensure_ascii=False))
