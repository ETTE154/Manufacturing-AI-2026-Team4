"""Offline chat scope checks: real indexed source, fake DB/config/GPT, no models.

Uses the real /api/chat ASGI route without lifespan or network/server access.
Production SQLite is never imported. Original data and cached index are readonly.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import asyncio,gc,hashlib,json,shutil,tempfile,types
from unittest.mock import patch
import pandas as pd

OUT=Path(__file__).resolve().parent;APP=OUT.parent
temporary=tempfile.TemporaryDirectory(prefix='chat_scope_',dir=OUT);DATA=Path(temporary.name)
config=types.ModuleType('config');config.APP=APP;config.ROOT=APP.parent;config.DATA=DATA;config.MODEL='fake-approved-model'
key_reads=[]
config.CONFIG=types.SimpleNamespace(api_key=lambda:key_reads.append('mock')or'FAKE-NO-TRANSPORT',public=lambda:{},slack_url=lambda:None)
sys.modules['config']=config
class FakeDB:
    def __init__(self):self.queries=[];self.writes=0;self.rows=[]
    def get(self,key,default=None):return True if key=='gpt_data_transfer_approved'else default
    def alarms(self,module='all',limit=30,start=None,end=None,source_kind=None):
        self.queries.append(dict(module=module,start=start,end=end,source_kind=source_kind))
        rows=[r for r in self.rows if(module=='all'or r['module_id']==module)and(start is None or r['time']>=start)and(end is None or r['time']<end)and(source_kind is None or r['source_kind']==source_kind)]
        return {'rows':rows[:limit],'total':len(rows)}
db=FakeDB();store=types.ModuleType('store');store.DB=db;store.now=lambda:'2026-10-15T00:00:00Z';store.alarm_id=lambda event:'unused'
sys.modules['store']=store;sys.path.insert(0,str(APP/'backend'))
import services
import app as api
from data_source import SourceStore,SOURCE,iso
shutil.copyfile(APP/'data/source_index.json',DATA/'source_index.json')
checks=[]
def check(name,value,details=None):
    checks.append({'name':name,'passed':bool(value),'details':details});assert value,name
async def post(body):
    messages=[];sent=False;forever=asyncio.Event();payload=json.dumps(body).encode()
    async def receive():
        nonlocal sent
        if not sent:sent=True;return {'type':'http.request','body':payload,'more_body':False}
        await forever.wait()
    async def send(message):messages.append(message)
    scope={'type':'http','asgi':{'version':'3.0','spec_version':'2.4'},'http_version':'1.1','method':'POST','scheme':'http','path':'/api/chat','raw_path':b'/api/chat','query_string':b'','headers':[(b'content-type',b'application/json')],'server':('isolated',1),'client':('test',1),'root_path':''}
    await api.app(scope,receive,send)
    return next(m['status']for m in messages if m['type']=='http.response.start'),json.loads(b''.join(m.get('body',b'')for m in messages if m['type']=='http.response.body'))
def call(question='전류',context=None,local_only=True):return asyncio.run(post({'question':question,'context':context or{},'local_only':local_only}))
def main():
    before=(SOURCE.stat().st_size,SOURCE.stat().st_mtime_ns);index_sha=hashlib.sha256((APP/'data/source_index.json').read_bytes()).hexdigest()
    source=SourceStore();engine=types.SimpleNamespace(cursor=pd.Timestamp('2024-12-01 00:00:00'));forecast_calls=[]
    def forecast(module,horizon,origin):forecast_calls.append((module,horizon,iso(origin)));return {'metrics':{'mae':1.25},'quality_note':'fake evaluation only'}
    reporting=services.Reporting(source,engine,forecast);api.REPORTING=reporting
    ctx={'module_id':'17','scope':'diagnosis','grain':'raw','source_kind':'recorded','cursor_time':'2024-12-01 00:00:00','start':'2025-03-13 19:45:05','end':'2025-03-13 19:45:20'}
    try:
        code,detail=call(context=ctx)
        direct=source.window(ctx['start'],ctx['end'],['17'])
        check('diagnosis_exact5second_interval_not_hour_floored',code==200 and detail['row_count']==3 and [r['timestamp']for r in detail['rows']]==[r['time']for r in direct])
        check('raw_values_match_immutable_source',all(abs(r['value']-d['currentR'])<1e-12 for r,d in zip(detail['rows'],direct)))
        check('explicit_archive_can_be_after_global_cursor',detail['query_scope']['start']==ctx['start']and detail['query_scope']['end_exclusive']==ctx['end']and engine.cursor==pd.Timestamp('2024-12-01'))
        check('raw_counts_units_and_recorded_scope',detail['source_observation_count']==3 and detail['unit']=='A'and detail['record_unit']=='source_row'and detail['query_scope']['source_kind']=='recorded')
        all_ctx={**ctx,'module_id':'all'};code,all_rows=call('전력',all_ctx)
        expected=source.window(ctx['start'],ctx['end'])
        first=sum(r['activePower']for r in expected if r['time']==ctx['start'])
        check('all_module_raw_sum_and_actual_source_row_count',code==200 and all_rows['row_count']==3 and all_rows['source_observation_count']==39 and all_rows['rows'][0]['records']==13 and abs(first-all_rows['rows'][0]['value'])<1e-9)
        code,energy=call('에너지',ctx)
        check('raw_energy_is5second_integral_not_hour_energy',code==200 and energy['query']['sensor']=='energy_kwh'and abs(energy['statistics']['sum_energy_kwh']-sum(r['activePower']*5/3600 for r in direct))<1e-12 and '5초'in energy['statistics']['aggregation'])
        terminal={**ctx,'start':'2025-04-30 00:00:00','end':'2025-04-30 00:00:05'};code,last=call(context=terminal)
        check('last_original_single_endpoint_is_available',code==200 and last['row_count']==1 and last['source_observation_count']==1)
        code,empty=call(context={**ctx,'start':'2025-03-13 19:45:01','end':'2025-03-13 19:45:04'})
        check('no_grid_points_returns_empty_without_fake_zero',code==200 and empty['rows']==[]and empty['source_observation_count']==0 and empty['statistics']['mean_display_value']is None)
        month_ctx={'module_id':'all','scope':'archive','grain':'month','source_kind':'recorded','start':'2024-12-01 00:00:00','end':'2025-04-30 00:00:00'}
        code,month=call('에너지',month_ctx)
        check('whole_retained_period_aggregates_without30day_truncation',code==200 and month['row_count']==5 and sum(r['records']for r in month['rows'])==3600 and month['source_observation_count']==33696000)
        check('partial_April_and_aggregation_basis_preserved',month['rows'][-1]['records']==696 and not month['rows'][-1]['complete']and month['statistics']['partial_period_count']==1 and month['record_unit']=='completed_hour')
        check('monthly_energy_conserves_completed_hour_sum',abs(month['statistics']['sum_energy_kwh']-float(source.hourly.energy_kwh.sum()))<1e-7)
        hour_ctx={**month_ctx,'module_id':'17','grain':'hour','start':'2025-01-01 00:30:00','end':'2025-01-01 02:30:00'};code,hour=call('전력',hour_ctx)
        check('incomplete_hour_boundaries_excluded_not_filled',code==200 and hour['row_count']==1 and hour['rows'][0]['timestamp']=='2025-01-01 01:00:00'and hour['source_observation_count']==720 and hour['statistics']['observed_start']=='2025-01-01 01:00:00'and hour['statistics']['observed_end_exclusive']=='2025-01-01 02:00:00')
        code,week=call('전력',{**month_ctx,'grain':'week'})
        check('weekly_end_label_does_not_extend_query_coverage',code==200 and week['rows'][-1]['timestamp']=='2025-05-04 00:00:00'and week['query_scope']['end_exclusive']=='2025-04-30 00:00:00'and week['rows'][-1]['records']==48 and not week['rows'][-1]['complete'])
        db.rows=[{'id':str(i),'module_id':mid,'time':t,'source_kind':kind}for i,(mid,t,kind)in enumerate([('17','2025-03-13 19:45:05','recorded'),('17','2025-03-13 19:45:10','recorded'),('17','2025-03-13 19:45:15','recorded'),('15','2025-03-13 19:45:10','recorded'),('17','2025-03-13 19:45:10','synthetic')])]
        code,alarms=call('이상 후보',{**ctx,'end':'2025-03-13 19:45:15'})
        check('stored_alarm_module_source_and_exclusive_end_applied',code==200 and alarms['row_count']==2 and alarms['statistics']['matched_count']==2 and db.queries[-1]=={'module':'17','start':ctx['start'],'end':'2025-03-13 19:45:15','source_kind':'recorded'})
        check('alarm_count_not_misreported_as_sensor_observation_count',alarms['source_observation_count']is None and alarms['unit']=='후보 건'and alarms['record_unit']=='stored_alarm')
        code,metrics=call('예측 모델 정확도',ctx)
        check('explicit_short_window_does_not_run_forecast_model',code==200 and metrics['rows']==[]and not forecast_calls and '새로 계산하지 않습니다'in metrics['answer'])
        code,modules=call('설비 목록',ctx)
        check('explicit_module_list_respects_selected_identity',code==200 and len(modules['rows'])==1 and modules['rows'][0]['id']=='17')
        # Default replay remains past seven completed days, as before.
        engine.cursor=pd.Timestamp('2025-03-13 19:45:20');replay={'module_id':'17','cursor_time':iso(engine.cursor)}
        code,default=call('전력',replay)
        check('default_replay_seven_day_daily_behavior_preserved',code==200 and default['query_scope']=={'module_id':'17','start':'2025-03-06 19:00:00','end_exclusive':'2025-03-13 19:00:00','scope':'replay','grain':'day','source_kind':'recorded'}and default['source_observation_count']==168*720)
        code,_=call('모델 예측',replay)
        check('default_replay_forecast_metrics_behavior_preserved',code==200 and forecast_calls==[('17',24,'2025-03-13 19:00:00')])
        captured=[]
        def fake_gpt(system,payload,schema,name):captured.append(payload);return {'kind':'sensor_summary','module_id':'17','sensor':'activePower','days':3}
        with patch.object(services,'gpt_json',fake_gpt):
            code,planned=call('전력',{**replay,'page':'PRIVATE_PAGE','start':'PRIVATE_START','end':'PRIVATE_END','source_kind':'recorded','scope':'replay','grain':'day','unapproved':'SECRET_CONTEXT'},False)
            check('GPT_receives_only_previously_approved_question_module_cursor',code==200 and captured==[{'question':'전력','context':{'module_id':'17','cursor_time':'2025-03-13 19:45:20'}}])
            check('GPT_day_plan_applied_locally',planned['query_scope']['start']=='2025-03-10 19:00:00')
            reads=len(key_reads);before_calls=len(captured);code,_=call('전류',ctx,False)
            check('explicit_scopes_always_local_no_key_or_GPT_call',code==200 and len(captured)==before_calls and len(key_reads)==reads)
        invalid=[('unknown_scope',{**ctx,'scope':'admin'}),('unknown_module',{**ctx,'module_id':'999'}),('unknown_grain',{**ctx,'grain':'year'}),('diagnosis_nonraw',{**ctx,'grain':'hour'}),('synthetic',{**ctx,'source_kind':'synthetic'}),('reversed',{**ctx,'start':ctx['end'],'end':ctx['start']}),('equal',{**ctx,'end':ctx['start']}),('before_source',{**ctx,'start':'2024-11-30 23:59:55'}),('after_source',{**ctx,'end':'2025-05-01'}),('missing_start',{k:v for k,v in ctx.items()if k!='start'}),('missing_end',{k:v for k,v in ctx.items()if k!='end'}),('invalid_time',{**ctx,'start':'not-a-time'}),('epoch_number',{**ctx,'start':12345}),('timezone',{**ctx,'start':'2025-03-13T19:45:05Z'}),('fractional_seconds',{**ctx,'end':'2025-03-13 19:45:20.1'}),('raw_over24h',{**ctx,'start':'2025-03-12 19:45:05','end':'2025-03-13 19:45:10'}),('aggregate_final_raw_endpoint',{**month_ctx,'start':'2025-04-30','end':'2025-04-30 00:00:05'}),('future_replay_cursor',{**replay,'cursor_time':'2025-04-01'}),('invalid_cursor',{**replay,'cursor_time':False})]
        for name,context in invalid:check('400_'+name,call(context=context)[0]==400)
        code,_=asyncio.run(post({'question':'전력','context':['not','object'],'local_only':True}));check('400_context_not_object',code==400)
        # Test the untrusted GPT plan without any HTTP transport or API-key verification.
        for name,plan in [('unknown_module',{'kind':'sensor_summary','module_id':'999','sensor':'activePower','days':7}),('unallowed_sensor',{'kind':'sensor_summary','module_id':'17','sensor':'secret','days':7}),('noninteger_days',{'kind':'sensor_summary','module_id':'17','sensor':'activePower','days':True})]:
            with patch.object(services,'gpt_json',return_value=plan):check('400_GPT_'+name,call('전력',replay,False)[0]==400)
        check('production_source_and_index_preserved',before==(SOURCE.stat().st_size,SOURCE.stat().st_mtime_ns)and index_sha==hashlib.sha256((APP/'data/source_index.json').read_bytes()).hexdigest())
        check('read_only_engine_and_no_DB_write',engine.cursor==pd.Timestamp('2025-03-13 19:45:20')and db.writes==0 and api.ENGINE is None)
        return {'example_query_scope':detail['query_scope'],'monthly_statistics':month['statistics'],'external_payload_fields':['question','context.module_id','context.cursor_time'],'production_DB_used':False,'model_or_report_generation':False,'source_index_sha256':index_sha}
    finally:source.mm.close();source.file.close()

if __name__=='__main__':
    status='passed';extra={}
    try:
        with patch('urllib.request.urlopen',side_effect=AssertionError('External HTTP forbidden')):extra=main()
    except Exception as exc:
        import traceback;traceback.print_exc();status='failed';checks.append({'exception_type':type(exc).__name__,'message':str(exc)})
    path=OUT/'chat_scope_verification.json';path.write_text(json.dumps({'status':status,'checks':checks,'production_HTTP_or_server_restart':False,'external_transport_calls':0,**extra},ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'status':status,'checks':len(checks),'path':str(path)}));gc.collect();temporary.cleanup();sys.exit(0 if status=='passed'else 1)
