"""Offline focused checks: isolated SQLite, fake source/detector, no live POST.

Does not start a server, run application lifespan, load model weights, or send
external notifications. The application's real ASGI GET detail route is used.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import asyncio,gc,hashlib,json,queue,tempfile,threading,types,uuid
from contextlib import contextmanager
from collections import OrderedDict
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
APP=OUT.parent
temporary=tempfile.TemporaryDirectory(prefix='new_detections_',dir=OUT)
DATA=Path(temporary.name).resolve()
assert DATA.parent==OUT.resolve()and DATA.name.startswith('new_detections_')
config=types.ModuleType('config')
config.APP=APP;config.ROOT=APP.parent;config.DATA=DATA;config.MODEL='unused'
config.CONFIG=types.SimpleNamespace(public=lambda:{},slack_url=lambda:None)
sys.modules['config']=config
sys.path.insert(0,str(APP/'backend'))
import store
import engine
import app as api

# Close test connections deterministically before Windows temporary cleanup.
original_connect=store.Database.connect
@contextmanager
def closing_connect(self):
    connection=original_connect(self)
    try:
        with connection:yield connection
    finally:connection.close()
store.Database.connect=closing_connect

checks=[]
def check(name,value,details=None):
    checks.append({'name':name,'passed':bool(value),'details':details})
    assert value,name
def iso(t):return pd.Timestamp(t).strftime('%Y-%m-%d %H:%M:%S')
def count(table):
    with store.DB.connect()as c:return c.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]

class Source:
    bounds={m:{'first':'20241201000000','last':'20250430000000'}for m in ['17','15']}
    def history_energy(self,module,cursor):return pd.Series([0.],index=[pd.Timestamp(cursor).floor('D')])
    def window(self,start,end,module_ids=None):
        out=[]
        for t in pd.date_range(pd.Timestamp(start),pd.Timestamp(end),freq='5s',inclusive='left'):
            for mid in module_ids or ['17','15']:
                row={'module_id':str(mid),'module(equipment)':str(mid)+'(fixture)','equipment':str(mid)+'(fixture)',
                     'time':iso(t),'localtime':int(t.strftime('%Y%m%d%H%M%S')),'timestamp':0,
                     'operation':1,'accumActiveEnergy':1,'activePower':100.,'reactivePowerLagging':15.,'energy_kwh':100*5/3600}
                for phase in 'RST':row['voltage'+phase]=220.;row['current'+phase]=10.;row['powerFactor'+phase]=95.
                for phase in ['RS','ST','TR']:row['voltage'+phase]=380.
                out.append(row)
        return out

class Detector:
    def __init__(self):self.program={}
    def diagnose_batch(self,rows):
        result=[]
        for row in rows:
            kinds=self.program.get((row['time'],row['module_id']),[])
            result.append({'module_id':row['module_id'],'time':row['time'],'status':'unconfirmed_candidate'if kinds else'no_candidate',
               'rules':[{'type':k,'score':2.,'threshold':1.,'evidence':{'fixture':True}}for k in kinds if k not in['AE','IF']],
               'ai':{'ae_alarm':'AE'in kinds,'iforest_alarm':'IF'in kinds,'ae_score':2. if'AE'in kinds else.2,'ae_threshold':1.,'iforest_score':2. if'IF'in kinds else.2,'iforest_threshold':1.}})
        return result
    def scores(self,frame,full_ai=True):
        n=len(frame)
        return {'rule_scores':np.zeros((n,6)),'rule_thresholds':np.ones((n,6)),'rule_flags':np.zeros((n,6),dtype=bool)}

def make_engine():
    e=engine.ReplayEngine.__new__(engine.ReplayEngine)
    e.source=Source();e.lock=threading.RLock();e.playing=False;e.speed=20;e.session=uuid.uuid4().hex;e.sequence=0
    e.listeners=[queue.Queue()];e.history=[];e.snapshot={};e.detector=Detector();e.detector_error=None;e.stopped=True
    e.detection_occurrences=OrderedDict();e.scenarios=[{'id':'fixture','name':'fixture','start':'2025-03-13 19:43:00','end':'2025-03-13 20:00:00','source_kind':'recorded'}]
    e.set_scenario('fixture');return e
def drain(e):
    out=[]
    while not e.listeners[0].empty():out.append(e.listeners[0].get_nowait())
    return out
def tick(e,t,patterns=None,log=True):
    e.cursor=pd.Timestamp(t)
    for mid,kinds in(patterns or{}).items():e.detector.program[(iso(t),mid)]=kinds
    e.receive(log=log)
    return drain(e)[-1]
async def get_detail(event_id):
    messages=[];first=True;forever=asyncio.Event()
    async def receive():
        nonlocal first
        if first:first=False;return {'type':'http.request','body':b'','more_body':False}
        await forever.wait()
    async def send(message):messages.append(message)
    path='/api/diagnosis/'+event_id
    scope={'type':'http','asgi':{'version':'3.0','spec_version':'2.4'},'http_version':'1.1','method':'GET','scheme':'http','path':path,'raw_path':path.encode(),'query_string':b'','headers':[],'server':('127.0.0.1',8099),'client':('127.0.0.1',1),'root_path':''}
    await api.app(scope,receive,send)
    status=next(m['status']for m in messages if m['type']=='http.response.start')
    body=b''.join(m.get('body',b'')for m in messages if m['type']=='http.response.body')
    return status,json.loads(body)

def main():
    prior={'scenario_id':'old','module_id':'17','equipment':'old','time':'2025-01-01 00:00:10','type':'power_factor_drop','title':'old','source_kind':'recorded'}
    store.DB.alarm(prior)
    e=make_engine();api.ENGINE=e;api.SOURCE=e.source
    initial=drain(e)[-1]
    check('preview_has_no_new_events_despite_old_alarms',not initial['new_detections']and bool(initial['snapshot']['alarms']))
    p=tick(e,'2025-03-13 19:44:00')
    check('normal_telemetry_does_not_republish_DB_history',p['new_detections']==[])
    times=pd.date_range('2025-03-13 19:45:05',periods=6,freq='5s')
    packets=[tick(e,t,{'17':['power_factor_drop']})for t in times]
    check('first_two_positive_points_are_not_new_detection',not packets[0]['new_detections']and not packets[1]['new_detections'])
    first=packets[2]['new_detections'][0]
    check('third_point_emits_once_with_onset_and_detected_time',first['first_time']=='2025-03-13 19:45:05'and first['detected_at']=='2025-03-13 19:45:15')
    check('sustained_fault_does_not_repeat_every_tick',sum(len(p['new_detections'])for p in packets)==1)
    check('event_has_resolvable_persistent_and_occurrence_IDs',first['id']==first['event_id']and first['detail_event_id']==first['occurrence_id']and store.DB.alarm_by_id(first['event_id'])is not None)
    check('no_new_events_in_snapshot',all('new_detections'not in p['snapshot']for p in packets))
    key=f"sustained-v1:recorded:recorded:17:power_factor_drop:{first['detected_at'][:15]}"
    check('legacy_persistent_ID_hash_unchanged',first['id']==hashlib.sha256(key.encode()).hexdigest()[:20])
    saved_count=count('alarms');outbox=count('outbox');store.DB.ack(first['id'])
    tick(e,'2025-03-13 19:46:00')
    second_packets=[tick(e,t,{'17':['power_factor_drop']})for t in pd.date_range('2025-03-13 19:46:05',periods=3,freq='5s')]
    second=second_packets[-1]['new_detections'][0]
    check('different_onset_same_bucket_emits_current_occurrence',second['id']==first['id']and second['occurrence_id']!=first['occurrence_id']and not second['new_persistent_alarm'])
    check('existing_ack_and_outbox_are_not_reset',count('alarms')==saved_count and count('outbox')==outbox and store.DB.alarm_by_id(first['id'])['status']=='acknowledged')
    check('current_detected_time_differs_from_stored_time',second['detected_at']=='2025-03-13 19:46:15'and second['stored_detected_at']=='2025-03-13 19:45:15')
    code,detail=asyncio.run(get_detail(second['detail_event_id']))
    check('ASGI_GET_occurrence_detail_returns_current_interval',code==200 and detail['event']['start']=='2025-03-13 19:46:05'and detail['event']['time']=='2025-03-13 19:46:15'and detail['series'][0]['time']=='2025-03-13 19:43:15')
    code,canonical=asyncio.run(get_detail(second['event_id']))
    check('ASGI_GET_persistent_ID_still_returns_legacy_record',code==200 and canonical['event']['time']=='2025-03-13 19:45:15')
    old_session=e.session;e.set_scenario('fixture');drain(e)
    replay=[tick(e,t,{'17':['power_factor_drop']})for t in times[:3]][-1]['new_detections'][0]
    check('new_session_replay_notifies_even_when_DB_insert_is_duplicate',e.session!=old_session and replay['id']==first['id']and replay['occurrence_id']!=first['occurrence_id']and not replay['new_persistent_alarm'])
    check('same_DB_replay_does_not_add_Slack_job',count('outbox')==outbox)
    e.control('seek',timestamp=iso(times[0]));drain(e)
    repeated=[tick(e,t,{'17':['power_factor_drop']})for t in times[:3]]
    check('same_session_same_onset_revisit_is_suppressed',not any(p['new_detections']for p in repeated))
    # A source-time gap must restart consecutive-point counting.
    points=['2025-03-13 19:47:00','2025-03-13 19:47:10','2025-03-13 19:47:15','2025-03-13 19:47:20']
    gap=[tick(e,t,{'15':['load_imbalance']})for t in points]
    check('missing_5second_point_restarts_streak',not gap[2]['new_detections']and gap[3]['new_detections'][0]['first_time']==points[1])
    ae=[tick(e,t,{'17':['AE'],'15':['reactive_power_surge']})for t in pd.date_range('2025-03-13 19:48:00',periods=4,freq='5s')]
    events=ae[2]['new_detections']
    check('AI_only_and_other_module_rule_notify_independently',len(events)==2 and{(x['module_id'],x['type'])for x in events}=={('17','abnormal_pattern'),('15','reactive_power_surge')})
    check('AI_sustained_frame_is_not_repeated',not ae[3]['new_detections'])
    before=count('telemetry');preview=tick(e,'2025-03-13 19:49:00',{'17':['AE']},log=False)
    check('seek_preview_does_not_log_or_notify',not preview['new_detections']and count('telemetry')==before)
    e.broadcast('state',[first]);state=drain(e)[-1]
    check('state_packet_explicitly_has_empty_new_events',state['new_detections']==[])
    async def connected():
        response=await api.stream(types.SimpleNamespace());value=await anext(response.body_iterator);await response.body_iterator.aclose();return json.loads(value[6:])
    check('SSE_reconnect_does_not_replay_last_alert',asyncio.run(connected())['new_detections']==[])
    # The occurrence lookup is deliberately bounded, while DB IDs remain durable.
    for i in range(260):
        t=pd.Timestamp('2025-03-14 00:00:00')+pd.Timedelta(seconds=i*20);k=('17','night_operation')
        e.streaks[k]=3;e.streak_first[k]=iso(t)
        event={'scenario_id':'fixture','module_id':'17','equipment':'17(fixture)','time':iso(t+pd.Timedelta(seconds=10)),'type':'night_operation','title':'fixture','source_kind':'recorded'}
        e.persist_detection(event,k,'fixture',[])
    check('occurrence_lookup_bounded_to256',len(e.detection_occurrences)==256)
    check('persistent_ID_lookup_survives_occurrence_eviction',store.DB.alarm_by_id(first['id'])is not None)
    code,unknown=asyncio.run(get_detail('unknown_event'))
    check('unknown_detail_still_404',code==404)
    return {'example_notification':second,'last_checked_occurrences':len(e.detection_occurrences)}

if __name__=='__main__':
    status='passed';extra={}
    try:extra=main()
    except Exception as exc:
        status='failed';checks.append({'exception_type':type(exc).__name__,'message':str(exc)})
        import traceback;traceback.print_exc()
    result={'status':status,'checks':checks,'external_GPT_or_Slack_calls':0,'production_server_restart_or_POST':False,'production_DB_used':False,**extra}
    (OUT/'new_detections_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':status,'checks':len(checks),'path':str(OUT/'new_detections_verification.json')}))
    # Resolved and checked above; this removes only this script's fresh test DB.
    gc.collect();temporary.cleanup()
    sys.exit(0 if status=='passed'else 1)
