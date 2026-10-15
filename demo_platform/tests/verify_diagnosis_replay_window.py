"""Readonly diagnosis-window checks using actual indexed source/model scores.

Production server/DB are never used. Existing detector regression25 runs first
on a separate SQLite database; no player threads or external services run.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import asyncio,gc,hashlib,json,math,queue,shutil
from urllib.parse import urlencode
from unittest.mock import patch
import pandas as pd
import verify_new_detections as base
from data_source import SourceStore,SOURCE

OUT=Path(__file__).resolve().parent;APP=OUT.parent;api=base.api;checks=[]
def check(name,value,details=None):
    checks.append({'name':name,'passed':bool(value),'details':details});assert value,name
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()
def db_digest():
    with base.store.DB.connect()as c:return digest({table:[dict(r)for r in c.execute('SELECT * FROM '+table+' ORDER BY rowid')]for table in ['telemetry','alarms','outbox','settings','reports']})
async def get(path,**query):
    messages=[];sent=False;forever=asyncio.Event()
    async def receive():
        nonlocal sent
        if not sent:sent=True;return {'type':'http.request','body':b'','more_body':False}
        await forever.wait()
    async def send(message):messages.append(message)
    scope={'type':'http','asgi':{'version':'3.0','spec_version':'2.4'},'http_version':'1.1','method':'GET','scheme':'http','path':path,'raw_path':path.encode(),'query_string':urlencode(query).encode(),'headers':[],'server':('127.0.0.1',8099),'client':('127.0.0.1',1),'root_path':''}
    await api.app(scope,receive,send)
    code=next(m['status']for m in messages if m['type']=='http.response.start')
    body=b''.join(m.get('body',b'')for m in messages if m['type']=='http.response.body')
    return code,json.loads(body)
def call(eid,**query):return asyncio.run(get('/api/diagnosis/'+eid+'/replay-window',**query))
class NoThread:
    def __init__(self,*a,**k):pass
    def start(self):pass

def main():
    base.main();check('existing_notification25_regression',len(base.checks)==25 and all(c.get('passed')for c in base.checks))
    shutil.copyfile(APP/'data/source_index.json',base.DATA/'source_index.json')
    before_source=(SOURCE.stat().st_size,SOURCE.stat().st_mtime_ns)
    model_paths=[APP/'analytics/diagnosis'/name for name in ['models.joblib','ae.pt','synthetic_xgb.json','baseline.json','thresholds.json']]
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in model_paths}
    source=SourceStore();api.SOURCE=source
    try:
        with patch.object(base.engine.threading,'Thread',NoThread),patch.object(base.engine.ReplayEngine,'detector_ready',lambda self:False):e=base.engine.ReplayEngine(source)
        e.listeners=[queue.Queue()];api.ENGINE=e
        def forbidden():raise AssertionError('Readonly GET must not call ENGINE.detector_ready')
        e.detector_ready=forbidden
        invariant=lambda:digest({'state':e.state(),'snapshot':e.snapshot,'history':e.history,'frames':e.frames,'occurrences':e.detection_occurrences})
        frozen_state=invariant();frozen_db=db_digest()
        ids=['abnormal_pattern_0091','abnormal_pattern_0092','abnormal_pattern_0093','power_factor_drop_0061']
        example=None
        for eid in ids:
            code,data=call(eid);w=data['window'];example=w
            check('original_default_inclusive27_'+eid,code==200 and w['point_count']==27 and len(data['series'])==27 and w['actual_before_seconds']==60 and w['actual_after_seconds']==60 and not w['clipped'])
            check('all_points_have6rules_AE_IF_'+eid,len(data['algorithm'])==27 and all(len(a['rules'])==6 for a in data['algorithm'])and len(data['ai'])==27 and all(math.isfinite(a['ae_score'])and math.isfinite(a['iforest_score'])for a in data['ai']))
        check('private_scorer_did_not_initialize_ENGINE',e.detector is None and api.DIAGNOSIS_READER is not None)
        code,zero=call(ids[0],before_seconds=0,after_seconds=0)
        check('zero_padding_retains_complete_fault_interval',code==200 and zero['window']['point_count']==3 and zero['window']['start']==zero['window']['event_start']and zero['window']['end_inclusive']==zero['window']['event_end'])
        code,asymmetric=call(ids[0],before_seconds=0,after_seconds=5)
        check('asymmetric_padding_has_correct_inclusive_count',code==200 and asymmetric['window']['point_count']==4 and asymmetric['window']['actual_after_seconds']==5)
        code,maximum=call(ids[0],before_seconds=1800,after_seconds=1800)
        check('maximum1800each_has_all723_points',code==200 and maximum['window']['point_count']==723 and len(maximum['algorithm'])==len(maximum['ai'])==723)
        code,legacy=asyncio.run(get('/api/diagnosis/'+ids[0]))
        check('legacy_detail_still72points_exclusive_3min',code==200 and len(legacy['series'])==72 and 'window'not in legacy)
        # Temporary occurrence records emulate current notifications without DB writes.
        boundary_events=[('occ_start_fixture','2024-12-01 00:00:00',0,60),('occ_end_fixture','2025-04-30 00:00:00',60,0)]
        for eid,t,before,after in boundary_events:
            e.detection_occurrences[eid]={'id':'saved_boundary_id','detail_event_id':eid,'module_id':'17','equipment':'17(6호기)','start':t,'end':t,'time':t,'source_kind':'recorded'}
            code,data=call(eid);w=data['window']
            check('source_bounds_clip_'+eid,code==200 and w['clipped']and w['point_count']==13 and w['actual_before_seconds']==before and w['actual_after_seconds']==after)
            check('end_exclusive_is_next5seconds_'+eid,(pd.Timestamp(w['end_exclusive'])-pd.Timestamp(w['end_inclusive'])).total_seconds()==5 and data['series'][-1]['time']==w['end_inclusive'])
        eid='occ_current_fixture';e.detection_occurrences[eid]={'id':'old_saved_id','detail_event_id':eid,'module_id':'17','equipment':'17(6호기)','first_time':'2025-03-13 19:46:05','time':'2025-03-13 19:46:15','source_kind':'recorded'}
        code,current=call(eid,before_seconds=0,after_seconds=0)
        check('occurrence_uses_current_first_time_not_old_saved_ID',code==200 and current['window']['event_start']=='2025-03-13 19:46:05'and current['window']['event_end']=='2025-03-13 19:46:15'and current['window']['point_count']==3)
        parent={'id':'synthetic_rebased_fixture','start':'2025-03-14 13:01:40','end':'2025-03-14 13:11:40','source_kind':'synthetic','module_id':'17','fault_type':'power_factor_drop','augmentation_origin_start':'2025-03-14 13:00:00'}
        e.scenarios.append(parent);eid='occ_synthetic_fixture'
        e.detection_occurrences[eid]={'id':'saved_synthetic_id','detail_event_id':eid,'scenario_id':parent['id'],'module_id':'17','equipment':'17(6호기)','start':'2025-03-14 13:02:10','end':'2025-03-14 13:02:20','time':'2025-03-14 13:02:20','source_kind':'synthetic','type':'abnormal_pattern'}
        raw=source.window('2025-03-14 13:01:10','2025-03-14 13:03:25',['17']);raw_by_time={r['time']:r for r in raw};raw_copy=digest(raw)
        code,synth=call(eid)
        check('synthetic_keeps_original_cause_and_anchor',code==200 and synth['source_kind']=='synthetic'and all(abs(r['powerFactorR']-raw_by_time[r['time']]['powerFactorR']*(.55 if r['time']>='2025-03-14 13:02:00'else 1.))<1e-9 for r in synth['series']))
        check('synthetic_does_not_mutate_raw_sensor_values',raw_copy==digest(raw)and all(r['currentR']==raw_by_time[r['time']]['currentR']for r in synth['series']))
        for name,value in [('negative','-5'),('over1800','1805'),('nonmultiple','7'),('fraction','1.5'),('decimal_float','5.0'),('boolean','true')]:
            check('invalid_padding400_'+name,call(ids[0],before_seconds=value)[0]==400)
        check('unknown_event404',call('does_not_exist')[0]==404)
        # These fixtures are set up by the test; requests must not change them.
        frozen_state=invariant()
        call(ids[0]);call('occ_synthetic_fixture');call('occ_current_fixture')
        check('ENGINE_all_public_and_buffer_state_unchanged',invariant()==frozen_state)
        check('no_DB_logs_alarms_outbox_writes',db_digest()==frozen_db)
        check('no_SSE_packet_or_alert_emitted',e.listeners[0].empty())
        check('model_and_threshold_hashes_unchanged',hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in model_paths})
        check('source_size_mtime_preserved',before_source==(SOURCE.stat().st_size,SOURCE.stat().st_mtime_ns))
        return {'example_window':example,'model_sha256':hashes,'notification_regression_checks':base.checks,'range_contract':'both observation endpoints included; existing detail still exclusive upper3min'}
    finally:source.mm.close();source.file.close()

if __name__=='__main__':
    status='passed';extra={}
    try:extra=main()
    except Exception as exc:
        import traceback;traceback.print_exc();status='failed';checks.append({'exception_type':type(exc).__name__,'message':str(exc)})
    dest=OUT/'diagnosis_replay_window_verification.json'
    dest.write_text(json.dumps({'status':status,'checks':checks,'production_server_restart_or_POST':False,'production_DB_used':False,'external_transport_calls':0,**extra},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':status,'checks':len(checks),'path':str(dest)}))
    gc.collect();base.temporary.cleanup();sys.exit(0 if status=='passed'else 1)
