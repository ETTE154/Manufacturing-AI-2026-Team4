"""Regression checks for cross-window replay, anchored augmentation and5-minute history."""
from pathlib import Path
import json,urllib.request,urllib.error,urllib.parse,time,datetime,sys
BASE='http://127.0.0.1:8099/api';OUT=Path(__file__).resolve().parent;checks=[]
def call(path,body=None):
    request=urllib.request.Request(BASE+path,data=json.dumps(body).encode()if body is not None else None,headers={'Content-Type':'application/json'},method='POST'if body is not None else'GET')
    with urllib.request.urlopen(request,timeout=25)as response:return json.load(response)
def control(action,**kwargs):return call('/replay',{'action':action,**kwargs})
def check(name,value,details=None):checks.append({'name':name,'passed':bool(value),'details':details});assert value,name
def stamp(value):return datetime.datetime.fromisoformat(value)
def main():
    integrations=call('/integrations');check('test_has_no_external_slack_destination',not integrations['slack']['configured'])
    control('scenario',scenario_id='recorded_pf');control('speed',speed=1)
    candidates=call('/diagnosis')['candidates'];check('four_original_candidates_available',len(candidates)==4)
    for candidate in candidates:
        previous=call('/bootstrap')['replay'];response=control('jump_event',event_id=candidate['id']);control('pause')
        event=response['event'];replay=response['replay'];expected=stamp(candidate.get('start')or candidate.get('start_time')or candidate['time'])-datetime.timedelta(seconds=30)
        check('exact_30_second_lead_'+candidate['id'],stamp(replay['start'])==expected and event['actual_lead_in_seconds']==30)
        check('jump_autoplays_new_session_'+candidate['id'],replay['playing']and replay['session_id']!=previous['session_id']and len(response['snapshot']['points'])==13)
        check('highlight_matches_detected_interval_'+candidate['id'],replay['highlight']['event_id']==candidate['id']and stamp(replay['highlight']['start'])==stamp(event['detected_at'])and stamp(replay['highlight']['end'])>stamp(event['detected_at']))
    before=call('/bootstrap')['replay']
    try:control('jump_event',event_id='nonexistent_event')
    except urllib.error.HTTPError as e:check('unknown_event_404',e.code==404)
    else:check('unknown_event_404',False)
    check('invalid_jump_preserves_active_window',call('/bootstrap')['replay']['session_id']==before['session_id'])
    synthetic=next((a for a in call('/alarms')['rows']if a.get('source_kind')=='synthetic'and a.get('scenario_id')=='synthetic_power_factor_drop'),None)
    if synthetic is None:
        control('scenario',scenario_id='synthetic_power_factor_drop');control('seek',timestamp='2025-03-14 13:02:00');control('speed',speed=60);control('play');time.sleep(1.2);control('pause')
        synthetic=next(a for a in call('/alarms')['rows']if a.get('source_kind')=='synthetic'and a.get('scenario_id')=='synthetic_power_factor_drop')
    response=control('jump_event',event_id=synthetic['id']);control('pause');t=response['event']['detected_at'];frame=control('seek',timestamp=t)['snapshot'];control('pause')
    row=next(r for r in frame['points']if r['module_id']=='17');params=urllib.parse.urlencode({'module_id':'17','sensor':'powerFactorR','grain':'raw','start':row['time'],'end':str(stamp(row['time'])+datetime.timedelta(seconds=5))});actual=call('/series?'+params)['series'][0]['value']
    check('synthetic_jump_retains_cause_and_original_anchor',row.get('augmentation_type')=='power_factor_drop'and abs(row['powerFactorR']-actual*.55)<1e-8,{'source_time':row['time']})
    check('synthetic_highlight_keeps_provenance',response['replay']['highlight']['source_kind']=='synthetic')
    # Drain seventy distinct source timestamps from SSE to cross the sixty-interval boundary.
    control('scenario',scenario_id='recorded_normal');control('speed',speed=60)
    request=urllib.request.Request(BASE+'/stream');connection=urllib.request.urlopen(request,timeout=35);control('play');times=[];deadline=time.monotonic()+30
    try:
        while len(times)<72 and time.monotonic()<deadline:
            line=connection.readline().decode()
            if not line.startswith('data:'):continue
            event=json.loads(line[5:]);value=event.get('snapshot',{}).get('timestamp')
            if event.get('type')=='telemetry'and value and value not in times:times.append(value)
    finally:control('pause');connection.close()
    check('stream_crossed_sixty_intervals',len(times)>=72,{'observed_ticks':len(times)})
    history=call('/bootstrap')['history'];unique=sorted({r['time']for r in history})
    check('server_bootstrap_keeps_61_endpoints',len(unique)==61 and len(history)==13*61)
    check('visible_history_spans_exactly_300_seconds',(stamp(unique[-1])-stamp(unique[0])).total_seconds()==300)
    check('all_modules_share_same_five_second_grid',all(sum(r['time']==t for r in history)==13 for t in unique))
    control('scenario',scenario_id='recorded_pf');control('speed',speed=20)
if __name__=='__main__':
    status='passed'
    try:main()
    except Exception as error:status='failed';checks.append({'exception_type':type(error).__name__,'message':str(error)})
    (OUT/'replay_revision_verification.json').write_text(json.dumps({'status':status,'checks':checks,'external_gpt_or_slack_calls':0},ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({'status':status,'checks':len(checks)}));sys.exit(0 if status=='passed'else 1)
