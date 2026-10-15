"""Offline source-bound/range/chunk checks, reusing isolated detection tests.

Reads verified hourly summaries and a few indexed five-second windows only.
No production HTTP request, server restart, model fitting or external transport.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import asyncio,gc,json,shutil
from urllib.parse import urlencode
from unittest.mock import patch
import pandas as pd
import verify_new_detections as base
from data_source import SourceStore,SOURCE

OUT=Path(__file__).resolve().parent;APP=OUT.parent
checks=[]
def check(name,value,details=None):
    checks.append({'name':name,'passed':bool(value),'details':details});assert value,name
async def get(path,**query):
    messages=[];first=True;forever=asyncio.Event()
    async def receive():
        nonlocal first
        if first:first=False;return {'type':'http.request','body':b'','more_body':False}
        await forever.wait()
    async def send(message):messages.append(message)
    scope={'type':'http','asgi':{'version':'3.0','spec_version':'2.4'},'http_version':'1.1','method':'GET','scheme':'http','path':path,'raw_path':path.encode(),'query_string':urlencode(query).encode(),'headers':[],'server':('127.0.0.1',8099),'client':('127.0.0.1',1),'root_path':''}
    await base.api.app(scope,receive,send)
    status=next(m['status']for m in messages if m['type']=='http.response.start')
    body=b''.join(m.get('body',b'')for m in messages if m['type']=='http.response.body')
    return status,json.loads(body)
def call(path,**query):return asyncio.run(get(path,**query))
class NoThread:
    def __init__(self,*a,**k):pass
    def start(self):pass

def main():
    # First rerun all25 notification checks on these current backend sources.
    base.main();check('notification_regression25_passed',len(base.checks)==25 and all(c.get('passed')for c in base.checks))
    detection_engine=base.api.ENGINE
    check('session_count_not_limited_by256_occurrence_lookup',detection_engine.state()['session_detection_count']>256)
    count_before=detection_engine.state()['session_detection_count'];modules_before=detection_engine.state()['session_detected_module_ids']
    detection_engine.control('seek',timestamp='2025-03-13 19:43:00')
    s=detection_engine.state()
    check('seek_clears_current_active_but_keeps_session_totals',not s['active_detection_module_ids']and s['session_detection_count']==count_before and s['session_detected_module_ids']==modules_before)
    detection_engine.set_scenario('fixture');s=detection_engine.state()
    check('new_session_clears_detection_totals_and_active',s['session_detection_count']==0 and not s['session_detected_module_ids']and not s['active_detection_module_ids'])
    index=APP/'data/source_index.json';shutil.copyfile(index,base.DATA/'source_index.json')
    before=(SOURCE.stat().st_size,SOURCE.stat().st_mtime_ns)
    source=SourceStore();base.api.SOURCE=source
    try:
        a=source.series_availability['by_module'];all_a=a['all']
        check('availability_has_all13_module_scopes',len(a)==14 and set(a)=={'all'}|set(source.bounds))
        check('raw_bounds_are_dynamic_original_endpoints',all_a['raw']['first_timestamp']=='2024-12-01 00:00:00'and all_a['raw']['last_timestamp']=='2025-04-30 00:00:00')
        check('raw_terminal_day_has_only_one_point',all_a['raw']['max_date']=='2025-04-30'and all_a['raw']['partial_last_day']and all_a['raw']['last_day_samples']==1)
        check('completed_hour_availability_ends_Apr29',all_a['hour']['period_count']==3600 and all_a['hour']['max_date']=='2025-04-29'and all_a['hour']['end_exclusive']=='2025-04-30 00:00:00')
        check('day_week_month_counts_and_completeness',[(all_a[g]['period_count'],all_a[g]['complete_period_count'])for g in['day','week','month']]==[(150,150),(23,21),(5,4)])
        check('week_label_not_mistaken_for_available_calendar_date',all_a['week']['last_timestamp']=='2025-05-04 00:00:00'and all_a['week']['max_date']=='2025-04-29')
        with patch.object(base.engine.threading,'Thread',NoThread),patch.object(base.engine.ReplayEngine,'detector_ready',lambda self:False):
            e=base.engine.ReplayEngine(source);base.api.ENGINE=e
            check('default_replay_starts_at_original_first_point',e.scenario['id']=='recorded_full'and str(e.cursor)=='2024-12-01 00:00:00')
            check('full_replay_end_includes_last_original_point',str(e.end)=='2025-04-30 00:00:05')
            check('initial_buffer_is_only_five_minutes',len(e.frames)==60 and sum(map(len,e.frames.values()))==13*60)
            check('start_has_no_completed_hour',e.state()['completed_history_hours']==0)
            code,meta=call('/api/bootstrap')
            check('bootstrap_exposes_dynamic_bounds_and_grain_metadata',code==200 and meta['meta']['data_range']=={'start':'2024-12-01 00:00:00','end':'2025-04-30 00:00:00'}and len(meta['meta']['series_availability']['by_module'])==14)
            code,raw=call('/api/series',grain='raw')
            check('empty_date_raw_default_clamps_to_first_point',code==200 and len(raw['series'])==1 and raw['effective_start']=='2024-12-01 00:00:00')
            code,hour=call('/api/series',grain='hour')
            check('empty_date_hour_default_is_valid_empty_prefix',code==200 and hour['series']==[])
            with patch.object(base.api,'load_module',side_effect=AssertionError('Empty history must not load a predictor')):
                code,forecast=call('/api/forecast',module_id='all',horizon=24)
            check('first_hour_forecast_waits_without_fake_zero_prediction',code==200 and forecast['status']=='awaiting_completed_hour'and forecast['history_count']==0 and forecast['forecast']==[]and forecast['history']==[]and forecast['uses_only_origin_history'])
            code,last=call('/api/series',module_id='17',grain='raw',start='2025-04-30 00:00:00',end='2025-05-01 00:00:00',range_mode='calendar')
            check('calendar_terminal_Apr30_returns_exactly_one_raw_point',code==200 and len(last['series'])==1 and last['series'][0]['timestamp']=='2025-04-30 00:00:00'and last['effective_end_exclusive']=='2025-04-30 00:00:05')
            invalid=[('timestamp_after_source',{'grain':'raw','start':'2025-04-30','end':'2025-05-01'}),('before_source',{'grain':'hour','start':'2024-11-30','end':'2024-12-02'}),('reversed',{'grain':'hour','start':'2025-01-02','end':'2025-01-01'}),('equal',{'grain':'hour','start':'2025-01-01','end':'2025-01-01'}),('raw_two_calendar_days',{'grain':'raw','start':'2025-04-29','end':'2025-05-01','range_mode':'calendar'}),('aggregated_Apr30_unavailable',{'grain':'hour','start':'2025-04-30','end':'2025-05-01','range_mode':'calendar'}),('unknown_module',{'module_id':'999','grain':'hour'}),('unknown_grain',{'grain':'year'})]
            for name,params in invalid:check('400_'+name,call('/api/series',**params)[0]==400)
            code,month=call('/api/series',sensor='energy_kwh',grain='month',start='2024-12-01',end='2025-04-30',range_mode='calendar')
            check('monthly_partial_record_flags_preserved',code==200 and len(month['series'])==5 and month['series'][-1]['records']==696 and not month['series'][-1]['complete'])
            code,week=call('/api/series',sensor='energy_kwh',grain='week',start='2024-12-01',end='2025-04-30',range_mode='calendar')
            check('weekly_labels_and_partial_flags_preserved',code==200 and len(week['series'])==23 and week['series'][-1]['timestamp']=='2025-05-04 00:00:00'and week['series'][-1]['records']==48 and not week['series'][-1]['complete'])
            for i in range(66):
                e.cursor=pd.Timestamp('2024-12-01')+pd.Timedelta(seconds=5*i);e.receive(log=False)
            times=sorted({r['time']for r in e.history})
            check('buffer_rolls_across_five_minutes_without_whole_source_load',e.frame_start==pd.Timestamp('2024-12-01 00:05:00')and len(e.frames)==60 and sum(map(len,e.frames.values()))==780)
            check('history_remains61_endpoints_300seconds',len(times)==61 and len(e.history)==13*61 and(pd.Timestamp(times[-1])-pd.Timestamp(times[0])).total_seconds()==300)
            e.control('seek',timestamp='2024-12-01 01:00:00')
            check('one_completed_hour_after_first_hour',e.state()['completed_history_hours']==1)
            e.control('seek',timestamp='2025-04-30 00:00:00')
            check('terminal_raw_point_loaded_with_bounded_final_chunk',len(e.frames)==1 and len(e.snapshot['points'])==13 and e.snapshot['timestamp']=='2025-04-30 00:00:00'and e.state()['completed_history_hours']==3600)
        check('source_bytes_and_mtime_unchanged',before==(SOURCE.stat().st_size,SOURCE.stat().st_mtime_ns))
        return {'availability_summary':{g:{k:v for k,v in all_a[g].items()if k in['min_date','max_date','period_count','complete_period_count','partial_last_day','last_day_samples','last_timestamp']}for g in all_a},'notification_regression_checks':base.checks}
    finally:source.mm.close();source.file.close()

if __name__=='__main__':
    status='passed';extra={}
    try:extra=main()
    except Exception as exc:
        import traceback;traceback.print_exc();status='failed';checks.append({'exception_type':type(exc).__name__,'message':str(exc)})
    output=OUT/'source_range_revision_verification.json'
    output.write_text(json.dumps({'status':status,'checks':checks,'production_POST_or_server_restart':False,'production_DB_used':False,'external_transport_calls':0,**extra},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':status,'checks':len(checks),'path':str(output)}))
    gc.collect();base.temporary.cleanup()
    sys.exit(0 if status=='passed'else 1)
