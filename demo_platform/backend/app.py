"""Local full-stack demo. Bind only127.0.0.1; no production authentication claim."""
from pathlib import Path
import sys
APP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(APP/'runtime_packages'));sys.dont_write_bytecode=True
import asyncio,queue,json,io,csv,datetime,threading,time
from contextlib import asynccontextmanager
from fastapi import FastAPI,Request,HTTPException
from fastapi.responses import JSONResponse,StreamingResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from config import CONFIG,ROOT
from data_source import SourceStore,SENSOR_KEYS,LABELS,unit,iso
from store import DB,now
from engine import ReplayEngine,load_module,TYPES,augment_records
from services import Reporting,IntegrationError,slack_send,notify_alarm
SOURCE=None;ENGINE=None;REPORTING=None;FORECAST=None;FORECAST_CACHE={};STOP=threading.Event()
DIAGNOSIS_READER=None;DIAGNOSIS_READER_MODULE=None;DIAGNOSIS_READER_LOCK=threading.Lock()
def forecast(module='all',horizon=24,origin=None):
    if module!='all'and module not in{m['id']for m in SOURCE.modules}:raise ValueError('알 수 없는 설비입니다.')
    cursor=pd_time(origin).floor('h')if origin is not None else ENGINE.cursor.floor('h')
    if cursor>ENGINE.cursor.floor('h'):raise ValueError('현재 재생 시점 이후를 학습 원점으로 사용할 수 없습니다.')
    key=(module,iso(cursor),int(horizon))
    if key not in FORECAST_CACHE:
        series=SOURCE.history_energy(module,cursor)
        if series.empty:
            return {'status':'awaiting_completed_hour','history_count':0,'forecast':[],'history':[],
                    'origin':iso(cursor),'module_id':module,'horizon':int(horizon),'uses_only_origin_history':True,
                    'quality_note':'완료된1시간 관측이 쌓이면 과거 관측만으로 예측을 시작합니다. 현재는 예측값을 생성하지 않습니다.'}
        global FORECAST
        if FORECAST is None:FORECAST=load_module('forecast_serving',APP/'analytics/forecast/api.py')
        result=FORECAST.serving_forecast(series,origin=cursor,points=int(horizon),module_id=module)
        result['history']=[{'timestamp':iso(t),'actual_kwh':float(v)}for t,v in series.tail(48).items()];FORECAST_CACHE[key]=result
        if len(FORECAST_CACHE)>100:FORECAST_CACHE.pop(next(iter(FORECAST_CACHE)))
    return FORECAST_CACHE[key]
def worker():
    while not STOP.wait(5):
        if CONFIG.slack_url():
            for job in DB.due_notifications():
                alarm=json.loads(job['payload']);attempt=job['attempts']+1
                try:notify_alarm(alarm);DB.notification_result(alarm['id'],True,attempt)
                except IntegrationError as e:DB.notification_result(alarm['id'],False,attempt,str(e))
        schedule=REPORTING.schedule();kst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
        if schedule['enabled']and kst.weekday()==schedule['weekday']and kst.hour==schedule['hour']and kst.minute==schedule.get('minute',0):
            stamp=kst.date().isoformat()
            if DB.get('report_schedule_last')!=stamp:
                DB.set('report_schedule_last',stamp)
                try:REPORTING.generate(schedule['module_id'],require_gpt=True);DB.set('report_schedule_status',{'status':'generated','at':now()})
                except Exception as e:DB.set('report_schedule_status',{'status':'failed','at':now(),'message':str(e)if isinstance(e,IntegrationError)else'주간 리포트 생성 실패'})
@asynccontextmanager
async def lifespan(app):
    global SOURCE,ENGINE,REPORTING
    SOURCE=SourceStore();ENGINE=ReplayEngine(SOURCE);REPORTING=Reporting(SOURCE,ENGINE,forecast);CONFIG.public();STOP.clear();threading.Thread(target=worker,daemon=True,name='integration-worker').start()
    yield
    STOP.set();ENGINE.stopped=True
app=FastAPI(title='FactoryLens local demo',lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173','http://localhost:8099','http://127.0.0.1:8099'],allow_methods=['GET','POST'],allow_headers=['Content-Type'])
@app.middleware('http')
async def local_mutations(request:Request,call_next):
    if request.method=='POST':
        origin=request.headers.get('origin')
        if origin and origin not in['http://localhost:5173','http://127.0.0.1:5173','http://localhost:8099','http://127.0.0.1:8099']:return JSONResponse({'detail':'이 로컬 데모의 허용된 출처에서 요청하세요.'},status_code=403)
    try:return await call_next(request)
    except(ValueError,IntegrationError)as e:return JSONResponse({'detail':str(e)},status_code=400)
@app.exception_handler(ValueError)
async def value_error(request,e):return JSONResponse({'detail':str(e)},status_code=400)
@app.exception_handler(IntegrationError)
async def integration_error(request,e):return JSONResponse({'detail':str(e)},status_code=503)
def read_json(path,default):return json.loads(path.read_text(encoding='utf8'))if path.exists()else default
def diagnostic_summary():
    folder=APP/'analytics/diagnosis';metrics=read_json(folder/'results.json',read_json(folder/'training_results.json',{}));archive=read_json(folder/'candidates.json',[])
    metadata=archive if isinstance(archive,dict)else{}
    if isinstance(archive,dict):archive=archive.get('candidates',archive.get('events',[]))
    primary=sorted([a for a in archive if a.get('point_count',0)>=3],key=lambda a:a.get('ae_score',0),reverse=True)
    return {'status':'ready'if ENGINE.detector_ready()else'training_in_progress','candidates':primary,'review_candidates':archive,'type_counts':metadata.get('types',{}),'alarms':DB.alarms()['rows'],'types':[{'id':t,'label':l}for t,l in TYPES.items()],'models':[{'id':'rules','label':'모듈·시간 맥락 규칙','description':'전류 불균형, Q/P, 역률, 전력·전류 관계와 야간 맥락을 비교합니다.'},{'id':'ae','label':'Autoencoder','description':'기준 센서 관계의 재구성 오차를 측정합니다.'},{'id':'iforest','label':'Isolation Forest','description':'희소한 센서 조합을 보수적인 임계값으로 탐지합니다.'},{'id':'classifier','label':'증강 유형 분류','description':'명시적인 합성 고장 유형과의 유사도입니다. 실제 고장 확률이 아닙니다.'}],'metrics':metrics,'coverage':read_json(folder/'scan_summary.json',{}),'limits':['실제 삽입 시각과 정답 라벨 미제공; 약6개라는 안내로 결과 수를 강제하지 않습니다.','분류 성능은 독립 합성 테스트 성능이며 실제 고장 성능으로 해석할 수 없습니다.','알람은3연속5초 관측 후 기록합니다. 임계값 초과 단일점은 로그의 검토 후보로 남습니다.']}
@app.get('/api/health')
def health():return {'status':'ok','product':'FactoryLens','version':'1.0','data_ready':SOURCE is not None,'diagnosis_ready':ENGINE.detector_ready(),'forecast_ready':(APP/'analytics/forecast/api.py').exists()}
@app.get('/api/bootstrap')
def bootstrap():return {'meta':{'product':'FactoryLens','project_title':'제조 설비 에너지·이상 진단 플랫폼','team_name':'FactoryLens 데모','source_mode':'recorded_replay','data_range':{'start':iso(SOURCE.raw_range()[0]),'end':iso(SOURCE.raw_range()[1])},'series_availability':SOURCE.series_availability,'scope_note':'13개 계측 모듈 · 분전반/개별 설비의 계층 미확인','timezone_note':'원본 localtime 기록 시각 · 실제 시간대 미확인; 서버 수신은 UTC','sampling_seconds':5},'modules':SOURCE.modules,'sensors':[{'key':k,'label':LABELS[k],'unit':unit(k)}for k in SENSOR_KEYS+['energy_kwh','pf_mean','current_unbalance_pct']],'scenarios':ENGINE.scenarios,'replay':ENGINE.state(),'snapshot':ENGINE.snapshot,'history':ENGINE.history,'integrations':CONFIG.public(),'forecast_summary':read_json(APP/'analytics/forecast/dashboard_summary.json',{'selected_model':'실험 진행 중','quality_note':'현재 재생 예측과 가이드745시간 실험은 별도 원점입니다.'}),'diagnosis_summary':diagnostic_summary()}
@app.post('/api/replay')
async def replay(request:Request):
    body=await request.json()
    if body.get('action')=='jump_event':
        event_id=body.get('event_id')
        if not isinstance(event_id,str)or not 1<=len(event_id)<=128:raise ValueError('재생할 후보 식별자가 필요합니다.')
        return await asyncio.to_thread(ENGINE.jump_event,resolve_diagnosis_event(event_id))
    return await asyncio.to_thread(ENGINE.control,**{k:v for k,v in body.items()if k in['action','scenario_id','timestamp','speed']})
@app.get('/api/stream')
async def stream(request:Request):
    q=queue.Queue(maxsize=8);ENGINE.listeners.append(q)
    async def events():
        try:
            yield 'data: '+json.dumps({'type':'connected','replay':ENGINE.state(),'snapshot':ENGINE.snapshot,'new_detections':[]},ensure_ascii=False,default=str)+'\n\n'
            while not await request.is_disconnected():
                try:item=await asyncio.to_thread(q.get,True,10);yield 'data: '+json.dumps(item,ensure_ascii=False,allow_nan=False,default=str)+'\n\n'
                except queue.Empty:yield ': heartbeat\n\n'
        finally:
            if q in ENGINE.listeners:ENGINE.listeners.remove(q)
    return StreamingResponse(events(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})
@app.get('/api/series')
def series(module_id='all',sensor='activePower',start=None,end=None,grain='hour',range_mode='timestamps'):
    explicit_dates=bool(start or end);end=pd_time(end)if end else pd_time(ENGINE.snapshot['timestamp'])+datetime.timedelta(seconds=5);start=pd_time(start)if start else end-datetime.timedelta(minutes=10)if grain=='raw'else end-datetime.timedelta(days=7)
    start,end=SOURCE.validate_series_range(module_id,grain,start,end,range_mode,explicit_dates)
    rows=SOURCE.series(module_id,sensor,start,end,grain)if start is not None and end>start else[]
    return {'module_id':module_id,'sensor':sensor,'unit':unit(sensor),'grain':grain,'source_kind':'recorded','query_scope':'historical_archive'if explicit_dates else'available_replay_prefix','range_mode':range_mode,
            'effective_start':iso(start)if start is not None else None,'effective_end_exclusive':iso(end)if end is not None else None,'series':rows,
            'coverage_note':'원자료5초/조회종료 이전 완료시간 집계. 명시적 날짜는 과거 기록 조회, 날짜 생략은 재생시점 가용 범위. 전체 모듈 전력은 합계, 전압·전류·역률은 모듈 평균. 주는 일요일 종료 라벨이며 부분 주·월은 complete=false로 표시합니다. 단위와 계측 계층 미확인.'}
def pd_time(value):
    import pandas as pd
    try:return pd.Timestamp(value)
    except Exception:raise ValueError('올바른 기록 시각을 입력하세요.')from None
@app.get('/api/logs')
def logs(module_id='all',limit:int=100):return DB.logs(module_id,min(max(limit,1),1000))
@app.get('/api/alarms')
def alarms(module_id='all',limit:int=1000):return DB.alarms(module_id,min(max(limit,1),1000))
@app.post('/api/alarms/{id}/ack')
def ack(id):return DB.ack(id)
@app.get('/api/diagnosis')
def diagnosis():return diagnostic_summary()
@app.get('/api/diagnosis/{event_id}')
def diagnosis_event(event_id):
    event=resolve_diagnosis_event(event_id)
    t=pd_time(event.get('time',event.get('start',event.get('start_time'))))
    return diagnosis_window_data(event,t-datetime.timedelta(minutes=3),t+datetime.timedelta(minutes=3))
def diagnosis_reader(required=False):
    """Scoring may be lazy, but it must never initialize or control ENGINE."""
    global DIAGNOSIS_READER,DIAGNOSIS_READER_MODULE
    with DIAGNOSIS_READER_LOCK:
        try:
            if DIAGNOSIS_READER_MODULE is None:DIAGNOSIS_READER_MODULE=load_module('diagnosis_readonly',APP/'analytics/diagnosis/detector.py')
            detector=getattr(ENGINE,'detector',None)
            if detector is None:
                if DIAGNOSIS_READER is None:DIAGNOSIS_READER=DIAGNOSIS_READER_MODULE.Detector()
                detector=DIAGNOSIS_READER
            return detector,DIAGNOSIS_READER_MODULE
        except Exception:
            if required:raise HTTPException(503,'진단 점수 모델이 아직 준비되지 않았습니다. 원본을 가짜 점수로 대체하지 않습니다.')from None
            return None,None
def diagnosis_window_data(event,start,end,required_scores=False):
    mid=str(event['module_id']);rows=SOURCE.window(start,end,[mid]);source_kind=event.get('source_kind','recorded')
    if source_kind=='synthetic':
        with ENGINE.lock:scenario=next((dict(s)for s in ENGINE.scenarios if s['id']==event.get('scenario_id')),None)
        if not scenario:raise ValueError('증강 시연의 변형 조건을 찾을 수 없습니다. 원본 그래프로 대체하지 않습니다.')
        rows=augment_records(rows,scenario)
    elif source_kind!='recorded':raise ValueError('알 수 없는 후보 원천입니다.')
    detector,module=diagnosis_reader(required_scores)if rows else(None,None)
    scored=detector.diagnose_batch(rows)if detector is not None else[];rules=[]
    if detector is not None and rows:
        all_scores=detector.scores(module.prepare(rows),True)
        for i,row in enumerate(rows):
            rules.append({'time':row['time'],'rules':[{'type':kind,'label':TYPES[kind],'score':float(all_scores['rule_scores'][i,j]),'threshold':float(all_scores['rule_thresholds'][i,j]),'alarm':bool(all_scores['rule_flags'][i,j])}for j,kind in enumerate(TYPES)]})
    return {'event':event,'series':rows,'algorithm':rules,'ai':[{'time':d['time'],**d['ai']}for d in scored],'source_kind':source_kind,'coverage_note':'규칙6종·AE·IF를 해당 원본 또는 명시된 증강 구간 전체에 적용; 실제 고장 라벨 미제공'}
def diagnosis_padding(value):
    text=str(value)
    if not text.isascii()or not text.isdecimal():raise ValueError('전후 구간은0~1800초의5초 단위 정수로 지정하세요.')
    seconds=int(text)
    if not 0<=seconds<=1800 or seconds%5:raise ValueError('전후 구간은0~1800초의5초 단위 정수로 지정하세요.')
    return seconds
@app.get('/api/diagnosis/{event_id}/replay-window')
def diagnosis_replay_window(event_id,before_seconds:str='60',after_seconds:str='60'):
    before=diagnosis_padding(before_seconds);after=diagnosis_padding(after_seconds);event=resolve_diagnosis_event(event_id)
    event_start=pd_time(event.get('start')or event.get('first_time')or event.get('start_time')or event.get('time'))
    event_end=pd_time(event.get('end')or event.get('end_time')or event.get('time')or event_start)
    if event_start is None or event_end is None or event_start!=event_start or event_end!=event_end or event_start.tzinfo is not None or event_end.tzinfo is not None or event_end<event_start:raise ValueError('후보의 원본 시작·끝 기록 시각을 확인하세요.')
    mid=str(event['module_id'])
    if mid not in SOURCE.bounds:raise ValueError('알 수 없는 후보 설비입니다.')
    first,last=SOURCE.raw_range(mid)
    if(event_start-first).total_seconds()%5 or(event_end-first).total_seconds()%5:raise ValueError('후보 시각은 원본5초 관측 간격에 맞아야 합니다.')
    requested_start=event_start-datetime.timedelta(seconds=before);requested_end=event_end+datetime.timedelta(seconds=after)
    start=max(first,requested_start);end=min(last,requested_end)
    if end<start:raise ValueError('후보 구간과 보유 원본 기간이 겹치지 않습니다.')
    end_exclusive=end+datetime.timedelta(seconds=5)
    result=diagnosis_window_data(event,start,end_exclusive,required_scores=True)
    result['window']={'start':iso(start),'end_inclusive':iso(end),'end_exclusive':iso(end_exclusive),
        'event_start':iso(event_start),'event_end':iso(event_end),'requested_before_seconds':before,'requested_after_seconds':after,
        'actual_before_seconds':max(0,int((event_start-start).total_seconds())),
        'actual_after_seconds':max(0,int((end-event_end).total_seconds())),
        'sampling_seconds':5,'point_count':len(result['series']),'clipped':start!=requested_start or end!=requested_end}
    return result
def resolve_diagnosis_event(event_id):
    event=ENGINE.detection_occurrence(event_id)or DB.alarm_by_id(event_id)
    if event:return event
    candidates=read_json(APP/'analytics/diagnosis/candidates.json',[])
    if isinstance(candidates,dict):candidates=candidates.get('candidates',candidates.get('events',[]))
    event=next((a for a in candidates if str(a.get('id',a.get('event_id')))==event_id),None)
    if not event:raise HTTPException(404,'후보 구간을 찾을 수 없습니다.')
    return event
@app.get('/api/forecast')
def forecast_route(module_id='all',horizon:int=24):return forecast(module_id,horizon)
@app.get('/api/forecast/evaluation')
def evaluation():
    folder=APP/'analytics/forecast';path=folder/'dashboard_evaluation.json'
    if path.exists():return read_json(path,{})
    return {'status':'training_in_progress','models':[],'selected_model':None,'series':[],'horizon_note':'Apr30 원점부터745시간 경로; May10~May31 정답505시간 평가','quality_note':'시간순 검증으로 설정을 확정한 뒤 May 평가를 진행합니다. 실험 진행 중입니다.','scale_audit':'평가 정답으로 사후 이동/배율을 맞추지 않습니다.'}
@app.get('/api/reports')
def reports():return {'rows':DB.reports(),'schedule':REPORTING.schedule(),'schedule_status':DB.get('report_schedule_status'),'integrations':CONFIG.public()}
@app.post('/api/reports/generate')
async def report_generate(request:Request):
    body=await request.json();return await asyncio.to_thread(REPORTING.generate,str(body.get('module_id',body.get('module','all'))),body.get('end'),bool(body.get('require_gpt',False)),bool(body.get('local_only',False)))
@app.post('/api/reports/schedule')
async def schedule(request:Request):return REPORTING.save_schedule(await request.json())
@app.post('/api/chat')
async def chat(request:Request):
    body=await request.json();return await asyncio.to_thread(REPORTING.chat,body.get('question',''),body.get('context',{}),bool(body.get('local_only',False)))
@app.get('/api/integrations')
def integrations():return CONFIG.public()
@app.post('/api/integrations/openai/consent')
async def gpt_consent(request:Request):
    body=await request.json()
    if not isinstance(body.get('approved'),bool):raise ValueError('통계·질문 전송 승인 여부를 명시하세요.')
    DB.set('gpt_data_transfer_approved',body['approved']);DB.set('gpt_data_transfer_consent_record',{'approved':body['approved'],'at':now(),'version':'weekly_stats_alarm_evidence_forecast_quality_and_query_context_v1','source':'explicit_user_interface_choice'})
    return CONFIG.public()
@app.post('/api/integrations/slack')
async def slack_config(request:Request):
    body=await request.json();return CONFIG.configure_slack(body.get('webhook_url'),body.get('enabled'))
@app.post('/api/integrations/slack/test')
def slack_test():return slack_send('FactoryLens 연결 테스트 · 실제 고장 알람이 아닌 사용자가 요청한 데모 연동 테스트입니다.')
@app.get('/api/export/{name}.csv')
def export(name):
    if name not in['logs','alarms']:raise HTTPException(404,'내보낼 항목을 찾을 수 없습니다.')
    rows=DB.logs(limit=1000)['rows']if name=='logs'else DB.alarms()['rows'];out=io.StringIO()
    if rows:
        columns=list(rows[0]);writer=csv.DictWriter(out,fieldnames=columns,extrasaction='ignore');writer.writeheader()
        for r in rows:writer.writerow({k:(json.dumps(v,ensure_ascii=False)if isinstance(v,(dict,list))else v)for k,v in r.items()})
    return StreamingResponse(iter(['\ufeff'+out.getvalue()]),media_type='text/csv',headers={'Content-Disposition':f'attachment; filename="{name}.csv"'})
DIST=APP/'frontend/dist'
if DIST.exists():
    if(DIST/'assets').exists():app.mount('/assets',StaticFiles(directory=DIST/'assets'),name='assets')
    @app.get('/')
    def index():return FileResponse(DIST/'index.html')
if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8099,access_log=False)
