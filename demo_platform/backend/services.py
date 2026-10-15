"""Server-side OpenAI and Slack transports with allowlisted data queries."""
import json,urllib.request,urllib.error,uuid,datetime,threading,time
import pandas as pd
from config import CONFIG,MODEL
from store import DB,now
from data_source import iso,unit
REPORT_SCHEMA={'type':'object','properties':{'title':{'type':'string'},'summary':{'type':'string'},'sections':{'type':'array','items':{'type':'object','properties':{'title':{'type':'string'},'content':{'type':'string'}},'required':['title','content'],'additionalProperties':False}},'actions':{'type':'array','items':{'type':'string'}},'limitations':{'type':'array','items':{'type':'string'}}},'required':['title','summary','sections','actions','limitations'],'additionalProperties':False}
QUERY_SCHEMA={'type':'object','properties':{'kind':{'type':'string','enum':['sensor_summary','fault_events','forecast_metrics','equipment_lookup']},'sensor':{'type':'string','enum':['activePower','energy_kwh','pf_mean','currentR','currentS','currentT','voltageR','voltageS','voltageT','reactivePowerLagging','current_unbalance_pct']},'module_id':{'type':'string','enum':['all','1','2','3','4','5','11','12','13','14','15','16','17','18']},'days':{'type':'integer','minimum':1,'maximum':30}},'required':['kind','sensor','module_id','days'],'additionalProperties':False}
class IntegrationError(Exception):pass
def gpt_json(system,payload,schema,name):
    if not DB.get('gpt_data_transfer_approved',False):raise IntegrationError('OpenAI로 전송할 통계·질문 항목에 대한 승인이 필요합니다. 로컬 통계·검색은 계속 사용할 수 있습니다.')
    key=CONFIG.api_key()
    if not key:raise IntegrationError('가이드에서 OpenAI 키를 확인하지 못했습니다. 서버 환경의 OPENAI_API_KEY를 설정하면 GPT-5 mini를 사용할 수 있습니다.')
    body={'model':MODEL,'store':False,'reasoning':{'effort':'low'},'max_output_tokens':4000,'input':[{'role':'system','content':system},{'role':'user','content':json.dumps(payload,ensure_ascii=False,allow_nan=False,default=str)}],'text':{'format':{'type':'json_schema','name':name,'strict':True,'schema':schema}}}
    request=urllib.request.Request('https://api.openai.com/v1/responses',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
    try:
        with urllib.request.urlopen(request,timeout=90)as response:data=json.load(response)
        if data.get('status')!='completed':raise IntegrationError('OpenAI 응답이 완료되지 않았습니다. 다시 시도하세요.')
        texts=[c.get('text','')for o in data.get('output',[])for c in o.get('content',[])if c.get('type')=='output_text']
        result=json.loads(''.join(texts));CONFIG.openai_status='connected';return result
    except urllib.error.HTTPError as e:
        CONFIG.openai_status='request_failed';raise IntegrationError(f'OpenAI 요청 실패(HTTP {e.code}). 키·접근 권한·결제 설정을 확인하세요.')from None
    except IntegrationError:raise
    except Exception:
        CONFIG.openai_status='request_failed';raise IntegrationError('OpenAI 연결 또는 응답 검증에 실패했습니다. 민감한 응답 내용은 로그에 저장하지 않습니다.')from None
def slack_send(text,blocks=None):
    url=CONFIG.slack_url()
    if not url:raise IntegrationError('Slack 웹훅을 설정하고 알림을 활성화하세요.')
    payload={'text':text}
    if blocks:payload['blocks']=blocks
    request=urllib.request.Request(url,data=json.dumps(payload,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'},method='POST')
    try:
        with urllib.request.urlopen(request,timeout=15)as response:body=response.read().decode()
        if body.strip()!='ok':raise IntegrationError('Slack이 메시지 수락을 확인하지 않았습니다.')
        CONFIG.slack_status='connected';return {'status':'sent','destination':'configured_incoming_webhook'}
    except urllib.error.HTTPError as e:CONFIG.slack_status='request_failed';raise IntegrationError(f'Slack 전송 실패(HTTP {e.code}).')from None
    except IntegrationError:raise
    except Exception:CONFIG.slack_status='request_failed';raise IntegrationError('Slack 연결에 실패했습니다. 웹훅 주소는 로그에 저장하지 않습니다.')from None
def notify_alarm(a):
    title=f"FactoryLens · {a['title']} 의심 후보"
    detail=f"설비: {a['equipment']}\n기록: {a['time']}\n원천: {'증강 시연'if a['source_kind']=='synthetic'else'실제 기록 재생'}\n상태: 실제 고장 여부 미확인"
    evidence=json.dumps(a.get('evidence',{}),ensure_ascii=False,default=str)[:1400]
    return slack_send(title,[{'type':'header','text':{'type':'plain_text','text':title}},{'type':'section','text':{'type':'mrkdwn','text':detail}},{'type':'section','text':{'type':'plain_text','text':'탐지 근거: '+evidence}},{'type':'context','elements':[{'type':'plain_text','text':'알람 확인은 고장 확정이 아닙니다. 원본 데이터의 실제 고장 라벨은 제공되지 않았습니다.'}]}])
class Reporting:
    def __init__(self,source,engine,forecast):self.source=source;self.engine=engine;self.forecast=forecast;self.lock=threading.Lock()
    def facts(self,module_id,end=None):
        available=self.engine.cursor.floor('h');cursor=pd.Timestamp(end).floor('h')if end else available
        if cursor>available:raise ValueError('보고서 종료 시각은 현재 재생 시점의 완료된 시간 이후로 지정할 수 없습니다.')
        start=cursor-pd.Timedelta(days=7)
        h=self.source.hourly[(self.source.hourly.time>=start)&(self.source.hourly.time+pd.Timedelta(hours=1)<=cursor)]
        if module_id!='all':h=h[h.module_id==str(module_id)]
        series=h.groupby('time').energy_kwh.sum();alarm_query=DB.alarms(module_id,30,start=iso(start),end=iso(cursor),source_kind='recorded');alarms=alarm_query['rows']
        prediction=self.forecast(module_id,24,origin=cursor)
        return {'module_id':module_id,'week_start':iso(start),'week_end':iso(cursor),'observed_hours':len(series),'expected_hours':168,'energy_kwh':float(series.sum()),'mean_hourly_kwh':float(series.mean())if len(series)else None,'minimum_hourly_kwh':float(series.min())if len(series)else None,'maximum_hourly_kwh':float(series.max())if len(series)else None,'project_bill_won':float(series.sum())*180,'project_carbon_kg':float(series.sum())*.424,'logged_alarm_candidates':alarms,'logged_alarm_candidates_total':alarm_query['total'],'alarm_archive_limit':'재생되어 로그에 저장된 후보만 집계; 전체 원본 고장 개수가 아님. 상세근거 최대30개,총건수 별도','forecast_origin':prediction.get('origin'),'forecast_next_24h_sum_kwh':sum(f['predicted_kwh']for f in prediction.get('forecast',[])),'forecast_quality':prediction.get('quality_note'),'source_mode':'recorded_replay','scope':'13계측 모듈 합계(설비 계층 미확인)'if module_id=='all'else'단일 모듈','unit':'가이드 단위의 잠정 kWh','coefficients':{'won_per_kwh':180,'kgCO2_per_kwh':.424},'fault_labels_available':False}
    def generate(self,module_id='all',end=None,require_gpt=False,local_only=False):
        with self.lock:
            facts=self.facts(str(module_id),end);created=now();identifier=uuid.uuid4().hex
            if not local_only and CONFIG.api_key()and DB.get('gpt_data_transfer_approved',False):
                report=gpt_json('당신은 제조 설비 운영 보고서 작성자입니다. 입력 facts의 수치만 사용하고 한국어로 작성하세요. 실제 라벨이 없는 고장을 확정하지 마세요. 관측 기간과 미래 예측 원점을 구별하세요. 단위·계층 미확인, 가이드 비용/탄소 계수, 예측 품질 제한을 명시하세요. 새로운 수치나 고객·고장 원인을 만들지 마세요. 실행 가능한 점검 권고를 제공하세요.',facts,REPORT_SCHEMA,'weekly_equipment_report');status='generated';mode='gpt-5-mini'
            else:
                if require_gpt:raise IntegrationError('GPT-5 mini 서버 API 키와 통계 전송 승인을 확인하세요.')
                status='local_summary';mode='local_statistics';report={'title':'주간 운영 통계 · 로컬 요약','summary':f"{facts['week_start']}부터 {facts['week_end']}까지 {facts['observed_hours']}개 시간 관측에서 전력량 합계는 {facts['energy_kwh']:,.2f} 잠정 kWh입니다.",'sections':[{'title':'관측 요약','content':f"가이드 계수 기준 비용 {facts['project_bill_won']:,.0f}원, 탄소 {facts['project_carbon_kg']:,.1f}kgCO₂. 저장된 원본 재생 알람 후보 {facts['logged_alarm_candidates_total']}건."},{'title':'예측','content':str(facts['forecast_quality'])}],'actions':['의심 시각의 상별 전압·전류·역률을 확인하고 실제 고장 라벨을 수집하세요.'],'limitations':['서버에서 계산한 통계 요약입니다. GPT가 생성한 보고서가 아닙니다.','단위·설비 계층·실제 고장 라벨이 미확인입니다.']}
            return DB.save_report({'id':identifier,'generated_at':created,'week_start':facts['week_start'],'week_end':facts['week_end'],'status':status,'model':mode,'facts':facts,'source_mode':'recorded_replay',**report})
    def schedule(self):return DB.get('report_schedule',{'enabled':False,'weekday':0,'hour':9,'time':'09:00','timezone':'Asia/Seoul','module_id':'all','execution_note':'데모 서버 실행 중에만 동작합니다. 종료 중 놓친 일정을 자동 재전송하지 않습니다.'})
    def save_schedule(self,body):
        days={'mon':0,'tue':1,'wed':2,'thu':3,'fri':4,'sat':5,'sun':6};weekday=body.get('weekday',0);weekday=days.get(str(weekday),weekday)
        hour=int(body.get('hour',str(body.get('time','09:00')).split(':')[0]));minute=int(str(body.get('time','09:00')).split(':')[1]);weekday=int(weekday)
        if not 0<=weekday<=6 or not 0<=hour<=23 or not 0<=minute<=59:raise ValueError('주간 일정의 요일과 시간을 확인하세요.')
        schedule={'enabled':bool(body.get('enabled',False)),'weekday':weekday,'hour':hour,'minute':minute,'time':f'{hour:02}:{minute:02}','timezone':'Asia/Seoul','module_id':str(body.get('module_id',body.get('module','all'))),'execution_note':'데모 서버 실행 중 KST 일정에 생성합니다. 실제 GPT 요청은 API 키 설정 시에만 진행합니다.'};DB.set('report_schedule',schedule);return schedule
    @staticmethod
    def _chat_time(value):
        # Client timestamps are recorded naive localtime, never Unix epochs or UTC conversions.
        if not isinstance(value,(str,pd.Timestamp))or isinstance(value,bool)or not str(value).strip():raise ValueError('시간대 없는 원본 기록 시각을 입력하세요.')
        try:stamp=pd.Timestamp(value)
        except(Exception):raise ValueError('올바른 원본 기록 시각을 입력하세요.')from None
        if pd.isna(stamp)or stamp.tzinfo is not None or stamp.microsecond or stamp.nanosecond:raise ValueError('초 단위의 시간대 없는 원본 기록 시각을 입력하세요.')
        return stamp
    def _chat_sensor_rows(self,module,sensor,start,end,grain):
        if grain=='raw':
            raw=self.source.window(start,end,None if module=='all'else[module])
            if not raw:return [],{'source_observation_count':0,'record_unit':'source_row','observed_start':None,'observed_end_exclusive':None}
            frame=pd.DataFrame(raw);agg='sum'if module=='all'and sensor in['activePower','reactivePowerLagging','energy_kwh']else'mean'
            grouped=frame.groupby('time')[sensor].agg(['count',agg]);expected=len(self.source.modules)if module=='all'else 1
            rows=[{'timestamp':str(t),'value':float(r[agg]),'records':int(r['count']),'complete':int(r['count'])==expected}for t,r in grouped.iterrows()]
            return rows,{'source_observation_count':len(raw),'record_unit':'source_row','observed_start':rows[0]['timestamp'],'observed_end_exclusive':iso(pd.Timestamp(rows[-1]['timestamp'])+pd.Timedelta(seconds=5))}
        rows=self.source.series(module,sensor,start,end,grain)
        frame=self.source.hourly
        frame=frame[(frame.time>=start)&(frame.time+pd.Timedelta(hours=1)<=end)]
        if module!='all':frame=frame[frame.module_id==module]
        return rows,{'source_observation_count':int(frame.records.sum()),'record_unit':'completed_hour',
                     'observed_start':iso(frame.time.min())if len(frame)else None,'observed_end_exclusive':iso(frame.time.max()+pd.Timedelta(hours=1))if len(frame)else None}
    def chat(self,question,context,local_only=False):
        if not isinstance(question,str)or not 1<=len(question.strip())<=2000:raise ValueError('질문은1~2000자로 입력하세요.')
        if not isinstance(context,dict):raise ValueError('검색 맥락은 객체로 입력하세요.')
        module=str(context.get('module_id','all'));allowed={str(m['id'])for m in self.source.modules}|{'all'}
        if module not in allowed:raise ValueError('알 수 없는 설비입니다.')
        scope=context.get('scope','replay');grain=context.get('grain','raw'if scope=='diagnosis'else'day')
        if scope not in['replay','archive','diagnosis']:raise ValueError('알 수 없는 검색 범위입니다.')
        if grain not in['raw','hour','day','week','month']:raise ValueError('알 수 없는 집계 간격입니다.')
        if context.get('source_kind','recorded')!='recorded':raise ValueError('검색은 원본 관측만 지원합니다. 합성 시연 센서와 구분하세요.')
        explicit=scope in['archive','diagnosis']
        if scope=='diagnosis'and grain!='raw':raise ValueError('진단 구간 검색은5초 원본 간격을 사용하세요.')
        cursor_value=context.get('cursor_time');cursor=self._chat_time(self.engine.cursor if cursor_value is None else cursor_value)
        if not explicit and cursor.floor('h')>self.engine.cursor.floor('h'):raise ValueError('검색 시각은 현재 재생 시점 이후로 지정할 수 없습니다.')
        if explicit:
            start=self._chat_time(context.get('start'));end=self._chat_time(context.get('end'))
            start,end=self.source.validate_series_range(module,grain,start,end,explicit=True)
        source='local_data_search'
        if not explicit and not local_only and CONFIG.api_key()and DB.get('gpt_data_transfer_approved',False):
            # Only these previously approved fields may leave the process. Scope/page/ranges stay local.
            transfer={'question':question,'context':{'module_id':module,'cursor_time':iso(cursor)}}
            plan=gpt_json('질문을 허용된 데이터 검색 계획으로 변환하세요. SQL이나 코드를 생성하지 마세요. 맥락의 모듈을 기본값으로 사용하세요. 고장 질문은 fault_events, 예측 정확도는 forecast_metrics, 설비 목록은 equipment_lookup, 나머지는 sensor_summary. 기간 명시 없으면7일.',transfer,QUERY_SCHEMA,'equipment_query');source=MODEL
        else:
            sensor='energy_kwh'
            for word,key in [('역률','pf_mean'),('불균형','current_unbalance_pct'),('전압','voltageR'),('전류','currentR'),('무효','reactivePowerLagging'),('전력','activePower')]:
                if word in question:sensor=key;break
            kind='fault_events'if any(w in question for w in ['고장','이상','알람'])else'forecast_metrics'if any(w in question for w in ['예측','mae','MAE','모델'])else'equipment_lookup'if '목록'in question else'sensor_summary'
            plan={'kind':kind,'module_id':module,'sensor':sensor,'days':7}
        if not isinstance(plan,dict)or plan.get('module_id')not in allowed or plan.get('kind')not in QUERY_SCHEMA['properties']['kind']['enum']or plan.get('sensor')not in QUERY_SCHEMA['properties']['sensor']['enum']:raise ValueError('허용된 검색 계획이 아닙니다.')
        days=plan.get('days')
        if isinstance(days,bool)or not isinstance(days,int)or not 1<=days<=30:raise ValueError('검색 기간은1~30일로 지정하세요.')
        if explicit:plan['module_id']=module
        else:
            module=plan['module_id'];end=cursor.floor('h');start=end-pd.Timedelta(days=days)
            start,end=self.source.validate_series_range(module,grain,start,end,explicit=False)
            if grain=='raw'and end-start>pd.Timedelta(hours=24):raise ValueError('5초 원본 검색은24시간 이하의 명시 범위를 사용하세요.')
        query_scope={'module_id':module,'start':iso(start),'end_exclusive':iso(end),'scope':scope,'grain':grain,'source_kind':'recorded'}
        stats={'source_observation_count':None,'record_unit':None};result_unit=None
        if plan['kind']=='fault_events':
            result=DB.alarms(module,30,start=iso(start),end=iso(end),source_kind='recorded');rows=result['rows']
            stats.update(record_unit='stored_alarm',matched_count=int(result['total']));result_unit='후보 건'
            answer=f"{iso(start)} 이상~{iso(end)} 미만의 저장된 원본 재생 로그에서 이상 후보 {result['total']}건 중 {len(rows)}건을 조회했습니다. 실제 고장 확정 결과나 전체 원본 후보 수가 아닙니다."
        elif plan['kind']=='equipment_lookup':
            rows=[m for m in self.source.modules if module=='all'or str(m['id'])==module];stats['record_unit']='module';result_unit='계측 모듈'
            answer=f'원자료의 선택 범위에 {len(rows)}개 계측 모듈이 있습니다. 모듈 목록은 기간별 관측 통계가 아닙니다. 분전반과 설비 계층은 확인이 필요합니다.'
        elif plan['kind']=='forecast_metrics':
            stats['record_unit']='model_evaluation'
            if explicit:rows=[];answer='이 명시 구간 검색은 원본 센서와 저장 알림을 조회합니다. 해당 구간의 예측 정확도를 새로 계산하지 않습니다. 고정 평가 구간의 모델 비교는 예측·비용 화면에서 확인하세요.'
            else:
                prediction=self.forecast(module,24,origin=cursor.floor('h'));rows=[prediction.get('metrics')or{'status':'평가 관측 부족'}];answer=prediction.get('quality_note','예측 평가 준비 중')
        else:
            rows,observations=self._chat_sensor_rows(module,plan['sensor'],start,end,grain);stats.update(observations);result_unit=unit(plan['sensor'])
            stats['partial_period_count']=sum(r.get('complete')is False for r in rows)
            stats['aggregation']='5초 관측 전력량'if grain=='raw'and plan['sensor']=='energy_kwh'else'기간 전력량 합계'if plan['sensor']=='energy_kwh'else'5초 관측값'if grain=='raw'else'완료시간 센서 평균의 기간 평균'
            values=[r['value']for r in rows];stats['mean_display_value']=sum(values)/len(values)if values else None
            if plan['sensor']=='energy_kwh':stats['sum_energy_kwh']=sum(values)
            labels={'raw':'5초 원본','hour':'시간','day':'일','week':'주','month':'월'}
            answer=f"{iso(start)} 이상~{iso(end)} 미만의 {plan['sensor']} {labels[grain]} 표시값 {len(rows)}개를 조회했습니다. 단위: {result_unit}. 원본 관측 {stats['source_observation_count']}행."
            if values:answer+=f" 표시값의 산술 평균 {stats['mean_display_value']:,.3f} {result_unit}."
            if grain!='raw':answer+=' 완료된 시간만 집계하며, 주·월 라벨은 실제 관측 끝과 다를 수 있습니다.'
            if stats['partial_period_count']:answer+=f" 부분 기간 {stats['partial_period_count']}개를 포함합니다."
        stats.update(row_count=len(rows),unit=result_unit)
        return {'answer':answer,'source':source,'query':plan,'query_scope':query_scope,'rows':rows,'row_count':len(rows),'source_observation_count':stats['source_observation_count'],'unit':result_unit,'record_unit':stats['record_unit'],'statistics':stats,
                'citations':[{'title':'원본 RTU 센서/저장 재생 알림','reference':'traing-dataset/rtu_data_full.csv 또는 demo_platform/data/platform.sqlite','as_of':iso(end),'start':iso(start),'end_exclusive':iso(end)}],
                'limitations':'원본 관측 기준 검색이며 합성 시연 값은 포함하지 않습니다. 가이드 단위 미검증 · 실제 고장 라벨 미제공; 서버에서 허용한 검색만 실행합니다.'}
