"""Historical records replayed at physical five-second intervals with durable logging."""
import json,threading,time,uuid,importlib.util,hashlib
from collections import OrderedDict
import pandas as pd
from config import APP
from store import DB,now,alarm_id
from data_source import iso
TYPES={'load_imbalance':'부하 불균형','reactive_power_surge':'무효전력 급증','power_factor_drop':'역률 급감','abnormal_pattern':'비정상 패턴','night_operation':'야간 이상가동','current_only_increase':'전류만 증가'}
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def augment_records(rows,scenario):
    """Pure deterministic replay transform; never mutates the source or engine."""
    output=[dict(r,source_kind=scenario['source_kind'])for r in rows]
    if scenario['source_kind']!='synthetic':return output
    for r in output:
        elapsed=(pd.Timestamp(r['time'])-pd.Timestamp(scenario.get('augmentation_origin_start',scenario['start']))).total_seconds();kind=scenario['fault_type']
        if r['module_id']!=scenario['module_id']or not 120<=elapsed<300:continue
        if kind=='load_imbalance':r['currentR']*=4
        elif kind=='reactive_power_surge':r['reactivePowerLagging']*=5
        elif kind=='power_factor_drop':
            for p in 'RST':r['powerFactor'+p]*=.55
        elif kind=='current_only_increase':
            for p in 'RST':r['current'+p]*=3
        elif kind=='night_operation':
            r['activePower']*=2.4
            for p in 'RST':r['current'+p]*=2.4
        elif kind=='abnormal_pattern':r['activePower']*=1+2.2*min(1,(elapsed-120)/180);r['voltageR']*=.78
        cs=[r['current'+p]for p in 'RST'];mean=sum(cs)/3;r['current_unbalance_pct']=max(abs(c-mean)for c in cs)/mean*100;r['pf_mean']=sum(r['powerFactor'+p]for p in 'RST')/3;r['energy_kwh']=r['activePower']*5/3600;r['augmentation_type']=kind
    return output
class ReplayEngine:
    def __init__(self,source):
        self.source=source;self.lock=threading.RLock();self.playing=False;self.speed=20;self.session=uuid.uuid4().hex;self.sequence=0;self.listeners=[];self.history=[];self.snapshot={};self.detector=None;self.detector_error=None;self.stopped=False;self.detection_occurrences=OrderedDict()
        first=max(pd.to_datetime(b['first'],format='%Y%m%d%H%M%S')for b in source.bounds.values());last=min(pd.to_datetime(b['last'],format='%Y%m%d%H%M%S')for b in source.bounds.values())
        self.scenarios=[{'id':'recorded_full','name':'실제 기록 · 전체 원본 기간','start':iso(first),'end':iso(last+pd.Timedelta(seconds=5)),'source_kind':'recorded'},
            {'id':'recorded_pf','name':'실제 기록 · 역률 의심 구간','start':'2025-03-13 19:43:00','end':'2025-03-13 19:55:00','source_kind':'recorded','module_id':'17','fault_type':'power_factor_drop'}, {'id':'recorded_normal','name':'실제 기록 · 일반 운전 구간','start':'2025-02-18 13:00:00','end':'2025-02-18 13:15:00','source_kind':'recorded'}]
        self.scenarios.extend([{'id':'recorded_ae_night_15','name':'실제 기록 · 15호 심야 AI 후보','start':'2025-02-28 04:36:00','end':'2025-02-28 04:42:00','source_kind':'recorded','module_id':'15','fault_type':'abnormal_pattern'}, {'id':'recorded_ae_night_17','name':'실제 기록 · 17호 심야 AI 후보','start':'2025-02-19 03:02:00','end':'2025-02-19 03:08:00','source_kind':'recorded','module_id':'17','fault_type':'abnormal_pattern'}, {'id':'recorded_weak_pf','name':'실제 기록 · 약한 PF 규칙 후보(검증 필요)','start':'2025-02-26 06:45:00','end':'2025-02-26 06:51:00','source_kind':'recorded','module_id':'5','fault_type':'power_factor_drop'}])
        for t,label in TYPES.items():self.scenarios.append({'id':'synthetic_'+t,'name':'증강 시연 · '+label,'start':'2025-03-14 02:00:00'if t=='night_operation'else'2025-03-14 13:00:00','end':'2025-03-14 02:10:00'if t=='night_operation'else'2025-03-14 13:10:00','source_kind':'synthetic','fault_type':t,'module_id':'17'})
        self.set_scenario('recorded_full');self.thread=threading.Thread(target=self.run,daemon=True,name='recorded-replay');self.thread.start()
    def state(self):
        completed=getattr(self.source,'complete_target',None)
        completed_count=int(completed.index.searchsorted(self.cursor-pd.Timedelta(hours=1),side='right'))if completed is not None else len(self.source.history_energy('all',self.cursor))
        return {'playing':self.playing,'speed':self.speed,'scenario_id':self.scenario['id'],'cursor_time':iso(self.cursor),'start':self.scenario['start'],'end':self.scenario['end'],'source_kind':self.scenario['source_kind'],'session_id':self.session,'sequence':self.sequence,'source_interval_seconds':5,'highlight':self.scenario.get('highlight'),'display_intervals':60,'display_window_seconds':300,
                'session_detection_count':len(self.emitted_occurrences),'session_detected_module_ids':sorted(self.session_detected_modules,key=int),
                'active_detection_module_ids':sorted({str(k[0])for k,n in self.streaks.items()if n>=3},key=int),'completed_history_hours':completed_count}
    def detector_ready(self):
        path=APP/'analytics/diagnosis'
        if self.detector:return True
        if (path/'models.joblib').exists()and(path/'ae.pt').exists():
            try:self.detector=load_module('diagnosis_detector',path/'detector.py').Detector();self.detector_error=None
            except Exception as e:self.detector_error=type(e).__name__
        return self.detector is not None
    def set_scenario(self,scenario_id):
        with self.lock:
            selected=next((s for s in self.scenarios if s['id']==scenario_id),None)
            if not selected:raise ValueError('알 수 없는 시연 구간입니다.')
            self.scenario=selected;self.cursor=pd.Timestamp(selected['start']);self.end=pd.Timestamp(selected['end']);self.session=uuid.uuid4().hex;self.history=[];self.frames={};self.streaks={};self.streak_first={};self.streak_last={};self.emitted_occurrences=set();self.session_detected_modules=set()
            self.load_frame_chunk()
            self.recalculate_energy();self.receive(log=False)
    def load_frame_chunk(self):
        self.frame_start=self.cursor;self.frame_end=min(self.end,self.cursor+pd.Timedelta(minutes=5));self.frames={}
        if self.frame_end<=self.frame_start:return
        for r in self.source.window(self.frame_start,self.frame_end):self.frames.setdefault(r['time'],[]).append(r)
    def jump_event(self,event):
        """Start any known event30s early, independent of the previous window."""
        with self.lock:
            event_id=str(event.get('id',event.get('event_id','')));mid=str(event.get('module_id',''))
            if not event_id or mid not in self.source.bounds:raise ValueError('후보의 설비와 식별자를 확인할 수 없습니다.')
            detected=pd.Timestamp(event.get('start')or event.get('start_time')or event.get('time')or event.get('peak_time'))
            event_end=pd.Timestamp(event.get('end')or event.get('end_time')or event.get('time')or detected)
            if detected.tzinfo is not None or event_end.tzinfo is not None or pd.isna(detected)or pd.isna(event_end):raise ValueError('후보는 원본의 기록 시각을 사용해야 합니다.')
            detected=detected.floor('5s');event_end=max(event_end.floor('5s'),detected)
            first=max(pd.to_datetime(b['first'],format='%Y%m%d%H%M%S')for b in self.source.bounds.values())
            last=min(pd.to_datetime(b['last'],format='%Y%m%d%H%M%S')for b in self.source.bounds.values())
            if not first<=detected<=last:raise ValueError('후보 감지 시각이 보유 원본 기간 밖에 있습니다.')
            start=max(first,detected-pd.Timedelta(seconds=30));end=min(last+pd.Timedelta(seconds=5),max(start+pd.Timedelta(minutes=10),event_end+pd.Timedelta(minutes=2)))
            kind=event.get('source_kind','recorded');parent=None
            if kind=='synthetic':
                parent=next((s for s in self.scenarios if s['id']==event.get('scenario_id')),None)
                if not parent:raise ValueError('증강 시연의 원래 변형 조건을 찾을 수 없습니다.')
            elif kind!='recorded':raise ValueError('알 수 없는 후보 원천입니다.')
            event_info={'id':event_id,'module_id':mid,'title':event.get('title',TYPES.get(event.get('type'),'이상 후보')),'detected_at':iso(detected),'start':iso(detected),'end':iso(event_end+pd.Timedelta(seconds=5)),'source_kind':kind,'requested_lead_in_seconds':30,'actual_lead_in_seconds':int((detected-start).total_seconds())}
            highlight={**event_info,'event_id':event_id,'type':event.get('type','abnormal_pattern')}
            scenario={**(parent or{}),'id':'event_'+event_id,'name':'후보 재생 · '+str(event.get('equipment',mid)),'start':iso(start),'end':iso(end),'source_kind':kind,'module_id':mid,'fault_type':event.get('type','abnormal_pattern'),'highlight':highlight}
            if parent:
                scenario['augmentation_origin_start']=parent.get('augmentation_origin_start',parent['start'])
                scenario['alarm_origin_scenario_id']=parent.get('alarm_origin_scenario_id',parent['id'])
                # An AE alarm can differ from its synthetic cause. Retain the
                # original cause transform while highlighting the detected event.
                scenario['fault_type']=parent['fault_type']
            self.scenarios[:]=[s for s in self.scenarios if s['id']!=scenario['id']];self.scenarios.append(scenario)
            self.playing=False;self.set_scenario(scenario['id']);self.playing=True;self.broadcast()
            return {'replay':self.state(),'snapshot':self.snapshot,'history':list(self.history),'event':event_info}
    def recalculate_energy(self):
        day=self.cursor.floor('D');s=self.source.history_energy('all',self.cursor);self.day_energy=float(s.loc[s.index>=day].sum())
        partial=self.source.window(self.cursor.floor('h'),self.cursor)if self.cursor>self.cursor.floor('h')else[]
        self.day_energy+=sum(r['energy_kwh']for r in partial)
    def synthetic(self,rows):
        return augment_records(rows,self.scenario)
    def advance_streak(self,key,source_time):
        previous=self.streak_last.get(key)
        if previous is None or(pd.Timestamp(source_time)-pd.Timestamp(previous)).total_seconds()!=5:
            self.streaks[key]=1;self.streak_first[key]=source_time
        else:self.streaks[key]=self.streaks.get(key,0)+1
        self.streak_last[key]=source_time
    def persist_detection(self,event,key,received,new_detections):
        # Preserve DB insertion/Slack outbox policy even after a local toast was
        # emitted. DB duplicate insertion and current replay occurrence differ.
        event={**event,'first_time':self.streak_first[key],'detected_at':event['time']}
        inserted=DB.alarm(event)
        if self.streaks[key]!=3:return
        first=self.streak_first[key]
        identity=f'{self.session}:{key[0]}:{key[1]}:{first}'
        occurrence_id='occ_'+hashlib.sha256(identity.encode()).hexdigest()[:28]
        if occurrence_id in self.emitted_occurrences:return
        stored=DB.alarm_by_id(alarm_id(event))
        if stored is None:raise RuntimeError('Persisted detection record is unavailable')
        notification={'id':stored['id'],'event_id':stored['id'],'detail_event_id':occurrence_id,'occurrence_id':occurrence_id,
            'session_id':self.session,'module_id':str(event['module_id']),'equipment':event['equipment'],
            'type':event['type'],'title':event['title'],'first_time':first,'detected_at':event['time'],
            'received_at':received,'source_kind':event['source_kind'],'status':'unconfirmed_candidate',
            'alarm_status':stored['status'],'stored_detected_at':stored.get('detected_at',stored['time']),
            'new_persistent_alarm':inserted,'detail_url':'/api/diagnosis/'+occurrence_id}
        current={**stored,**event,**notification,'start':first,'start_time':first,'end':event['time'],
                 'end_time':event['time'],'time':event['time'],'algorithm':event.get('algorithm','규칙')}
        self.detection_occurrences[occurrence_id]=current
        while len(self.detection_occurrences)>256:self.detection_occurrences.popitem(last=False)
        self.emitted_occurrences.add(occurrence_id);self.session_detected_modules.add(str(event['module_id']));new_detections.append(notification)
    def detection_occurrence(self,event_id):
        with self.lock:
            event=self.detection_occurrences.get(str(event_id))
            return dict(event)if event is not None else None
    def receive(self,log=True):
        if not self.frame_start<=self.cursor<self.frame_end:self.load_frame_chunk()
        rows=[dict(r,source_kind=self.scenario['source_kind'])for r in self.frames.get(iso(self.cursor),[])];rows=self.synthetic(rows);received=now();diagnosed=[]
        alarm_scenario_id=self.scenario.get('alarm_origin_scenario_id',self.scenario['id'])
        if rows and self.detector_ready():
            try:diagnosed=self.detector.diagnose_batch(rows)
            except AttributeError:diagnosed=self.detector_module_batch(rows)
        bymodule={str(r['module_id']):r for r in diagnosed}
        present=set();new_detections=[]
        for r in rows:
            d=bymodule.get(r['module_id'],{});rules=d.get('rules',[]);ai=d.get('ai',{});r.update(source_kind=self.scenario['source_kind'],received_at=received,status=d.get('status','not_scored'),rule_score=max([float(a.get('score',0))for a in rules],default=0),ai_score=ai.get('ae_score',0),diagnosis=d)
            if log:
                for rule in rules:
                    if rule.get('triggered',True)and float(rule.get('score',0))>0:
                        kind=rule.get('type',rule.get('type_id','abnormal_pattern'))
                        sk=(r['module_id'],kind)
                        if sk in present:continue
                        present.add(sk);self.advance_streak(sk,r['time'])
                        if self.streaks[sk]>=3:self.persist_detection({'scenario_id':alarm_scenario_id,'module_id':r['module_id'],'equipment':r['equipment'],'time':r['time'],'type':kind,'title':TYPES.get(kind,kind),'severity':'warning','score':rule.get('score'),'source_kind':r['source_kind'],'evidence':rule.get('evidence',{}),'confirmation':'3연속 관측 의심 패턴 · 실제 고장 라벨 미제공','ai':ai},sk,received,new_detections)
                if not rules and(ai.get('ae_alarm')or ai.get('iforest_alarm')):
                    sk=(r['module_id'],'abnormal_pattern');present.add(sk);self.advance_streak(sk,r['time'])
                    if self.streaks[sk]>=3:self.persist_detection({'scenario_id':alarm_scenario_id,'module_id':r['module_id'],'equipment':r['equipment'],'time':r['time'],'type':'abnormal_pattern','title':'AI 이상 패턴 후보','severity':'warning','score':ai.get('ae_score'),'source_kind':r['source_kind'],'evidence':{'AE':ai.get('ae_score'),'AE_threshold':ai.get('ae_threshold'),'IF':ai.get('iforest_score'),'IF_threshold':ai.get('iforest_threshold')},'algorithm':'AE / Isolation Forest','confirmation':'3연속 관측 · 실제 고장 라벨 미제공','ai':ai},sk,received,new_detections)
        if log:
            for sk in list(self.streaks):
                if sk not in present:self.streaks[sk]=0;self.streak_first.pop(sk,None);self.streak_last.pop(sk,None)
        if log and rows:DB.log(rows,self.session,received);self.day_energy+=sum(r['energy_kwh']for r in rows)
        alarms=DB.alarms(limit=20)['rows'];power=sum(r['activePower']for r in rows)
        self.sequence+=1;self.history.extend(rows);self.history=self.history[-13*61:]
        count=DB.log_count();self.snapshot={'timestamp':iso(self.cursor),'received_at':received,'points':rows,'kpis':{'power_kw':power,'energy_today_kwh':self.day_energy,'bill_today_won':self.day_energy*180,'carbon_today_kg':self.day_energy*.424,'connected_modules':len(rows),'active_alerts':DB.active_alarm_count(),'logged_records':count,'scope':'13개 계측 모듈 합계 · 설비 계층 미확인','assumptions':'가이드 단위 및180원/kWh,0.424kgCO₂/kWh 가정'},'alarms':alarms,'log_count':count}
        self.broadcast('telemetry',new_detections)
    def detector_module_batch(self,rows):
        # The detector implementation exposes a module-level convenience function.
        path=APP/'analytics/diagnosis/detector.py';module=load_module('diagnosis_batch',path);return module.diagnose_batch(rows)
    def broadcast(self,type='state',new_detections=None):
        payload={'type':type,'sequence':self.sequence,'replay':self.state(),'snapshot':self.snapshot,
                 'new_detections':list(new_detections or[])if type=='telemetry'else[]}
        for q in list(self.listeners):
            try:q.put_nowait(payload)
            except Exception:
                try:q.get_nowait();q.put_nowait(payload)
                except Exception:pass
    def control(self,action,scenario_id=None,timestamp=None,speed=None):
        with self.lock:
            if speed is not None:
                if int(speed)not in[1,5,20,60]:raise ValueError('재생 속도는1,5,20,60 중 선택하세요.')
                self.speed=int(speed)
            if action=='play':
                if self.cursor>=self.end:self.set_scenario(self.scenario['id'])
                self.playing=True
            elif action=='pause':self.playing=False
            elif action=='scenario':self.playing=False;self.set_scenario(scenario_id)
            elif action=='seek':
                t=pd.Timestamp(timestamp)
                if not pd.Timestamp(self.scenario['start'])<=t<self.end:raise ValueError('현재 시연 구간 안의 시간을 선택하세요.')
                self.cursor=t.floor('5s');self.history=[];self.streaks={};self.streak_first={};self.streak_last={};self.recalculate_energy();self.receive(log=False)
            elif action!='speed':raise ValueError('지원하지 않는 재생 동작입니다.')
            self.broadcast();return {'replay':self.state(),'snapshot':self.snapshot,'history':self.history}
    def run(self):
        next_tick=time.monotonic()
        while not self.stopped:
            if self.playing and time.monotonic()>=next_tick:
                with self.lock:
                    try:
                        if self.cursor>=self.end:self.playing=False;self.broadcast()
                        else:self.receive();self.cursor+=pd.Timedelta(seconds=5)
                    except Exception:self.playing=False;self.broadcast()
                next_tick=time.monotonic()+5/self.speed
            time.sleep(.04)
