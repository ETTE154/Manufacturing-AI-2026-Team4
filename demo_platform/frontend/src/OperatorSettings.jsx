import React,{useEffect,useState} from 'react';
import {ArrowLeft,Check,Settings2} from 'lucide-react';
import {UXTabs} from './UXViews.jsx';
import {contextError} from './diagnosisPlayback.js';
import {moduleTitle} from './uiScope.js';
import {api} from './api.js';

function ReportSchedule({data,module,modules,onUpdate,notify,error}){
  const [weekday,setWeekday]=useState('mon'),[clock,setClock]=useState('09:00'),[enabled,setEnabled]=useState(false),[saving,setSaving]=useState(false);
  useEffect(()=>{const schedule=data?.schedule;if(schedule){setWeekday(schedule.weekday||'mon');setClock(schedule.time||'09:00');setEnabled(!!schedule.enabled);}},[data?.schedule]);
  const save=async event=>{event.preventDefault();setSaving(true);try{await api.schedule({enabled,weekday,time:clock,module});onUpdate(await api.reports());notify('주간 리포트 일정을 저장했습니다.');}catch(e){error(e.message);}finally{setSaving(false);}};
  return <section className="op-settings-card"><h2>주간 리포트 자동 생성</h2><p>대상: {moduleTitle(modules,module)} · 한국 시각</p><form onSubmit={save}><label className="toggle-row"><span>자동 생성 사용</span><input type="checkbox" checked={enabled} onChange={e=>setEnabled(e.target.checked)}/></label><div className="op-settings-actions"><label className="field-label">요일<select value={weekday} onChange={e=>setWeekday(e.target.value)}>{[['mon','월요일'],['tue','화요일'],['wed','수요일'],['thu','목요일'],['fri','금요일'],['sat','토요일'],['sun','일요일']].map(([id,label])=><option key={id} value={id}>{label}</option>)}</select></label><label className="field-label">시각<input type="time" value={clock} required onChange={e=>setClock(e.target.value)}/></label></div><p className="op-section-note">저장된 일정 대상: {moduleTitle(modules,data?.schedule?.module_id||'all')} · 서버 실행 중 적용</p><button className="button secondary" disabled={saving}><Check size={15}/>{saving?'저장 중…':'일정 저장'}</button></form></section>;
}

export default function OperatorSettings({section,onSection,replayControls,padding,onPadding,hasReview,onReview,integrations,onConnections,reportData,module,modules,onReports,notify,error,ui,children}){
  const {Badge}=ui;
  const [before,setBefore]=useState(String(padding.before)),[after,setAfter]=useState(String(padding.after));
  useEffect(()=>{setBefore(String(padding.before));setAfter(String(padding.after));},[padding.before,padding.after]);
  const validation=contextError(before,after),dirty=Number(before)!==padding.before||Number(after)!==padding.after;
  const gpt=integrations?.openai||integrations?.gpt||{},slack=integrations?.slack||{};
  const apply=event=>{event.preventDefault();if(validation)return;onPadding({before:Number(before),after:Number(after)});if(hasReview)onReview();};
  return <div className="op-settings"><UXTabs label="설정 항목" value={section} onChange={onSection} items={[{id:'playback',label:'데이터·시연'},{id:'services',label:'알림·리포트'},{id:'models',label:'모델 검증'}]}/>
    {section==='playback'&&<><p className="op-section-note">기록 데이터를 사용하는 시연 설정입니다. 운영 화면에는 관측 결과만 표시합니다.</p><div className="op-settings-grid"><section className="op-settings-card"><h2>데이터 재생</h2><p>재생 여부, 배속과 시연 구간을 설정합니다.</p>{replayControls}</section><section className="op-settings-card"><h2>이상 구간 검토 범위</h2><p>선택한 이상 구간의 발생 전·후 데이터를 함께 재생합니다.</p><form onSubmit={apply} noValidate><div className="op-settings-actions"><label className="field-label">발생 전 (초)<input type="number" aria-label="발생 전 범위 (초)" min="0" max="1800" step="5" value={before} onChange={e=>setBefore(e.target.value)} aria-invalid={!!validation}/></label><label className="field-label">발생 후 (초)<input type="number" aria-label="발생 후 범위 (초)" min="0" max="1800" step="5" value={after} onChange={e=>setAfter(e.target.value)} aria-invalid={!!validation}/></label></div><div className="op-settings-actions">{[60,120,300].map(seconds=><button type="button" className="button secondary small" key={seconds} aria-pressed={Number(before)===seconds&&Number(after)===seconds} onClick={()=>{setBefore(String(seconds));setAfter(String(seconds));}}>전후 {seconds/60}분</button>)}</div><p className="op-section-note" role={validation?'alert':undefined}>{validation||`각 0~1,800초 · 5초 단위${dirty?' · 변경 대기':''}`}</p><button className="button primary" type="submit" disabled={!!validation}>{hasReview?'적용하고 진단 보기':'검토 범위 적용'}</button>{hasReview&&<button type="button" className="text-button" onClick={onReview}><ArrowLeft size={14}/>변경 없이 진단으로</button>}</form></section></div></>}
    {section==='services'&&<div className="op-settings-grid"><section className="op-settings-card"><h2>서비스 연결</h2><div className="op-service-status"><span>Slack 이상 알림</span><Badge tone={slack.configured?'green':'amber'}>{slack.configured?'연결 설정됨':'미연결'}</Badge></div><div className="op-service-status"><span>GPT 주간 요약</span><Badge tone={gpt.configured?'green':'amber'}>{gpt.configured?'키 설정됨':'로컬 요약 사용'}</Badge></div><p>연결 상태와 외부 전송 승인은 세부 설정에서 관리합니다.</p><button className="button secondary" onClick={onConnections}><Settings2 size={15}/>연결 세부 설정</button></section><ReportSchedule data={reportData} module={module} modules={modules} onUpdate={onReports} notify={notify} error={error}/></div>}
    {section==='models'&&<><p className="op-section-note">개발·검증용 고정 평가입니다. 운영 화면의 현재 전망과 평가 범위가 다릅니다.</p>{children}</>}
  </div>;
}
