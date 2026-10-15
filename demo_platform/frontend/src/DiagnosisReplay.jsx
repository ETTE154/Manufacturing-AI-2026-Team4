import React,{useEffect,useId,useMemo,useRef,useState} from 'react';
import {Pause,Play,RefreshCw,X} from 'lucide-react';
import {api} from './api.js';
import {recordedMillis} from './data.js';
import {equipmentLabel} from './notifications.js';
import {advancePlayback,replayEventId,replayScores,replayTrace,sensorValueDomain} from './diagnosisPlayback.js';

const SENSOR_COMPARISON=['voltage','current','powerFactor','activePower','reactivePowerLagging'];

export default function DiagnosisReplay({request,onClose,ui,onReviewContext,padding={before:60,after:60},onOpenSettings}){
  const {Badge,Empty,SensorChart,ScoreChart,EvidenceSummary,SIGNALS,fmt,time}=ui;
  const [detail,setDetail]=useState(null),[error,setError]=useState(''),[loading,setLoading]=useState(false);
  const [playing,setPlaying]=useState(false),[speed,setSpeed]=useState(5),[position,setPosition]=useState({index:0,cycle:1}),[rule,setRule]=useState('');
  const [modelsOpen,setModelsOpen]=useState(false);
  const ticket=useRef(0),loadedRequest=useRef(null),section=useRef(null),heading=useRef(null),evidence=useRef(null),models=useRef(null);
  const syncId=useId(),before=Number(padding?.before??60),after=Number(padding?.after??60),requestKey=`${request.request_id}:${before}:${after}`;
  const event=detail?.event||request.event,id=replayEventId(request.event),trace=useMemo(()=>replayTrace(detail),[detail]);
  const sensorDomains=useMemo(()=>Object.fromEntries(SENSOR_COMPARISON.map(key=>[key,sensorValueDomain(trace.rows,SIGNALS[key].keys)])),[trace.rows,SIGNALS]);
  const title=`${equipmentLabel(event)} · ${event.label||event.title||event.type||'이상 후보'}`;
  const load=async(b,a)=>{
    const current=++ticket.current;loadedRequest.current=null;
    evidence.current?.removeAttribute('open');models.current?.removeAttribute('open');setModelsOpen(false);
    setError('');setLoading(true);setPlaying(false);setDetail(null);setPosition({index:0,cycle:1});
    try{
      const result=await api.diagnosisReplay(id,{before_seconds:Number(b),after_seconds:Number(a)});
      if(current!==ticket.current)return false;
      loadedRequest.current=requestKey;
      setDetail(result);setRule(result.event?.type||'');setPlaying(!!result.series?.length);
      return true;
    }catch(e){if(current===ticket.current)setError(e.message);return false;}
    finally{if(current===ticket.current)setLoading(false);}
  };
  useEffect(()=>{
    load(before,after);
    const frame=requestAnimationFrame(()=>{section.current?.scrollIntoView({block:'start',behavior:'auto'});heading.current?.focus({preventScroll:true});});
    return()=>{ticket.current+=1;loadedRequest.current=null;cancelAnimationFrame(frame);};
  },[request.request_id,before,after]);
  useEffect(()=>{
    if(!playing||!trace.rows.length)return;
    const timer=setInterval(()=>setPosition(old=>advancePlayback(old,trace.rows.length)),5000/speed);
    return()=>clearInterval(timer);
  },[playing,speed,trace.rows.length]);
  useEffect(()=>{if(trace.rules.length&&!trace.rules.some(r=>r.id===rule))setRule(trace.rules[0].id);},[trace.rules,rule]);
  // Publish the loaded immutable review window, never the advancing cursor.
  // The request guard prevents a previous event's detail leaking into a new selection.
  useEffect(()=>{
    if(typeof onReviewContext!=='function')return;
    if(!detail||loadedRequest.current!==requestKey){onReviewContext(null);return;}
    const reviewed=detail.event||request.event,window=detail.window||{};
    onReviewContext({module:String(reviewed.module_id??''),source_kind:detail.source_kind||reviewed.source_kind||'recorded',start:window.start,end:window.end_exclusive,event_id:replayEventId(reviewed)||id,title:`${equipmentLabel(reviewed)} · ${reviewed.label||reviewed.title||reviewed.type||'이상 후보'}`});
    return()=>onReviewContext(null);
  },[detail,requestKey,onReviewContext]);
  const rows=trace.rows.slice(0,position.index+1),current=rows.at(-1),w=detail?.window||{};
  const actualRule=current?.diagnosis?.rules?.find(r=>r.type===rule),ruleLabel=trace.rules.find(r=>r.id===rule)?.label||'규칙';
  const savedScores=[['ae','Autoencoder'],['iforest','Isolation Forest']].filter(([key])=>event.ai?.[`${key}_score`]!=null&&event.ai?.[`${key}_threshold`]!=null&&Number.isFinite(Number(event.ai[`${key}_score`]))&&Number.isFinite(Number(event.ai[`${key}_threshold`])));
  const atCandidate=current&&trace.highlight&&current.chart_time>=recordedMillis(trace.highlight.start)&&current.chart_time<recordedMillis(trace.highlight.end);
  const synthetic=detail?.source_kind==='synthetic'||event.source_kind==='synthetic';
  const exceededAI=savedScores.filter(([key])=>Number(event.ai[`${key}_score`])>Number(event.ai[`${key}_threshold`])).map(([key])=>key==='ae'?'AE':'IF');
  const evidenceLabel=exceededAI.length?`${exceededAI.join('·')} 임계값 초과`:savedScores.length?'저장 AI 점수 임계값 이내':'센서·규칙 근거';
  const eventStart=w.event_start||event.start||event.start_time||event.first_time||event.time;
  const eventEnd=w.event_end||event.end||event.end_time||event.time||eventStart;
  return <section ref={section} className="diagnosis-replay ux-replay-shell" aria-labelledby="diagnosis-replay-title"
    data-event-id={id} data-local-cursor={current?.recorded_at||''} data-loop-count={position.cycle} data-playing={playing?'true':'false'}
    data-window-start={w.start||''} data-window-end={w.end_inclusive||''} data-point-count={rows.length} data-total-points={trace.rows.length}>
    <div className="diagnosis-replay-heading"><div><h2 id="diagnosis-replay-title" ref={heading} tabIndex={-1}>{title}</h2>
      <p className="diagnosis-replay-occurrence">발생 {eventStart?time(eventStart,true):'시각 확인 중'}{eventEnd&&eventEnd!==eventStart?` ~ ${time(eventEnd,true)}`:''} · 이상 후보 · 실제 고장 미확인{synthetic?' · 합성 시연':''}</p></div>
      <button className="icon-button" onClick={onClose} aria-label="구간 반복 재생 닫기"><X size={19}/></button></div>
    <div className="diagnosis-review-toolbar" aria-label="진단 구간 재생 조작"><div className="diagnosis-replay-controls"><button className="button primary" disabled={loading||!trace.rows.length} aria-label={playing?'구간 일시정지':'구간 재생'} onClick={()=>setPlaying(v=>!v)}>{playing?<Pause size={16}/>:<Play size={16}/>} {playing?'일시정지':'재생'}</button>
      <button className="button secondary" disabled={loading||!trace.rows.length} aria-label="처음부터 재생" onClick={()=>{setPosition({index:0,cycle:1});setPlaying(true);}}><RefreshCw size={15}/>처음</button>
      <label><span>배속</span><select aria-label="구간 재생 속도" value={speed} onChange={e=>setSpeed(Number(e.target.value))}>{[1,5,20,60].map(s=><option key={s} value={s}>{s}×</option>)}</select></label>
      <Badge tone={playing?'teal':'neutral'}>{loading||!trace.rows.length?'대기':playing?'재생 중':'정지'} · {position.cycle}회차</Badge>
      <button className="text-button diagnosis-review-settings-link" disabled={typeof onOpenSettings!=='function'} onClick={onOpenSettings} title={detail?`적용 범위 ${time(w.start,true)} ~ ${time(w.end_inclusive,true)}`:undefined}>전 {before}초 / 후 {after}초 · 검토범위 설정</button>{w.clipped&&<small>원본 경계에 맞춰 조정</small>}</div>
      <div className="diagnosis-replay-progress"><div><span>검토 시각 <time>{time(current?.recorded_at,true)}</time></span><small>{current?position.index+1:0} / {trace.rows.length} · {position.cycle}회차{atCandidate?' · 후보 시각':''}</small></div>
        <input type="range" aria-label="진단 구간 재생 위치" min="0" max={Math.max(0,trace.rows.length-1)} step="1" value={position.index} disabled={loading||!trace.rows.length} aria-valuetext={current?`${time(current.recorded_at,true)} · ${position.index+1} / ${trace.rows.length} 관측`:'관측 대기'} onChange={e=>{setPlaying(false);setPosition(old=>({...old,index:Number(e.target.value)}));}}/></div></div>
    {error&&<p className="error-banner" role="alert">{error}</p>}
    {loading?<Empty title="선택 구간의 센서를 준비하고 있습니다" text=""/>:!trace.rows.length?<Empty title="재생할 관측 데이터가 없습니다" text="검토범위 설정에서 범위를 확인하세요."/>:<>
      <section className="diagnosis-replay-comparison" aria-labelledby="diagnosis-sensor-comparison-title"><div className="diagnosis-replay-comparison-heading"><div><h3 id="diagnosis-sensor-comparison-title">5종 센서 동시 비교</h3><p>빨강: 이상 후보 · 점선: 검토 시각</p></div></div>
        <div className="diagnosis-replay-sensor-grid">{SENSOR_COMPARISON.map(key=>{const cfg=SIGNALS[key];return <article key={key} className="diagnosis-replay-sensor-card" data-signal={key} data-local-cursor={current?.recorded_at||''} data-point-count={rows.length}>
          <header><div><h4>{key==='reactivePowerLagging'?'무효전력 (지상)':cfg.label}</h4><p>{cfg.unit}</p></div><time>{time(current?.recorded_at)}</time></header>
          <SensorChart rows={rows} signal={key} domain={trace.domain} ticks={trace.ticks} highlight={trace.highlight} cursorTime={current?.chart_time} valueDomain={sensorDomains[key]} syncId={syncId} height={190}/>
          <div className="diagnosis-replay-sensor-value">{cfg.keys.map(sensor=><span key={sensor}><small>{cfg.keys.length>1?`${sensor.slice(-1)}상`:'현재'}</small><strong>{fmt(current?.[sensor],2)}</strong></span>)}</div>
        </article>})}</div></section>
      <details className="ux-collection-details diagnosis-stored-evidence" ref={evidence}><summary>탐지 근거 · {evidenceLabel}</summary>
        {savedScores.length?<div className="ux-replay-evidence-grid">{savedScores.map(([key,label])=>{const score=Number(event.ai[`${key}_score`]),threshold=Number(event.ai[`${key}_threshold`]),exceeded=score>threshold;return <div className="ux-replay-evidence-item" key={key}><div><strong>{label}</strong><Badge tone={exceeded?'rose':'muted'}>{exceeded?'임계값 초과':'임계값 이내'}</Badge></div><p>저장 점수 <b>{fmt(score,5)}</b> / 임계값 <b>{fmt(threshold,5)}</b></p></div>;})}</div>:null}
        <EvidenceSummary event={event}/><p className="cell-note">조회 {time(w.start,true)} ~ {time(w.end_inclusive,true)} · {trace.rows.length}개 관측 · 실제 전 {w.actual_before_seconds}초 / 후 {w.actual_after_seconds}초</p>
      </details>
      <details className="ux-collection-details diagnosis-model-details" ref={models} onToggle={e=>setModelsOpen(e.currentTarget.open)}><summary>상세 모델 점수 · 3종</summary>{modelsOpen&&<>
        <div className="diagnosis-replay-evidence"><strong>현재 검토 시각의 규칙 점수</strong><label>비교할 규칙 <select aria-label="구간 재생 규칙 선택" value={rule} onChange={e=>setRule(e.target.value)}>{trace.rules.map(r=><option key={r.id} value={r.id}>{r.label}</option>)}</select></label>
          <p>{time(current?.recorded_at)} · {ruleLabel} 점수 {fmt(actualRule?.score,4)} / 임계값 {fmt(actualRule?.threshold,4)} {actualRule?.score!=null&&actualRule?.threshold!=null?(Number(actualRule.score)>Number(actualRule.threshold)?'· 임계값 초과':'· 임계값 이내'):''}</p><p>점수와 임계값은 모델마다 다릅니다.</p></div>
        <div className="diagnosis-replay-scores"><ScoreChart title="맥락 규칙 점수" note={ruleLabel} color="#2563eb" rows={replayScores(rows,'rules',rule)} domain={trace.domain} ticks={trace.ticks} highlight={trace.highlight} cursorTime={current?.chart_time}/><ScoreChart title="Autoencoder 재구성 오차" note="센서 관계의 재구성 오차" rows={replayScores(rows,'ae',rule)} domain={trace.domain} ticks={trace.ticks} highlight={trace.highlight} cursorTime={current?.chart_time}/><ScoreChart title="Isolation Forest 점수" note="희소한 센서 조합 점수" color="#0284c7" rows={replayScores(rows,'iforest',rule)} domain={trace.domain} ticks={trace.ticks} highlight={trace.highlight} cursorTime={current?.chart_time}/></div>
      </>}</details>
    </>}
  </section>;
}
