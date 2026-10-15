import React,{useCallback,useEffect,useRef,useState} from 'react';
import {Bell,ChevronRight,RefreshCw,TriangleAlert,X} from 'lucide-react';
import {api} from './api';
import {collectStreamDetections,equipmentLabel,faultLabel,mergeNotices,noticeKey,occurrenceTime} from './notifications.js';

const clock=value=>String(value||'기록 시각 확인 필요').replace('T',' ').replace('Z',' UTC');
function NoticeText({event}){
  return <><span className="stream-alert-title"><strong>{equipmentLabel(event)}</strong><span>{faultLabel(event)}</span></span><span className="stream-alert-meta"><span>발생</span> <time dateTime={occurrenceTime(event).replace(' ','T')}>{clock(occurrenceTime(event))}</time></span><span className="stream-alert-source">{event.source_kind==='synthetic'?'증강 시연':'원본 재생'} · 미확인 이상 후보</span></>;
}

export function useStreamNotifications(){
  const [notices,setNotices]=useState([]),[visible,setVisible]=useState([]),[open,setOpen]=useState(false),[selection,setSelection]=useState(null);
  const seen=useRef(new Set()),request=useRef(0);
  const hydrate=useCallback(events=>setNotices(old=>mergeNotices(old,events)),[]);
  const consume=useCallback(payload=>{
    const events=collectStreamDetections(payload,seen.current);
    if(!events.length)return;
    const session=payload.replay?.session_id||'';
    setNotices(old=>mergeNotices(old,events,{live:true,session}));
    setVisible(old=>[...events.slice().reverse().map(event=>noticeKey(event,session)),...old].slice(0,3));
  },[]);
  const clearVisible=useCallback(()=>setVisible([]),[]);
  const dismiss=useCallback(key=>setVisible(old=>old.filter(item=>item!==key)),[]);
  const readAll=useCallback(()=>setNotices(old=>old.map(event=>({...event,unread:false}))),[]);
  const closeDetail=useCallback(()=>{request.current+=1;setSelection(null);},[]);
  const inspect=useCallback(async event=>{
    const ticket=++request.current;
    setOpen(false);dismiss(event.notice_key);
    setNotices(old=>old.map(item=>item.notice_key===event.notice_key?{...item,unread:false}:item));
    setSelection({event,detail:null,loading:true,error:''});
    try{
      const detail=await api.diagnosisDetail(event.detail_event_id||event.event_id||event.id);
      if(request.current===ticket)setSelection({event,detail,loading:false,error:''});
    }catch(error){if(request.current===ticket)setSelection({event,detail:null,loading:false,error:error.message});}
  },[dismiss]);
  return {notices,popups:visible.map(key=>notices.find(event=>event.notice_key===key)).filter(Boolean),unread:notices.filter(event=>event.unread).length,open,setOpen,hydrate,consume,clearVisible,dismiss,readAll,inspect,selection,closeDetail};
}

export function StreamAlerts({events,onInspect,onDismiss}){
  return <div className="stream-alerts" role="region" aria-label="실시간 이상 알림" aria-live="polite" aria-relevant="additions">{events.map(event=><article className="stream-alert-card" data-stream-alert={event.notice_key} key={event.notice_key}><TriangleAlert size={19} aria-hidden="true"/><button className="stream-alert-content" onClick={()=>onInspect(event)}><NoticeText event={event}/><span className="notification-item-action">센서 변화·탐지 근거 확인 <ChevronRight size={13}/></span></button><button className="stream-alert-dismiss" aria-label={`${equipmentLabel(event)} 알림 닫기`} onClick={()=>onDismiss(event.notice_key)}><X size={17}/></button></article>)}</div>;
}

export function NotificationCenter({state,onViewAll}){
  const anchor=useRef(null);
  useEffect(()=>{
    if(!state.open)return;
    const outside=event=>{if(!anchor.current?.contains(event.target))state.setOpen(false);};
    const escape=event=>{if(event.key==='Escape'){state.setOpen(false);anchor.current?.querySelector('button')?.focus();}};
    document.addEventListener('pointerdown',outside);document.addEventListener('keydown',escape);
    return()=>{document.removeEventListener('pointerdown',outside);document.removeEventListener('keydown',escape);};
  },[state.open,state.setOpen]);
  return <div className="notification-anchor" ref={anchor}><button className="icon-button notification" aria-label={`알림 센터 · 미읽음 ${state.unread}건`} aria-expanded={state.open} aria-controls="stream-notification-center" onClick={()=>state.setOpen(value=>!value)}><Bell size={19}/>{state.unread>0&&<span className="notification-count">{state.unread>99?'99+':state.unread}</span>}</button>{state.open&&<section className="notification-popover" id="stream-notification-center" aria-label="이상 알림 센터"><div className="notification-popover-header"><div><strong>이상 알림</strong><span>미읽음 {state.unread}건 · 최근 {state.notices.length}건</span></div>{state.unread>0&&<button className="text-button" onClick={state.readAll}>모두 읽음 표시</button>}<button className="icon-button" aria-label="알림 센터 닫기" onClick={()=>state.setOpen(false)}><X size={17}/></button></div><div className="notification-list">{state.notices.length?state.notices.map(event=><button key={event.notice_key} className={`notification-item ${event.unread?'unread':''}`} onClick={()=>state.inspect(event)}><NoticeText event={event}/><span className="notification-item-action">센서 변화·탐지 근거 <ChevronRight size={13}/></span></button>):<p className="notification-empty">새로 감지된 이상이 없습니다.<br/>스트리밍 중 감지한 알림을 이곳에 보관합니다.</p>}</div><button className="text-button" onClick={()=>{state.setOpen(false);onViewAll();}}>고장진단 전체 보기 <ChevronRight size={14}/></button></section>}</div>;
}

export function NotificationDetailDialog({selection,onClose,onRetry,onReplay,renderDetail}){
  const dialog=useRef(null);
  useEffect(()=>{
    const previous=document.activeElement,overflow=document.body.style.overflow;
    document.body.style.overflow='hidden';dialog.current?.focus();
    const keydown=event=>{
      if(event.key==='Escape'){event.preventDefault();onClose();return;}
      if(event.key!=='Tab')return;
      const controls=[...dialog.current.querySelectorAll('button,select,input,a[href],[tabindex]:not([tabindex="-1"])')].filter(element=>!element.disabled&&element.getClientRects().length);
      if(!controls.length){event.preventDefault();dialog.current.focus();return;}
      const first=controls[0],last=controls.at(-1);
      if(event.shiftKey&&(document.activeElement===first||document.activeElement===dialog.current)){event.preventDefault();last.focus();}
      else if(!event.shiftKey&&(document.activeElement===last||document.activeElement===dialog.current)){event.preventDefault();first.focus();}
    };
    document.addEventListener('keydown',keydown);
    return()=>{document.body.style.overflow=overflow;document.removeEventListener('keydown',keydown);if(previous?.isConnected&&previous!==document.body)previous.focus();else document.querySelector('.notification-anchor>button')?.focus();};
  },[onClose]);
  const event=selection.event;
  return <div className="notification-dialog-layer"><button className="notification-dialog-backdrop" tabIndex={-1} aria-label="알림 상세 배경 닫기" onClick={onClose}/><div className="notification-dialog" role="dialog" aria-modal="true" aria-labelledby="notification-detail-title" tabIndex={-1} ref={dialog}><div className="notification-detail-header"><div><span className="stream-alert-source">{event.source_kind==='synthetic'?'증강 시연':'원본 재생'} · 미확인 이상 후보</span><h2 id="notification-detail-title">{equipmentLabel(event)} · {faultLabel(event)}</h2><p className="notification-event-clocks"><span>발생 <time>{clock(occurrenceTime(event))}</time></span>{event.detected_at&&event.detected_at!==occurrenceTime(event)&&<span>감지 <time>{clock(event.detected_at)}</time></span>}</p></div><button className="icon-button" aria-label="알림 상세 닫기" onClick={onClose}><X size={22}/></button></div>{selection.loading?<div className="notification-detail-loading" role="status"><RefreshCw className="spin" size={22}/><strong>센서 변화와 탐지 근거를 불러오고 있습니다.</strong><span>발생 구간 주변의 실제 관측을 조회합니다.</span></div>:selection.error?<div className="notification-detail-error" role="alert"><TriangleAlert size={23}/><p>{selection.error}</p><button className="button secondary" onClick={()=>onRetry(event)}>다시 불러오기</button></div>:renderDetail(selection.detail,onClose,onReplay)}</div></div>;
}
