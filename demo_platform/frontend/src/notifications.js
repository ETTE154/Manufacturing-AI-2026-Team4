import {equipmentTitle} from './uiScope.js';
const FAULT_LABELS={load_imbalance:'부하 불균형',reactive_power_surge:'무효전력 급증',power_factor_drop:'역률 급감',abnormal_pattern:'비정상 패턴',night_operation:'야간 이상가동',current_only_increase:'전류만 증가'};

export function faultLabel(event){return FAULT_LABELS[event.type]||event.title||'이상 패턴';}
export function equipmentLabel(event){
  const name=String(event.equipment||event.equipment_name||'');
  return equipmentTitle(event.module_id||name.match(/^(\d+)\(/)?.[1],name);
}
export function occurrenceTime(event){return event.first_time||event.start||event.start_time||event.time||event.detected_at||'';}
export function noticeKey(event,session=''){
  return event.occurrence_id||`${event.session_id||session}:${event.event_id||event.id}:${occurrenceTime(event)}`;
}
export function normalizeNotice(event,{live=false,session=''}={}){
  const id=event.event_id||event.id;
  if(!id)return null;
  return {...event,id,event_id:id,notice_key:live?noticeKey(event,session):`stored:${id}`,is_live:live,unread:live};
}
export function mergeNotices(previous,incoming,{live=false,session=''}={}){
  const additions=incoming.map(event=>normalizeNotice(event,{live,session})).filter(Boolean);
  if(live){
    const keys=new Set(additions.map(event=>event.notice_key)),ids=new Set(additions.map(event=>event.event_id));
    return [...additions.slice().reverse(),...previous.filter(event=>!keys.has(event.notice_key)&&!(!event.is_live&&ids.has(event.event_id)))].slice(0,50);
  }
  const keys=new Set(previous.map(event=>event.notice_key)),liveIds=new Set(previous.filter(event=>event.is_live).map(event=>event.event_id));
  return [...previous,...additions.filter(event=>!keys.has(event.notice_key)&&!liveIds.has(event.event_id))].slice(0,50);
}
export function collectStreamDetections(payload,seen){
  if(payload.type!=='telemetry'||!Array.isArray(payload.new_detections))return [];
  const session=payload.replay?.session_id||payload.session_id||'';
  const fresh=[];
  for(const event of payload.new_detections){
    if(!(event.event_id||event.id)||!event.module_id||!occurrenceTime(event))continue;
    const key=noticeKey(event,session);
    if(seen.has(key))continue;
    seen.add(key);fresh.push(event);
  }
  while(seen.size>1000)seen.delete(seen.values().next().value);
  return fresh;
}
