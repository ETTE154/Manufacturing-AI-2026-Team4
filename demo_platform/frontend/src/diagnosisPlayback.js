import {normalizeRow,recordedClock,recordedMillis,eventHighlight} from './data.js';

export function replayEventId(event){
  return event?.detail_event_id||event?.occurrence_id||event?.event_id||event?.id||'';
}
export function contextError(before,after){
  return [before,after].some(v=>String(v).trim()===''||!Number.isInteger(Number(v))||Number(v)<0||Number(v)>1800||Number(v)%5!==0)
    ?'전·후 범위는 각각 0~1,800초 사이의 5초 단위로 입력하세요.':'';
}
export function advancePlayback(state,count){
  if(count<1)return {index:0,cycle:1};
  return state.index>=count-1?{index:0,cycle:state.cycle+1}:{...state,index:state.index+1};
}
export function initialSignal(type){
  return type==='power_factor_drop'?'powerFactor':['load_imbalance','current_only_increase'].includes(type)?'current':type==='reactive_power_surge'?'reactivePowerLagging':'activePower';
}
export function replayTrace(detail){
  const algorithms=new Map((detail?.algorithm||[]).map(r=>[recordedClock(r.time),r]));
  const ai=new Map((detail?.ai||[]).map(r=>[recordedClock(r.time),r]));
  const rows=(detail?.series||[]).map(normalizeRow).filter(r=>Number.isFinite(r.chart_time)).sort((a,b)=>a.chart_time-b.chart_time)
    .map(r=>({...r,diagnosis:{...(algorithms.get(r.recorded_at)||{}),ai:ai.get(r.recorded_at)||{}}}));
  const rules=[...new Map((detail?.algorithm||[]).flatMap(r=>r.rules||[]).map(r=>[r.type,{id:r.type,label:r.label||r.type}])).values()];
  const start=recordedMillis(detail?.window?.start),end=recordedMillis(detail?.window?.end_inclusive);
  const domain=rows.length?[Number.isFinite(start)?start:rows[0].chart_time,Number.isFinite(end)?end:rows.at(-1).chart_time]:null;
  // Recharts needs a nonzero axis even for a one-point, zero-context event.
  if(domain&&domain[0]===domain[1])domain[1]+=5000;
  const event=detail?.event||{},window=detail?.window||{};
  const highlight=eventHighlight({...event,start:window.event_start||event.start||event.start_time,end:window.event_end||event.end||event.end_time},{inclusiveEnd:true});
  const ticks=domain?[...new Set(Array.from({length:5},(_,i)=>domain[0]+Math.round((domain[1]-domain[0])*i/4/5000)*5000))]:[];
  return {rows,rules,domain,ticks,highlight};
}
export function replayScores(rows,kind,rule){
  return rows.map(r=>{const q=(r.diagnosis?.rules||[]).find(s=>s.type===rule),a=r.diagnosis?.ai||{};return {
    recorded_at:r.recorded_at,score:kind==='rules'?q?.score:a[`${kind}_score`],threshold:kind==='rules'?q?.threshold:a[`${kind}_threshold`],
  };});
}

// Keep each sensor's native scale stable throughout a complete replay window.
// Missing measurements stay missing and are never used as artificial zeroes.
export function sensorValueDomain(rows,keys){
  let minimum=Infinity,maximum=-Infinity;
  for(const row of rows)for(const key of keys){
    const value=row[key];
    if(value==null||value===''||!Number.isFinite(Number(value)))continue;
    minimum=Math.min(minimum,Number(value));maximum=Math.max(maximum,Number(value));
  }
  if(!Number.isFinite(minimum))return null;
  const padding=(maximum-minimum||Math.abs(maximum)||1)*.06;
  return [minimum-padding,maximum+padding];
}
