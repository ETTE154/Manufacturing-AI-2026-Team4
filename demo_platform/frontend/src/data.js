export function recordedClock(value){
  const text=String(value??'');
  if(/^\d{14}$/.test(text))return `${text.slice(0,4)}-${text.slice(4,6)}-${text.slice(6,8)} ${text.slice(8,10)}:${text.slice(10,12)}:${text.slice(12,14)}`;
  return text.replace('T',' ');
}
export function normalizeRow(raw){
  const r={...raw,...(raw.values||raw.sensors||{})};
  const identity=String(r.module_id??r.module??r['module(equipment)']??r.equipment??'');
  const recorded_at=recordedClock(r.time||r.recorded_at||r.localtime||r.hour||r.timestamp);
  return {...r,module:identity.split('(')[0],recorded_at,chart_time:recordedMillis(recorded_at)};
}
export function appendRows(old,incoming){
  const unique=new Map();
  for(const row of [...old,...incoming])unique.set(`${row.module}|${row.recorded_at}`,row);
  return [...unique.values()].sort((a,b)=>recordedMillis(a.recorded_at)-recordedMillis(b.recorded_at)||String(a.module).localeCompare(String(b.module))).slice(-3900);
}

export const SAMPLE_INTERVAL_SECONDS=5;
export const REALTIME_INTERVALS=60;

// The source is a naive recorded clock. Use UTC only as a numeric axis encoding;
// never reinterpret these labels in the browser's local time zone.
export function recordedMillis(value){
  if(typeof value==='number')return Number.isFinite(value)?value:NaN;
  const text=recordedClock(value),parts=text.match(/^(\d{4})-(\d{2})-(\d{2})[ ](\d{2}):(\d{2})(?::(\d{2}))?/);
  return parts?Date.UTC(Number(parts[1]),Number(parts[2])-1,Number(parts[3]),Number(parts[4]),Number(parts[5]),Number(parts[6]||0)):NaN;
}
export function recordedFromMillis(value){
  return Number.isFinite(Number(value))?new Date(Number(value)).toISOString().slice(0,19).replace('T',' '):'';
}
export function realtimeWindow(rows,latest,intervalSeconds=SAMPLE_INTERVAL_SECONDS){
  const points=rows.map(r=>({...r,chart_time:recordedMillis(r.recorded_at)})).filter(r=>Number.isFinite(r.chart_time));
  const actualLatest=recordedMillis(latest),right=Number.isFinite(actualLatest)?actualLatest:points.at(-1)?.chart_time;
  if(!Number.isFinite(right))return {rows:[],domain:null,ticks:[]};
  const seconds=Number(intervalSeconds)>0?Number(intervalSeconds):SAMPLE_INTERVAL_SECONDS;
  const left=right-REALTIME_INTERVALS*seconds*1000;
  return {rows:points.filter(r=>r.chart_time>=left&&r.chart_time<=right),domain:[left,right],ticks:Array.from({length:6},(_,i)=>left+(right-left)*i/5)};
}
export function rowIsCandidate(row){
  const diagnosis=row.diagnosis||{},ai=diagnosis.ai||row.ai||{};
  return !!(row._anomaly||row.status==='unconfirmed_candidate'||diagnosis.status==='unconfirmed_candidate'||ai.ae_alarm||ai.iforest_alarm||
    (diagnosis.rules||row.rules||[]).some(r=>r.triggered!==false&&(r.score==null||r.threshold==null||Number(r.score)>Number(r.threshold))));
}
export function eventHighlight(event,{inclusiveEnd=false}={}){
  if(!event)return null;
  const start=event.start||event.start_time||event.detected_at||event.time||event.peak_time;
  const end=event.end||event.end_time;
  const startMs=recordedMillis(start),endMs=recordedMillis(end);
  if(!Number.isFinite(startMs))return null;
  const finish=Number.isFinite(endMs)&&endMs>=startMs?endMs+(inclusiveEnd?SAMPLE_INTERVAL_SECONDS*1000:0):startMs+SAMPLE_INTERVAL_SECONDS*1000;
  return {start:recordedFromMillis(startMs),end:recordedFromMillis(finish>startMs?finish:startMs+SAMPLE_INTERVAL_SECONDS*1000),module_id:String(event.module_id||event.module||'all'),event_id:event.event_id||event.id};
}
export function inHighlight(row,highlight){
  if(!highlight)return false;
  const module=String(row.module??row.module_id??'all');
  if(module!=='all'&&String(highlight.module_id)!=='all'&&module!==String(highlight.module_id))return false;
  const point=recordedMillis(row.recorded_at),start=recordedMillis(highlight.start),end=recordedMillis(highlight.end);
  return Number.isFinite(point)&&point>=start&&point<end;
}
export function anomalyRanges(rows,predicate,intervalSeconds=SAMPLE_INTERVAL_SECONDS){
  const ranges=[];let current=null;const step=intervalSeconds*1000;
  for(const row of rows){
    const point=recordedMillis(row.recorded_at);
    if(!Number.isFinite(point)||!predicate(row)){current=null;continue;}
    if(current&&point-current.last<=step*1.5){current.end=point+step;current.last=point;}
    else{current={start:point,end:point+step,last:point};ranges.push(current);}
  }
  return ranges;
}
