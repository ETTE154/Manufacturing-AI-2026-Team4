import {searchPayload} from './uxScope.js';
const BASE = import.meta.env.VITE_API_BASE || '/api';
export async function request(path, options = {}) {
  const response = await fetch(`${BASE}${path}`, {headers: {'Content-Type':'application/json', ...options.headers}, ...options});
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || body.message || `요청 실패 (${response.status})`);
  return body;
}
export const api = {
  metadata: async () => {const b=await request('/bootstrap');return {...b,...b.meta,cases:b.scenarios};},
  logs: () => request('/logs?limit=60'),
  control: (action, payload={}) => {const {case_id,...fields}=payload;return request('/replay', {method:'POST', body:JSON.stringify({action:action==='jump'?'scenario':action,...fields,...(case_id?{scenario_id:case_id}:{})})});},
  sensors: async (query) => {
    const signal=query.signal||'activePower';
    const keys=({voltage:['voltageR','voltageS','voltageT'],current:['currentR','currentS','currentT'],powerFactor:['powerFactorR','powerFactorS','powerFactorT']})[signal]||[signal];
    const end=query.end?new Date(Date.parse(query.end+'T00:00:00Z')+86400000).toISOString().slice(0,10)+' 00:00:00':null;
    const params={module_id:query.module,grain:query.grain||'raw',range_mode:'calendar',...(query.start?{start:query.start+' 00:00:00'}:{}),...(end?{end}:{})};
    const responses=await Promise.all(keys.map(sensor=>request(`/series?${new URLSearchParams({...params,sensor})}`)));
    const rows=new Map();responses.forEach((d,i)=>(d.series||[]).forEach(r=>{if(!rows.has(r.timestamp))rows.set(r.timestamp,{time:r.timestamp,module_id:query.module,records:r.records,complete:r.complete,minimum:r.minimum,maximum:r.maximum});rows.get(r.timestamp)[keys[i]]=r.value;}));
    return {...responses[0],rows:[...rows.values()],coverage_note:responses[0]?.coverage_note,grain:responses[0]?.grain};
  },
  forecast: async (query) => {
    const results=await Promise.allSettled([request(`/forecast?${new URLSearchParams({module_id:query.module,horizon:24})}`),request('/forecast/evaluation')]);
    const online=results[0].status==='fulfilled'?results[0].value:null,evaluation=results[1].status==='fulfilled'?results[1].value:null;if(!online&&!evaluation)throw results[0].reason;
    const models=(evaluation?.models||[]).map(m=>({...m,metrics:m.metrics||m,predictions:(evaluation.series||[]).map(r=>({...r,time:r.timestamp,actual:r.actual_kwh,predicted:r[m.id]??r[`${m.id}_predicted_kwh`]??r[`${m.id}_prediction`]??r.predictions?.[m.id]}))}));
    return {...evaluation,models,online,online_error:results[0].status==='rejected'?results[0].reason.message:null,origin:online?.origin,total_energy_kwh:online?.forecast?.length?online.forecast.reduce((sum,r)=>sum+Number(r.predicted_kwh),0):undefined,period_label:'현재 재생 원점의 향후 24시간 예측',quality_note:evaluation?.quality_note||online?.quality_note,notes:evaluation?.quality_note||online?.quality_note};
  },
  diagnosis: async () => {const [d,a]=await Promise.all([request('/diagnosis'),request('/alarms?limit=100')]);return {...d,alarms:a.rows||[],alarm_total:a.total};},
  diagnosisDetail: (id) => request(`/diagnosis/${encodeURIComponent(id)}`),
  diagnosisReplay: (id,range) => request(`/diagnosis/${encodeURIComponent(id)}/replay-window?${new URLSearchParams(range)}`),
  acknowledge: (id) => request(`/alarms/${encodeURIComponent(id)}/ack`, {method:'POST',body:'{}'}),
  integrations: () => request('/integrations'),
  slack: (payload) => request('/integrations/slack', {method:'POST',body:JSON.stringify(payload)}),
  slackTest: () => request('/integrations/slack/test', {method:'POST',body:'{}'}),
  openaiConsent: (approved) => request('/integrations/openai/consent', {method:'POST',body:JSON.stringify({approved})}),
  reports: async () => {const r=await request('/reports');return {...r,rows:(r.rows||[]).map(normalizeReport),schedule:r.schedule?{...r.schedule,weekday:['mon','tue','wed','thu','fri','sat','sun'][r.schedule.weekday],time:r.schedule.time||String(r.schedule.hour??9).padStart(2,'0')+':'+String(r.schedule.minute??0).padStart(2,'0')}:null};},
  report: async (payload) => normalizeReport(await request('/reports/generate', {method:'POST',body:JSON.stringify({module_id:payload.module,local_only:!!payload.local_only,...(payload.end?{end:new Date(Date.parse(payload.end+'T00:00:00Z')+86400000).toISOString().slice(0,10)+' 00:00:00'}:{})})})),
  schedule: (payload) => request('/reports/schedule', {method:'POST',body:JSON.stringify({enabled:payload.enabled,weekday:['mon','tue','wed','thu','fri','sat','sun'].indexOf(payload.weekday),hour:Number(payload.time.split(':')[0]),time:payload.time,module_id:payload.module})}),
  search: (question, context) => request('/chat', {method:'POST',body:JSON.stringify(searchPayload(question,context))}),
};
function normalizeReport(r){return {...r,created_at:r.generated_at||r.created_at,content:r.content||[r.title,r.summary,...(Array.isArray(r.sections)?r.sections.map(s=>typeof s==='string'?s:[s.title||s.heading,s.content||s.body||s.text].filter(Boolean).join('\n')):[]),...(r.actions||[]).map(s=>'점검 제안: '+s),...(r.limitations||[]).map(s=>'해석 범위: '+s),r.error].filter(Boolean).join('\n\n')};}
export function openStream(onEvent, onState) {
  const stream = new EventSource(`${BASE}/stream`);
  stream.onopen = () => onState('connected');
  stream.onerror = () => onState('reconnecting');
  stream.onmessage = (event) => { try {onEvent(JSON.parse(event.data));} catch {} };
  for (const type of ['observation','snapshot','replay','alarm','status']) stream.addEventListener(type, (event) => {try {onEvent({...JSON.parse(event.data),event_type:type});} catch {}});
  return () => stream.close();
}
