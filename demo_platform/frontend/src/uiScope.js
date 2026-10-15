export const GRAIN_LABELS={raw:'5초 원본',hour:'시간 평균',day:'일 평균',week:'주 평균',month:'월 평균'};
export function availabilityFor(metadata,module,grain){return metadata?.series_availability?.by_module?.[module]?.[grain]||null;}
export function sameQuery(applied,draft){return !!applied&&['module','signal','grain','start','end'].every(key=>String(applied[key]??'')===String(draft[key]??''));}
export function clampDate(value,min,max){return value&&value>=min&&value<=max?value:value&&value>max?max:min;}
export function receivedClock(value){
  if(!value)return '수신 대기';
  const date=new Date(value);
  return Number.isFinite(date.getTime())?new Intl.DateTimeFormat('en-GB',{timeZone:'Asia/Seoul',hour:'2-digit',minute:'2-digit',second:'2-digit',hourCycle:'h23'}).format(date):'수신 시각 확인 필요';
}
export function equipmentTitle(id,label=''){
  id=String(id||'');if(id==='all')return '전체 계측 모듈';
  const name=String(label).match(/^\d+\((.*)\)$/)?.[1]||String(label);
  if(!name||name===id||name===`모듈 ${id}`)return `모듈 ${id||'확인 필요'}`;
  return name.includes(`모듈 ${id}`)?name:`${name} · 모듈 ${id}`;
}
export function moduleTitle(modules,id){id=String(id);return id==='all'?'전체 계측 모듈':equipmentTitle(id,modules.find(module=>String(module.id)===id)?.label);}
