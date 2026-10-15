export function makeSearchContext({page,module,cursorTime,appliedQuery,historyActive,reviewContext,gptConfigured}={}){
  const base={module:String(module??'all'),cursor_time:cursorTime,scope:'replay',source_kind:'recorded',gpt_configured:!!gptConfigured};
  if(page==='diagnosis'&&reviewContext?.start&&reviewContext?.end)return {...base,...reviewContext,module:String(reviewContext.module),scope:'diagnosis',grain:'raw',source_kind:'recorded',origin_source_kind:reviewContext.source_kind};
  if(page==='equipment'&&historyActive&&appliedQuery?.query_start&&appliedQuery?.query_end)return {...base,module:String(appliedQuery.module),scope:'archive',start:appliedQuery.query_start,end:appliedQuery.query_end,grain:appliedQuery.grain};
  return base;
}
export function searchPayload(question,context){
  context=context||{};
  const explicit=['archive','diagnosis'].includes(context.scope);
  return {question,local_only:explicit||!context.gpt_configured,context:{module_id:String(context.module??'all'),cursor_time:context.cursor_time,scope:context.scope||'replay',source_kind:'recorded',...(explicit?{start:context.start,end:context.end,grain:context.grain||'raw'}:{})}};
}
