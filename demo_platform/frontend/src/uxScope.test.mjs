import test from 'node:test';
import assert from 'node:assert/strict';
import {makeSearchContext,searchPayload} from './uxScope.js';
import {equipmentTitle,moduleTitle} from './uiScope.js';
import {equipmentLabel,mergeNotices,normalizeNotice} from './notifications.js';

const cursor='2024-12-01 00:00:05';
const base={page:'overview',module:'all',cursorTime:cursor,gptConfigured:false};
const applied=Object.freeze({module:'17',signal:'current',grain:'raw',start:'2025-04-30',end:'2025-04-30',query_start:'2025-04-30 00:00:00',query_end:'2025-04-30 00:00:05'});
const recordedReview=Object.freeze({module:'15',start:'2025-02-28 04:37:20',end:'2025-02-28 04:39:35',source_kind:'recorded',event_id:'abnormal_pattern_0091',title:'예비건조기 · 모듈 15'});

test('applied archive module and effective clipped range win over draft module and calendar dates',()=>{
  const context=makeSearchContext(Object.freeze({...base,page:'equipment',module:'18',historyActive:true,appliedQuery:applied,gptConfigured:true}));
  assert.equal(context.module,'17');assert.equal(context.scope,'archive');assert.equal(context.grain,'raw');
  assert.equal(context.start,'2025-04-30 00:00:00');assert.equal(context.end,'2025-04-30 00:00:05');
  const payload=searchPayload('마지막 원본 관측은?',context);
  assert.equal(payload.context.module_id,'17');assert.equal(payload.context.end,'2025-04-30 00:00:05');
  assert.equal(payload.local_only,true);assert.equal(payload.context.cursor_time,cursor);
  assert.equal(payload.context.source_kind,'recorded');assert.equal(applied.module,'17');
});

test('aggregate archive preserves the applied whole-period bounds and grain',()=>{
  const context=makeSearchContext({...base,page:'equipment',module:'15',historyActive:true,appliedQuery:{...applied,module:'all',grain:'month',query_start:'2024-12-01 00:00:00',query_end:'2025-04-30 00:00:00'}});
  assert.deepEqual(searchPayload('월 전력량',context).context,{module_id:'all',cursor_time:cursor,scope:'archive',source_kind:'recorded',start:'2024-12-01 00:00:00',end:'2025-04-30 00:00:00',grain:'month'});
});

test('inactive history and unapplied draft dates cannot silently turn live search into archive search',()=>{
  for(const query of [{historyActive:false,appliedQuery:applied},{historyActive:true,appliedQuery:null},{historyActive:true,appliedQuery:{module:'17',start:'2025-04-30',end:'2025-04-30'}},{historyActive:true,appliedQuery:{...applied,query_end:undefined}}]){
    const context=makeSearchContext({...base,page:'equipment',module:'5',...query});
    assert.equal(context.module,'5');assert.equal(context.scope,'replay');
    const payload=searchPayload('전력',context);assert.equal(payload.context.module_id,'5');
    assert.equal(Object.hasOwn(payload.context,'start'),false);assert.equal(Object.hasOwn(payload.context,'end'),false);assert.equal(Object.hasOwn(payload.context,'grain'),false);
  }
});

test('applied archive state is ignored on another page',()=>{
  for(const page of ['overview','forecast','report']){
    const context=makeSearchContext({...base,page,module:'12',historyActive:true,appliedQuery:applied});
    assert.equal(context.scope,'replay');assert.equal(context.module,'12');assert.equal(context.start,undefined);
  }
});

test('diagnosis follows the reviewed event module and exact exclusive five-second window',()=>{
  const context=makeSearchContext({...base,page:'diagnosis',module:'all',reviewContext:recordedReview,historyActive:true,appliedQuery:applied});
  assert.equal(context.module,'15');assert.equal(context.scope,'diagnosis');assert.equal(context.grain,'raw');
  assert.equal(context.start,recordedReview.start);assert.equal(context.end,recordedReview.end);assert.equal(context.event_id,recordedReview.event_id);
  assert.equal(context.source_kind,'recorded');assert.equal(context.origin_source_kind,'recorded');
  assert.equal(searchPayload('이 구간 전류',context).local_only,true);
});

test('synthetic review is visibly retained as origin while its search is original recorded data only',()=>{
  const context=makeSearchContext({...base,page:'diagnosis',gptConfigured:true,reviewContext:Object.freeze({...recordedReview,source_kind:'synthetic',scope:'archive',grain:'month'})});
  assert.equal(context.origin_source_kind,'synthetic');assert.equal(context.source_kind,'recorded');
  assert.equal(context.scope,'diagnosis');assert.equal(context.grain,'raw');
  const payload=searchPayload('센서 변화',context);assert.equal(payload.local_only,true);assert.equal(payload.context.source_kind,'recorded');
  assert.equal(Object.hasOwn(payload.context,'origin_source_kind'),false);assert.equal(Object.hasOwn(payload.context,'event_id'),false);assert.equal(Object.hasOwn(payload.context,'title'),false);
});

test('missing or incomplete review context falls back to current replay without leaking an archive range',()=>{
  for(const reviewContext of [undefined,null,{}, {...recordedReview,start:''},{...recordedReview,end:undefined}]){
    const context=makeSearchContext({...base,page:'diagnosis',module:'17',reviewContext,historyActive:true,appliedQuery:applied});
    assert.equal(context.scope,'replay');assert.equal(context.module,'17');assert.equal(context.start,undefined);assert.equal(context.end,undefined);
  }
});

test('missing top-level input and nullish module use a local all-module replay fallback',()=>{
  for(const input of [undefined,{}, {module:null},{module:undefined}]){
    const context=makeSearchContext(input);
    assert.equal(context.module,'all');assert.equal(context.scope,'replay');assert.equal(context.source_kind,'recorded');assert.equal(context.gpt_configured,false);
    const payload=searchPayload('관측 요약',context);assert.equal(payload.local_only,true);assert.equal(payload.context.module_id,'all');
    assert.equal(Object.hasOwn(payload.context,'start'),false);assert.equal(Object.hasOwn(payload.context,'end'),false);
  }
  assert.equal(makeSearchContext({...base,module:17}).module,'17');
});

test('a missing or null payload context cannot implicitly authorize GPT or manufacture a time range',()=>{
  for(const context of [undefined,null,{}]){
    const payload=JSON.parse(JSON.stringify(searchPayload('관측 요약',context)));
    assert.deepEqual(payload,{question:'관측 요약',local_only:true,context:{module_id:'all',scope:'replay',source_kind:'recorded'}});
  }
});

test('a review context from a previous page does not override the current selected module',()=>{
  const context=makeSearchContext({...base,page:'overview',module:'5',reviewContext:recordedReview});
  assert.equal(context.module,'5');assert.equal(context.scope,'replay');assert.equal(context.event_id,undefined);
});

test('all explicit scopes force local search irrespective of GPT approval flag',()=>{
  for(const scope of ['archive','diagnosis'])for(const gpt_configured of [true,false,undefined]){
    const payload=searchPayload('역률',{module:'17',cursor_time:cursor,scope,start:recordedReview.start,end:recordedReview.end,grain:'raw',gpt_configured,source_kind:'synthetic'});
    assert.equal(payload.local_only,true);assert.equal(payload.context.source_kind,'recorded');assert.equal(payload.context.start,recordedReview.start);
  }
});

test('default replay uses only the GPT permission flag and strips stale explicit-range fields',()=>{
  for(const gpt_configured of [true,false,undefined]){
    const payload=searchPayload('전력',{module:'1',cursor_time:cursor,gpt_configured,configured:true,api_key_present:true,source_kind:'synthetic',start:recordedReview.start,end:recordedReview.end,grain:'raw',page:'diagnosis'});
    assert.equal(payload.local_only,!gpt_configured);assert.equal(payload.context.scope,'replay');assert.equal(payload.context.source_kind,'recorded');
    assert.deepEqual(Object.keys(payload.context).sort(),['module_id','cursor_time','scope','source_kind'].sort());
  }
});

test('the serialized API payload preserves the exclusive end and never includes internal title or permission metadata',()=>{
  const context=Object.freeze({...makeSearchContext({...base,page:'diagnosis',reviewContext:recordedReview,gptConfigured:true}),note:'internal-only'});
  const result=JSON.parse(JSON.stringify(searchPayload('  상별 역률  ',context)));
  assert.equal(result.question,'  상별 역률  ');assert.equal(result.context.end,'2025-02-28 04:39:35');assert.equal(result.local_only,true);
  assert.deepEqual(Object.keys(result.context).sort(),['module_id','cursor_time','scope','source_kind','start','end','grain'].sort());
  assert.equal(context.note,'internal-only');assert.equal(context.gpt_configured,true);
});

const sourceModules=Object.freeze([
  ['1','1(PM-3)','PM-3'],['2','2(L-1전등)','L-1전등'],['3','3(분쇄기(2))','분쇄기(2)'],['4','4(분쇄기(1))','분쇄기(1)'],
  ['5','5(좌측분전반)','좌측분전반'],['11','11(우측분전반1)','우측분전반1'],['12','12(4호기)','4호기'],['13','13(3호기)','3호기'],
  ['14','14(2호기)','2호기'],['15','15(예비건조기)','예비건조기'],['16','16(호이스트)','호이스트'],['17','17(6호기)','6호기'],['18','18(우측분전반2)','우측분전반2'],
].map(([id,source_tag,label])=>Object.freeze({id,source_tag,label})));

test('all thirteen audited source tags retain the correct name and module ID across helpers',()=>{
  for(const module of sourceModules){
    const expected=`${module.label} · 모듈 ${module.id}`;
    assert.equal(equipmentTitle(module.id,module.source_tag),expected);
    assert.equal(moduleTitle(sourceModules,module.id),expected);
    assert.equal(moduleTitle(sourceModules,Number(module.id)),expected);
    assert.equal(equipmentLabel({module_id:module.id,equipment:module.source_tag}),expected);
    assert.equal(equipmentLabel({equipment:module.source_tag}),expected);
    assert.equal(equipmentLabel({module_id:module.id,equipment_name:module.label}),expected);
  }
});

test('nested parentheses in real grinder names are preserved rather than mistaken for a module suffix',()=>{
  assert.equal(equipmentTitle('3','3(분쇄기(2))'),'분쇄기(2) · 모듈 3');
  assert.equal(equipmentLabel({equipment:'4(분쇄기(1))'}),'분쇄기(1) · 모듈 4');
});

test('already decorated names stay idempotent without repeated module IDs',()=>{
  for(const module of sourceModules){
    const once=equipmentTitle(module.id,module.label);
    assert.equal(equipmentTitle(module.id,once),once);assert.equal(equipmentLabel({module_id:module.id,equipment:once}),once);
  }
  assert.equal(equipmentTitle('17','모듈 17'),'모듈 17');assert.equal(equipmentTitle('17','17'),'모듈 17');
});

test('all-module and missing-name labels preserve scope instead of inventing an equipment name',()=>{
  assert.equal(moduleTitle(sourceModules,'all'),'전체 계측 모듈');assert.equal(equipmentTitle('all','공장 전체'),'전체 계측 모듈');
  assert.equal(moduleTitle(sourceModules,'99'),'모듈 99');assert.equal(equipmentLabel({module_id:'17'}),'모듈 17');
  assert.equal(equipmentLabel({}),'모듈 확인 필요');assert.equal(equipmentTitle(''),'모듈 확인 필요');
});

test('receiving or hydrating an ACKed alarm does not recast ACK as read status',()=>{
  const event=Object.freeze({id:'saved-id',module_id:'17',equipment:'17(6호기)',type:'abnormal_pattern',time:'2025-03-13 19:45:15',status:'acknowledged',occurrence_id:'occ-current'});
  const stored=normalizeNotice(event),live=normalizeNotice(event,{live:true,session:'current'});
  assert.equal(stored.unread,false);assert.equal(live.unread,true);assert.equal(stored.status,'acknowledged');assert.equal(live.status,'acknowledged');
  const merged=mergeNotices([stored],[event],{live:true,session:'current'});
  assert.equal(merged.length,1);assert.equal(merged[0].notice_key,'occ-current');assert.equal(merged[0].status,'acknowledged');
  assert.equal(equipmentLabel(merged[0]),'6호기 · 모듈 17');assert.equal(Object.hasOwn(event,'unread'),false);
});
