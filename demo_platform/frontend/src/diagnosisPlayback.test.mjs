import test from 'node:test';
import assert from 'node:assert/strict';
import {advancePlayback,contextError,initialSignal,replayEventId,replayScores,replayTrace,sensorValueDomain} from './diagnosisPlayback.js';
import {recordedMillis,rowIsCandidate} from './data.js';

test('padding accepts exact five-second intervals including zero and rejects invalid drafts',()=>{
  for(const pair of [[60,60],[0,0],[1800,1800],['30','90']])assert.equal(contextError(...pair),'');
  for(const pair of [['',60],[' ',60],[61,60],[-5,60],[1805,60],[5.5,60],['x',60]])assert.ok(contextError(...pair));
});
test('stream returns to first sample after each complete pass without retaining earlier pass rows',()=>{
  let state={index:0,cycle:1};const seen=[];
  for(let tick=0;tick<6;tick++){seen.push([state.index,state.cycle]);state=advancePlayback(state,3);}
  assert.deepEqual(seen,[[0,1],[1,1],[2,1],[0,2],[1,2],[2,2]]);
  assert.deepEqual(state,{index:0,cycle:3});
  assert.deepEqual(advancePlayback({index:0,cycle:4},1),{index:0,cycle:5});
  assert.deepEqual(advancePlayback({index:0,cycle:2},0),{index:0,cycle:1});
});
test('current occurrence identity wins over persistent alarm bucket identity',()=>{
  assert.equal(replayEventId({detail_event_id:'occ-current',occurrence_id:'occ-other',event_id:'saved',id:'saved'}),'occ-current');
  assert.equal(replayEventId({occurrence_id:'occ-current',id:'saved'}),'occ-current');
  assert.equal(replayEventId({event_id:'archive'}),'archive');
});
const payload={event:{module_id:'17',type:'abnormal_pattern'},window:{start:'2025-03-13 19:44:05',end_inclusive:'2025-03-13 19:46:15',event_start:'2025-03-13 19:45:05',event_end:'2025-03-13 19:45:15'},
  series:[{time:'2025-03-13 19:45:10',module_id:'17',activePower:2},{time:'2025-03-13 19:45:05',module_id:'17',activePower:1}],
  algorithm:[{time:'2025-03-13T19:45:05',rules:[{type:'abnormal_pattern',label:'비정상 패턴',score:2,threshold:1}]}],
  ai:[{time:'2025-03-13 19:45:05',ae_score:.2,ae_threshold:.1,ae_alarm:true,iforest_score:.01,iforest_threshold:.1}]};
test('sensor/rules/AI align on original clock without timezone conversion or invented missing scores',()=>{
  const trace=replayTrace(payload);assert.equal(trace.rows[0].recorded_at,'2025-03-13 19:45:05');
  assert.deepEqual(trace.domain,[recordedMillis(payload.window.start),recordedMillis(payload.window.end_inclusive)]);
  assert.ok(trace.ticks.every(t=>(t-trace.domain[0])%5000===0));
  assert.equal(trace.ticks[0],trace.domain[0]);assert.equal(trace.ticks.at(-1),trace.domain[1]);
  assert.equal(trace.rules[0].label,'비정상 패턴');assert.ok(rowIsCandidate(trace.rows[0]));
  assert.equal(trace.highlight.end,'2025-03-13 19:45:20');
  assert.deepEqual(replayScores(trace.rows,'rules','abnormal_pattern').map(r=>r.score),[2,undefined]);
  assert.deepEqual(replayScores(trace.rows,'ae','abnormal_pattern').map(r=>r.score),[.2,undefined]);
  assert.deepEqual(replayScores(trace.rows,'iforest','abnormal_pattern').map(r=>r.score),[.01,undefined]);
});
test('zero-context single point still has visible axis, with no added observation',()=>{
  const trace=replayTrace({series:[{time:'2025-03-13 19:45:05',module_id:'17'}],window:{start:'2025-03-13 19:45:05',end_inclusive:'2025-03-13 19:45:05'}});
  assert.equal(trace.rows.length,1);assert.equal(trace.domain[1]-trace.domain[0],5000);assert.equal(new Set(trace.ticks).size,trace.ticks.length);assert.equal(replayTrace(null).rows.length,0);
});
test('fault-specific default signals show the relevant electrical measurement',()=>{
  assert.equal(initialSignal('power_factor_drop'),'powerFactor');assert.equal(initialSignal('reactive_power_surge'),'reactivePowerLagging');
  assert.equal(initialSignal('current_only_increase'),'current');assert.equal(initialSignal('abnormal_pattern'),'activePower');
});
test('per-sensor scale covers all phases in the complete window, excluding missing or invalid values',()=>{
  const rows=[{voltageR:210,voltageS:212,voltageT:null},{voltageR:205,voltageS:'',voltageT:215},{voltageR:NaN,voltageS:Infinity}];
  assert.deepEqual(sensorValueDomain(rows,['voltageR','voltageS','voltageT']),[204.4,215.6]);
  assert.equal(sensorValueDomain(rows,['missing']),null);
  assert.deepEqual(sensorValueDomain([{activePower:0}],['activePower']),[-.06,.06]);
  assert.deepEqual(sensorValueDomain([{currentR:10},{currentR:10}],['currentR']),[9.4,10.6]);
});
