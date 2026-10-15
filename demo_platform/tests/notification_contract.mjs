import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import {collectStreamDetections,equipmentLabel,faultLabel,mergeNotices,occurrenceTime} from '../frontend/src/notifications.js';

const checks=[];
const check=(name,run)=>{run();checks.push({name,passed:true});};
const event={id:'stored-alarm-1',event_id:'stored-alarm-1',occurrence_id:'session1-module17-pattern-onset',module_id:'17',equipment:'17(6호기)',type:'abnormal_pattern',first_time:'2025-03-13 19:45:05',detected_at:'2025-03-13 19:45:15',source_kind:'recorded'};
const frame={type:'telemetry',replay:{session_id:'session1'},new_detections:[event]};

check('old_snapshot_alarms_never_popup',()=>assert.deepEqual(collectStreamDetections({type:'telemetry',snapshot:{alarms:[event]}},new Set()),[]));
check('connection_and_state_are_quiet',()=>{for(const type of ['connected','state','replay'])assert.deepEqual(collectStreamDetections({...frame,type},new Set()),[]);});
check('only_first_delivery_of_occurrence_is_new',()=>{const seen=new Set();assert.equal(collectStreamDetections(frame,seen).length,1);assert.equal(collectStreamDetections(frame,seen).length,0);});
check('different_session_replay_not_silenced_by_saved_id',()=>{const seen=new Set();collectStreamDetections(frame,seen);const next={...event,occurrence_id:'session2-module17-pattern-onset'};assert.equal(collectStreamDetections({...frame,new_detections:[next]},seen).length,1);});
check('later_onset_in_same_saved_bucket_remains_separate',()=>{const seen=new Set();collectStreamDetections(frame,seen);assert.equal(collectStreamDetections({...frame,new_detections:[{...event,occurrence_id:'session1-later-onset',first_time:'2025-03-13 19:46:00'}]},seen).length,1);});
check('incomplete_detection_not_exposed_as_notice',()=>{const invalid=[{...event,id:null,event_id:null},{...event,module_id:null},{id:'missing-clock',module_id:'17'}];assert.deepEqual(collectStreamDetections({...frame,new_detections:invalid},new Set()),[]);});
check('stored_history_hydrates_read_without_live_popup',()=>{const history=mergeNotices([],[event]);assert.equal(history.length,1);assert.equal(history[0].unread,false);assert.equal(history[0].is_live,false);});
check('current_detection_replaces_its_archive_copy',()=>{const history=mergeNotices([],[event]);const live=mergeNotices(history,[event],{live:true});assert.equal(live.length,1);assert.equal(live[0].unread,true);assert.equal(live[0].is_live,true);});
check('history_refresh_preserves_current_unread_flag',()=>{const live=mergeNotices([],[event],{live:true});const refreshed=mergeNotices(live,[event]);assert.equal(refreshed.length,1);assert.equal(refreshed[0].unread,true);});
check('multiple_occurrences_of_saved_alarm_are_kept',()=>{let notices=mergeNotices([],[event],{live:true});notices=mergeNotices(notices,[{...event,occurrence_id:'later'}],{live:true});assert.equal(notices.length,2);assert.equal(notices[0].occurrence_id,'later');});
check('notification_history_is_bounded',()=>{const events=Array.from({length:70},(_,n)=>({...event,id:`id-${n}`,event_id:`id-${n}`,occurrence_id:`occurrence-${n}`}));assert.equal(mergeNotices([],events,{live:true}).length,50);});
check('equipment_and_fault_are_human_readable',()=>{assert.equal(equipmentLabel(event),'6호기');assert.equal(faultLabel(event),'비정상 패턴');});
check('occurrence_clock_precedes_confirmation_clock',()=>assert.equal(occurrenceTime(event),'2025-03-13 19:45:05'));
check('synthetic_source_and_exact_detail_reference_preserved',()=>{const notices=mergeNotices([],[{...event,source_kind:'synthetic',detail_event_id:'occurrence-current'}],{live:true});assert.equal(notices[0].source_kind,'synthetic');assert.equal(notices[0].detail_event_id,'occurrence-current');});

const result={status:'passed',checks,external_calls:0};
writeFileSync(new URL('./notification_contract_verification.json',import.meta.url),JSON.stringify(result,null,2),'utf8');
console.log(JSON.stringify({status:result.status,checks:checks.length}));
