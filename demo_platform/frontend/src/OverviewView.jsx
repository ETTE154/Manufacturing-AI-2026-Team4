import React from 'react';
import {Bell,CheckCircle2,ChevronRight,Clock3,Cpu,Gauge,TriangleAlert,Zap} from 'lucide-react';
import {moduleTitle,receivedClock} from './uiScope.js';
import {equipmentLabel,faultLabel} from './notifications.js';

export default function OverviewView({ui,module,metadata,status,modules,latest,latestByModule,activeModuleIds,visibleModules,selectedActiveIds,sessionDetections,currentSessionNotices,running,liveWindow,chartHighlight,logs,onEquipment,onAlerts}){
  const {Stat,Panel,Badge,SensorChart,Empty,fmt,time}=ui;
  const activeIds=new Set(activeModuleIds.map(String));
  const receivedCount=visibleModules.filter(item=>latestByModule[item.id]).length;
  const phases=['R','S','T'].map(phase=>latest[`powerFactor${phase}`]);
  const meanPowerFactor=phases.every(value=>value!=null&&Number.isFinite(Number(value)))?phases.reduce((sum,value)=>sum+Number(value),0)/3:null;
  const notices=currentSessionNotices.filter(event=>module==='all'||String(event.module_id??event.module)===String(module)).slice(0,2);
  const activeCount=selectedActiveIds.length;
  return <div className="op-overview">
    <div className="stat-grid op-overview-kpis">
      <Stat label="현재 이상 감지 모듈" value={activeCount} unit="모듈" icon={TriangleAlert} detail={activeCount?'3연속 관측 기준 · 확인 필요':'현재 감지 없음'} tone={activeCount?'alert':''}/>
      <Stat label={module==='all'?'현재 전력 합계':'현재 전력'} value={fmt(latest.activePower)} unit="kW*" icon={Zap} detail={module==='all'?'선택 범위 모듈 합계':'선택 모듈의 최신 관측'} tone="emphasis"/>
      <Stat label="평균 역률" value={fmt(meanPowerFactor)} unit="%*" icon={Gauge} detail="선택 범위 · R·S·T 평균"/>
    </div>
    <div className="op-overview-main">
      <Panel title="전력 흐름" subtitle="최근 5분 · 5초 관측" className="op-power-panel" action={<Badge tone={running?'teal':'muted'}>{running?'재생 중':'일시정지'}</Badge>}>
        <SensorChart rows={liveWindow.rows} height={340} domain={liveWindow.domain} ticks={liveWindow.ticks} highlight={chartHighlight}/>
        <div className="chart-foot"><span>{moduleTitle(modules,module)} · 유효전력 kW*</span><span>기록 {time(status.recorded_at)}</span></div>
      </Panel>
      <Panel title="이상 알림" className="op-alert-panel" action={<Bell size={18} aria-hidden="true"/>}>
        {activeCount>0&&<div className="op-active-modules">{selectedActiveIds.slice(0,3).map(id=><button className="text-button" key={id} onClick={()=>onEquipment(id)}>{moduleTitle(modules,id)}<ChevronRight size={14}/></button>)}{activeCount>3&&<span>외 {activeCount-3}개 모듈</span>}</div>}
        <div className="op-recent-notices">{notices.length?<ul>{notices.map((notice,index)=><li key={notice.id||notice.event_id||index}><strong>{equipmentLabel(notice)}</strong><span>{faultLabel(notice)}</span><time>{time(notice.first_time||notice.time||notice.start,true)}</time></li>)}</ul>:<p>이 재생에서 수신한 알림이 없습니다.</p>}</div>
        <button className="button secondary op-alert-open" onClick={onAlerts}>알림 센터 열기<ChevronRight size={15}/></button>
      </Panel>
    </div>
    <Panel title="계측 모듈" className="op-module-panel" action={<div className="op-module-heading"><Badge tone={receivedCount?'teal':'muted'}>관측 수신 {receivedCount} / {visibleModules.length}</Badge><button className="text-button" onClick={()=>onEquipment()}>설비분석<ChevronRight size={14}/></button></div>}>
      <div className="table-scroll"><table><thead><tr><th>모듈</th><th>유효전력*</th><th>R상 전압</th><th>R상 전류</th><th>관측 상태</th></tr></thead><tbody>{visibleModules.map(item=>{const row=latestByModule[item.id],active=activeIds.has(String(item.id));return <tr key={item.id} onClick={()=>onEquipment(item.id)}><td><span className="module-cell"><span className="module-icon"><Cpu size={14}/></span><button className="text-button" onClick={event=>{event.stopPropagation();onEquipment(item.id);}} aria-label={`${moduleTitle(modules,item.id)} 설비분석 열기`}>{moduleTitle(modules,item.id)}</button></span></td><td>{fmt(row?.activePower)} <small>kW*</small></td><td>{fmt(row?.voltageR)} <small>V</small></td><td>{fmt(row?.currentR)} <small>A</small></td><td><Badge tone={active?'rose':row?'teal':'muted'}>{active?'이상 감지':row?'관측 수신':'수신 대기'}</Badge></td></tr>})}</tbody></table>{!visibleModules.length&&<Empty title="모듈 정보를 기다리고 있습니다"/>}</div>
    </Panel>
    <details className="op-details op-overview-reference"><summary>데이터 참고</summary><div className="op-reference-copy"><p>이 화면은 과거 기록의 재생이며 실제 공장의 현재 상태가 아닙니다. 감지 없음은 정상 운전을 확정하는 뜻이 아닙니다.</p><p>* 전력·역률은 가이드와 원자료 표기의 잠정 단위입니다. 계측 모듈을 독립 설비로 단정할 수 없으며, 전체 전력 합계에는 분전반과 개별 설비의 중복 계측이 포함될 수 있습니다.</p><p>원본 기록 {metadata?.data_range?.start||'확인 중'} ~ {metadata?.data_range?.end||'확인 중'} · 관측 간격 5초 · {status.source_kind==='synthetic'?'원본 기반 합성 시연':'원본 관측 재생'}.</p><p>평균 역률은 상별·모듈별 산술 평균이며 소비량으로 가중하지 않습니다. 이번 재생 전체 감지 알림 {sessionDetections}건.</p></div>
      <details className="op-receive-details"><summary>최근 수신 로그 · {logs.length}건</summary><div className="log-list">{logs.length?logs.slice(0,6).map((log,index)=><div className="log" key={index}><CheckCircle2 size={15}/><div><strong>{log.type==='stored'?(log.equipment||log.module_id)+' · 수집 기록':log.count+'개 관측 수신'}</strong><span>기록 {time(log.recorded_at,true)}</span></div><time>{receivedClock(log.received_at)}</time></div>):<Empty icon={Clock3} title="수신 로그가 없습니다" text="재생을 시작하면 수집 이력이 표시됩니다."/>}</div></details>
    </details>
  </div>;
}
