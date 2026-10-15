"""Read-only source review with AST-isolated pure logic and in-memory fixtures.

Never imports backend/config, starts a server, calls integrations, or opens the
real SQLite database. The reviewed backend source files are never modified.
"""
from __future__ import annotations
import ast
import copy
import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import types
import pandas as pd

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[1] / 'backend'

def extract(filename, name, namespace, class_name=None):
    text = (BACKEND / filename).read_text(encoding='utf-8')
    tree = ast.parse(text)
    if class_name:
        candidates=[n for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(m,ast.FunctionDef) and m.name==name for m in n.body)]
        selected=next((n for n in candidates if n.name==class_name),None)
        if selected is None:
            assert len(candidates)==1, f'ambiguous class for {name}'
            selected=candidates[0]
        scope=selected.body
    else:
        scope=tree.body
    node = next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(BACKEND / filename), 'exec'), namespace)
    return namespace[name]

def run():
    checks = []
    ns = {'pd': pd}
    augment = extract('engine.py', 'augment_records', ns)
    start = pd.Timestamp('2025-03-14 13:00:00')
    rows = []
    for elapsed in [119, 120, 299, 300]:
        for module in ['17', '15']:
            rows.append({'time': str(start + pd.Timedelta(seconds=elapsed)), 'module_id': module,
                         'activePower': 3000., 'reactivePowerLagging': 50., 'energy_kwh': 3000*5/3600,
                         'current_unbalance_pct': 0., 'pf_mean': 95.,
                         **{f'current{p}': 10. for p in 'RST'}, **{f'powerFactor{p}': 95. for p in 'RST'},
                         **{f'voltage{p}': 215. for p in 'RST'}})
    original = copy.deepcopy(rows)
    scenario = {'id':'synthetic_pf_fixture','source_kind':'synthetic','start':str(start),
                'module_id':'17','fault_type':'power_factor_drop'}
    altered = augment(rows, scenario)
    assert rows == original
    assert altered == augment(rows, scenario)
    for raw, result in zip(rows, altered):
        elapsed = (pd.Timestamp(raw['time'])-start).total_seconds()
        changed = raw['module_id']=='17' and 120<=elapsed<300
        assert result['powerFactorR'] == raw['powerFactorR']*(.55 if changed else 1.)
        assert result['source_kind']=='synthetic'
    checks.append({'check':'pure_synthetic_transform', 'status':'pass',
                   'scope':'module17 only; [120,300) seconds; input immutable; deterministic'})

    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript('CREATE TABLE alarms(id TEXT PRIMARY KEY,module_id TEXT,source_time TEXT,type TEXT,status TEXT,slack_status TEXT,created_at TEXT,payload TEXT); CREATE TABLE outbox(alarm_id TEXT PRIMARY KEY,status TEXT,attempts INTEGER DEFAULT 0,next_attempt TEXT,last_error TEXT);')
    db = types.SimpleNamespace(lock=threading.RLock(), connect=lambda:conn)
    ns = {'json':json,'hashlib':hashlib,'now':lambda:datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'dumps':lambda obj:json.dumps(obj,ensure_ascii=False,allow_nan=False,default=str)}
    alarm = extract('store.py','alarm',ns,'Database')
    alarms = extract('store.py','alarms',ns,'Database')
    active_count = extract('store.py','active_alarm_count',ns,'Database')
    base={'module_id':'17','type':'power_factor_drop','time':'2025-03-14 13:02:00',
          'source_kind':'synthetic','scenario_id':'scenario_A'}
    assert alarm(db,base) is True
    assert alarm(db,base) is False
    assert alarm(db,dict(base,scenario_id='scenario_B')) is True
    assert conn.execute('SELECT COUNT(*) FROM outbox').fetchone()[0] == 2
    for i in range(25):
        assert alarm(db,dict(base,source_kind='recorded',time=str(start+pd.Timedelta(hours=i))))
    assert len(alarms(db,'all',20)['rows']) == 20
    assert active_count(db) == 27
    conn.execute("UPDATE alarms SET status='acknowledged' WHERE id=(SELECT id FROM alarms LIMIT 1)")
    assert active_count(db)==26
    checks.append({'check':'alarm_identity_and_total_count','status':'pass',
                   'distinct_synthetic_scenarios':2,'same_event_inserted_once':True,
                   'open_count_before_ack':27,'display_limit':20,'open_count_after_ack':26,
                   'storage':'in-memory SQLite fixture; real DB never opened'})

    # Rows outside a historical window must never consume its SQL LIMIT.
    for i in range(35):
        assert alarm(db,dict(base,source_kind='recorded',time=str(pd.Timestamp('2025-02-11 13:00:00')+pd.Timedelta(hours=i))))
    for i in range(6):
        assert alarm(db,dict(base,scenario_id='range_fixture',time=str(pd.Timestamp('2025-02-11 13:00:00')+pd.Timedelta(hours=i))))
    for i in range(50):
        assert alarm(db,dict(base,source_kind='recorded',time=str(pd.Timestamp('2025-04-01 00:00:00')+pd.Timedelta(hours=i))))
    for i in range(10):
        assert alarm(db,dict(base,module_id='15',source_kind='recorded',time=str(pd.Timestamp('2025-02-11 13:00:00')+pd.Timedelta(hours=i))))
    bounded=alarms(db,'17',3,start='2025-02-11 13:00:00',end='2025-02-18 13:00:00',source_kind='recorded')
    assert bounded['total']==35 and len(bounded['rows'])==3
    assert all(r['source_kind']=='recorded' and r['module_id']=='17' and pd.Timestamp('2025-02-11 13:00:00')<=pd.Timestamp(r['time'])<pd.Timestamp('2025-02-18 13:00:00') for r in bounded['rows'])
    try: alarms(db,'17',3,source_kind='untrusted')
    except ValueError:pass
    else:raise AssertionError('invalid source kind accepted')
    checks.append({'check':'sql_scope_before_limit_and_exact_total','status':'pass',
                   'module':'17','source_kind':'recorded','eligible_total':35,'returned_rows':3,
                   'newer_outside_rows':50,'excluded_synthetic':6,'excluded_other_module':10,
                   'invalid_source_kind_rejected':True})

    hourly = pd.DataFrame({'time':pd.to_datetime(['2025-03-13 18:00:00','2025-03-13 19:00:00']),
                           'module_id':['17','17'],'energy_kwh':[10.,999.]})
    source = types.SimpleNamespace(hourly=hourly,modules=[{'id':'17'}])
    series_ns={'pd':pd,'SENSOR_KEYS':['activePower'],'iso':lambda t:str(pd.Timestamp(t))}
    series=extract('data_source.py','series',series_ns,'Source')
    got=series(source,'17','energy_kwh','2025-03-13 18:00:00','2025-03-13 19:43:00','hour')
    assert len(got)==1 and got[0]['value']==10.
    checks.append({'check':'hourly_bucket_availability','status':'pass',
                   'end':'2025-03-13 19:43:00','included_bucket':'18:00-19:00','excluded_incomplete_bucket':'19:00-20:00'})

    fixture_db=types.SimpleNamespace(alarms=lambda *a,**kw:alarms(db,*a,**kw))
    ns={'pd':pd,'iso':lambda t:str(pd.Timestamp(t)),'DB':fixture_db}
    facts=extract('services.py','facts',ns,'Reporting')
    calls=[]
    def forecast(module,points,**kwargs):
        calls.append(kwargs['origin']);return {'origin':str(kwargs['origin']),'forecast':[]}
    report=types.SimpleNamespace(source=source,engine=types.SimpleNamespace(cursor=pd.Timestamp('2025-03-13 19:43:00')),forecast=forecast)
    historical=facts(report,'17','2025-03-12 13:00:00')
    assert historical['forecast_origin']=='2025-03-12 13:00:00'
    try: facts(report,'17','2025-04-29 00:00:00')
    except ValueError: pass
    else: raise AssertionError('future report end accepted')
    checks.append({'check':'report_requested_origin_and_future_rejection','status':'pass',
                   'historical_report_forecast_origin':historical['forecast_origin']})
    report_past=facts(report,'17','2025-02-18 13:00:00')
    assert report_past['logged_alarm_candidates_total']==35
    assert len(report_past['logged_alarm_candidates'])==30
    assert all(a['source_kind']=='recorded' for a in report_past['logged_alarm_candidates'])
    checks.append({'check':'weekly_report_source_scope_and_untruncated_count','status':'pass',
                   'recorded_total':35,'evidence_rows':30,'synthetic_excluded':True})

    # Fault-event queries must apply their cursor window before limiting rows.
    fake_events=[{'time':'2025-03-14 13:02:10','module_id':'17'},
                 {'time':'2025-02-18 12:58:00','module_id':'17'},
                 {'time':'2025-02-01 00:00:00','module_id':'17'}]
    def fake_alarms(module='all',limit=1000,start=None,end=None,**kwargs):
        eligible=[r for r in fake_events if (module=='all' or r['module_id']==str(module))
                  and (start is None or pd.Timestamp(r['time'])>=pd.Timestamp(start))
                  and (end is None or pd.Timestamp(r['time'])<pd.Timestamp(end))]
        return {'rows':eligible[:limit],'total':len(eligible)}
    ns={'pd':pd,'iso':lambda t:str(pd.Timestamp(t)),'DB':types.SimpleNamespace(alarms=fake_alarms)}
    chat=extract('services.py','chat',ns,'Reporting')
    got=chat(report,'최근7일 고장 후보',{'module_id':'17','cursor_time':'2025-02-18 13:00:00'},local_only=True)
    expected_start=pd.Timestamp('2025-02-11 13:00:00');expected_end=pd.Timestamp('2025-02-18 13:00:00')
    time_safe=all(expected_start<=pd.Timestamp(r['time'])<expected_end for r in got['rows'])
    checks.append({'check':'chat_fault_events_cursor_scope','status':'pass'if time_safe else'fail',
                   'as_of':got['citations'][0]['as_of'],'returned_event_times':[r['time']for r in got['rows']],
                   'expected_start':str(expected_start),'expected_end_exclusive':str(expected_end)})
    conn.close()
    result={'status':'pass'if all(c['status']=='pass'for c in checks)else'issues_found',
            'method':'read-only source AST isolation; no backend imports, real DB, server or integration calls',
            'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_sha256':{f:hashlib.sha256((BACKEND/f).read_bytes()).hexdigest()
                             for f in ['app.py','engine.py','data_source.py','services.py','store.py','config.py']},
            'checks':checks}
    (HERE/'backend_readonly_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    review=dict(result)
    review.update({
        'summary':'검토한 범위에서 재현 가능한 중대 미해결 오류가 발견되지 않았습니다.'if result['status']=='pass'else'미해결 재현 오류가 있습니다.',
        'unresolved_findings':[]if result['status']=='pass'else[c for c in checks if c['status']!='pass'],
        'resolved_findings':[
            {'issue':'완료되지 않은 시간 집계의 미래 관측 포함','resolution':'time+1h<=end; 날짜 생략 조회는 현재 snapshot+5초까지만'},
            {'issue':'과거 보고서와 예측 원점 불일치','resolution':'선택된 보고서 종료 시각을 forecast origin에 전달; 미래 시각 거부'},
            {'issue':'과거 채팅에 미래 또는 기간 밖 알림 포함','resolution':'SQL의 module/start/end 조건을 LIMIT 전에 적용하고 정확한 total 반환'},
            {'issue':'주간 원본 보고서에 합성 시연 알림 포함 및 상세 제한을 총건수로 표시','resolution':'recorded 원천만 조회; 상세 최대30개와 total 별도'},
            {'issue':'합성 이벤트 상세를 원본 센서로 재현','resolution':'순수 결정적 augment_records로 해당 scenario 변환 재현'},
            {'issue':'서로 다른 합성 시연 알림 식별자 충돌','resolution':'synthetic scenario_id를 알림 identity에 포함; 반복 삽입은 멱등'},
            {'issue':'열린 알림 수를 최신20개 표시 개수로 제한','resolution':'전체 open 상태를 SQL COUNT로 집계'}],
        'constraints':['서버 코드 및 실제 SQLite DB 수정 없음','GPT/Slack/외부 API 호출 없음',
                       '예측 모델·선정·고정 평가·예측 경로 파일 수정 없음',
                       'AST로 추출한 실제 함수와 메모리 모의자료 검증; 네트워크·배포 전반의 보안 인증을 의미하지 않음'],
        'verification_script':str(HERE/'verify_backend_readonly.py')})
    docs=HERE.parents[1]/'docs';docs.mkdir(parents=True,exist_ok=True)
    (docs/'backend_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':result['status'],'checks':checks},ensure_ascii=False,indent=2))

if __name__=='__main__':run()
