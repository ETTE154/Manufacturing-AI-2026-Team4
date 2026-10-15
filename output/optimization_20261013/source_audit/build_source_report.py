from pathlib import Path
import json,html,hashlib,datetime
root=Path(r"E:\AI work\AWS_프로젝트");out=root/"output/optimization_20261013/source_audit"
path=out/"audit_report.json";d=json.loads(path.read_text(encoding="utf-8"))
physics_path=root/"output/optimization_20261013/research/forecastability_audit.json"
physics=json.loads(physics_path.read_text(encoding="utf-8"))["phase_power_diagnostic"]
d["conditional_phase_power_diagnostic"]={"source":str(physics_path),"source_sha256":hashlib.sha256(physics_path.read_bytes()).hexdigest(),**physics}
item={"topic":"Conditional electrical power consistency","status":"conditional_source_scope_or_units_inconsistency","finding":"Under same-circuit/time and line-to-neutral voltage assumptions, reported activePower divided by sum(Vphase*Iphase*PFphase/100)/1000 has median288.2914 when activePower is treated as guide kW. Treating reported activePower as W gives median0.28829, so division by1000 alone does not resolve this conditional identity.","evidence_scope":"16,849 previously sampled observations; q01/q99 ratios271.3145/307.3368 in the guide-kW interpretation. Not an exhaustive physical audit or proof of sensor failure.","modeling_action":"Obtain authoritative power units, phase channel topology, meter scaling and circuit inclusion from the provider. Preserve the original values and provisional model scale."}
d["semantic_findings"].insert(2,item)
path.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
esc=lambda x:html.escape(str(x))
def section(title,body):return "<section><h2>"+esc(title)+"</h2>"+body+"</section>"
def table(headers,rows):
 return '<div class="scroll"><table><thead><tr>'+"".join("<th>"+esc(h)+"</th>" for h in headers)+"</tr></thead><tbody>"+"".join("<tr>"+"".join("<td>"+cell+"</td>" for cell in row)+"</tr>" for row in rows)+"</tbody></table></div>"
parts=[]
rows=[]
for x in d["file_comparisons"]:
 p=d["csv_profiles"][x["source_key"]]["downloaded_profile"]
 bounds=(p["recorded_localtime_start"]+" ~ "+p["recorded_localtime_end"]) if "recorded_localtime_start" in p else (p["id_start"]+" ~ "+p["id_end"])
 rows.append([esc(x["source_key"]),f'{x["original"]["bytes"]:,}',f'{x["downloaded"]["bytes"]:,}',f'{p["rows"]:,}',esc(bounds),'<span class="pass">완전 동일</span><br><code>'+esc(x["original"]["sha256"])+"</code>"])
parts.append(section("공식 사본과 기존 파일 비교",table(["파일","기존 bytes","다운로드 bytes","행 수","기록 시간 범위","SHA-256 비교"],rows)+"<p>다운로드와 기존 파일에서 SHA-256을 각각 직접 계산했습니다. S3 ETag는 객체 버전 관련 메타데이터로만 기록했으며 MD5나 내용 검증값이라고 가정하지 않았습니다.</p>"))
parts.append(section("공개 파일 목록",table(["공식 S3 key","bytes","Last modified"],[[esc(o["key"]),f'{o["bytes"]:,}',esc(o["last_modified"])] for o in d["source_inventory"]["objects"]])+"<p>공식 페이지의 JavaScript가 지정한 공개 S3 목록 API를 폴더별로 끝까지 조회했습니다. 공개 목록에 ZIP, 추가 가이드 PDF, 고장 라벨 또는 삽입 위치 마스크는 없습니다. 비공개 자료의 존재 여부는 확인하지 않았습니다.</p>"))
train=d["csv_profiles"]["traing-dataset/rtu_data_full.csv"]["downloaded_profile"];may=d["csv_profiles"]["may-dataset/rtu_ground_truth_may.csv"]["downloaded_profile"]
parts.append(section("CSV 구조·시간 검증","<p>학습 파일은 <strong>19열·13개 모듈·33,696,013행</strong>입니다. 모듈별 2,592,001행이며 기록된 localtime은 2024-12-01 00:00부터 2025-04-30 00:00까지입니다. 마지막 시각의 모듈별 1행은 완전한 1시간으로 취급하면 안 됩니다.</p><p>5월 정답은 5열·505행이며 2025-05-10 00:00부터 2025-05-31 00:00까지 1시간 간격으로 연속합니다. id 중복과 결측은 없습니다.</p><p><strong>학습 헤더:</strong> "+esc(", ".join(train["headers"]))+"</p><p><strong>5월 헤더:</strong> "+esc(", ".join(may["headers"]))+"</p><p>행 수·키 범위는 공식 다운로드 사본을 직접 청크 파싱했습니다. 기존 파일은 바이트 단위로 동일하므로 같은 결과가 보장되며, 이전 원본 전수 시간 감사의 33,696,013행과도 일치합니다.</p>"))
rows=[]
labels={"Timestamp/localtime":"timestamp와 localtime","Power/energy units":"전력·에너지 단위","Conditional electrical power consistency":"조건부 전기적 일관성","Meter hierarchy / May target scope":"메인·서브미터 범위","Injected anomalies / verified faults":"삽입 이상치 정답","Guide and source paths":"가이드와 실제 경로"}
notes={
"Timestamp/localtime":"기록 localtime은 5초 연속이나 Unix timestamp의 local-naive 오프셋은 3/9에 −8h에서 −7h로 변합니다. 한국시간 가이드와 다르고 Unix 키 중복을 유발합니다. 미국 Pacific DST와 닮은 패턴이나 의도한 시간대는 출처 명세가 필요합니다.",
"Power/energy units":"CSV 이름만으로 물리 단위를 인증할 수 없습니다. 가이드의 accumActiveEnergy는 Wh/kWh 표기가 혼재합니다. 5월 요금·탄소 열은 180 및 0.424 계수에 수치상 일치하지만 독립적인 단위 확인 근거는 아닙니다.",
"Conditional electrical power consistency":"같은 회로·시각, 상전압=상대중성점 전압이라는 가정에서 상별 V×I×PF 합과 activePower가 맞지 않습니다. 표본 16,849행에서 가이드 kW 해석 비율의 중앙값은 288.29배이며 W로 해석해도 0.28829배입니다. /1000 수정만으로 해결된다고 단정할 수 없습니다.",
"Meter hierarchy / May target scope":"메인·서브미터 연결 관계와 5월 target에 포함된 모듈 목록이 없습니다. 전체 모듈 합계가 인증된 공장 전체 소비량인지는 미확인입니다.",
"Injected anomalies / verified faults":"공개 삽입 마스크·실제 고장 라벨이 없습니다. 이전 규칙 후보 63,963행은 확인된 삽입 고장 수가 아니며 operation도 고장 정답이 아닙니다.",
"Guide and source paths":"파일명은 가이드 요약과 일치합니다. 실제 공식 폴더는 traing-dataset이며 가이드의 training-dataset(s)와 철자가 다릅니다. 공개 가이드 PDF가 없어 가이드 자체의 원격 동일성은 확인하지 못했습니다."
}
for x in d["semantic_findings"]:rows.append([esc(labels[x["topic"]]),esc(notes[x["topic"]])])
parts.append(section("파일 전달과 구분해야 하는 출처 데이터 문제",table(["항목","확인 결과와 해석 범위"],rows)+"<p>복사 손상이 없다는 결론이 단위·시간대·고장 정답의 타당성을 보증하지는 않습니다. 원본 값은 보존하고 출처 제공자에게 단위, 시간대, 계측 회로·합산 범위, 삽입 마스크를 확인해야 합니다.</p>"))
parts.append(section("재현 가능한 증거","<ul><li><a href=\"source_manifest.json\">source_manifest.json</a>: 공개 파일 목록, 다운로드 URL, HTTP 헤더와 다운로드 증거</li><li><a href=\"byte_comparison.json\">byte_comparison.json</a>: 기존·다운로드 SHA-256 및 bytes</li><li><a href=\"csv_profiles.json\">csv_profiles.json</a>: 공식 사본의 헤더·행 수·시간 범위</li><li><a href=\"audit_report.json\">audit_report.json</a>: 전체 감사와 기존 전수 감사 연결</li><li><a href=\"source_page.html\">source_page.html</a>: 직접 받은 공식 페이지 원문</li></ul><p>모든 신규 파일은 source_audit 폴더에 저장했습니다. 기존 원본·analysis·modeling 파일을 변경하지 않았고, 외부 스크립트나 PDF 인증 문자열을 실행·출력하지 않았습니다.</p>"))
css="""*{box-sizing:border-box}body{background:#edf3f5;color:#183043;font-family:'Malgun Gothic','Segoe UI',sans-serif;line-height:1.7;margin:0}header{background:#143f4b;color:white;padding:38px max(24px,5vw)}h1{font-size:29px;line-height:1.4;max-width:1100px;margin:8px 0 14px}header p{max-width:1050px;color:#d9ebed}main{max-width:1400px;padding:10px 24px;margin:auto}section{background:white;border:1px solid #d9e5e9;border-radius:12px;padding:25px;margin:20px 0}h2{font-size:21px;margin:0 0 18px;color:#164d58}p{margin:12px 0}.eyebrow{color:#90e0ce;font-size:12px;letter-spacing:1px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th{background:#eaf3f4;color:#184b57;text-align:left;padding:12px;white-space:nowrap}td{padding:12px;border-bottom:1px solid #e0e9ec;vertical-align:top}tr:last-child td{border-bottom:0}td:first-child{min-width:160px}td:last-child{min-width:250px}code{word-break:break-all;font-size:11px}.pass{font-weight:bold;color:#057954}a{color:#087a86;text-underline-offset:3px}footer{padding:10px 24px 35px;max-width:1400px;margin:auto;color:#627781;font-size:12px}@media(max-width:720px){header{padding:25px 18px}h1{font-size:23px}main{padding:0 10px}section{padding:18px}}"""
content='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>공식 데이터 직접 다운로드·동일성 감사</title><style>'+css+'</style></head><body><header><div class="eyebrow">SOURCE AUDIT · 2026-10-13 KST</div><h1>공식 데이터는 기존 파일과 완전히 같습니다.</h1><p>학습 4.91GB와 5월 정답을 공식 공개 S3에서 직접 내려받아 기존 파일과 바이트 수·SHA-256·CSV 구조를 확인했습니다. 전달 과정의 손상과 출처 자체의 데이터 명세 문제를 구분합니다.</p></header><main>'+"".join(parts)+'</main><footer>공식 페이지: <a href="'+esc(d["official_page"])+'">sahmyook-project 공개 다운로드 페이지</a> · 감사 원본 보존 · PDF 인증정보 비노출</footer></body></html>'
(out/"audit_report.html").write_text(content,encoding="utf-8")
assert "<h1>공식 데이터는 기존 파일과 완전히 같습니다." in content
assert len(d["file_comparisons"])==2 and d["all_byte_identical"]
print(json.dumps({"report":str(out/"audit_report.html"),"html_bytes":(out/"audit_report.html").stat().st_size,"json_bytes":path.stat().st_size,"train_rows":train["rows"],"source_bytes":[x["downloaded"]["bytes"] for x in d["file_comparisons"]],"status":"complete"},ensure_ascii=False))

