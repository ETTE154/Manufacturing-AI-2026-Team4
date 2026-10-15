from pathlib import Path
import json, html, datetime
folder=Path(r"E:\AI work\AWS_프로젝트\output\modeling\research")
d=json.loads((folder/"literature_review.json").read_text(encoding="utf-8-sig"))
sources={s["id"]:s for s in d["sources"]}
esc=lambda x:html.escape(str(x))
def cite(ids):
    return " ".join('<a href="'+esc(sources[i]["url"])+'" target="_blank" rel="noopener">'+esc(sources[i]["title"])+'</a>' for i in ids)
def p(value): return "<p>"+esc(value)+"</p>"
def ul(items): return "<ul>"+"".join("<li>"+esc(x)+"</li>" for x in items)+"</ul>"
def table(head, rows):
    return '<div class="scroll"><table><thead><tr>'+"".join("<th>"+esc(x)+"</th>" for x in head)+"</tr></thead><tbody>"+"".join("<tr>"+"".join("<td>"+x+"</td>" for x in row)+"</tr>" for row in rows)+"</tbody></table></div>"
def section(title, body): return "<section><h2>"+esc(title)+"</h2>"+body+"</section>"
parts=[]
parts.append(section("판단과 적용 범위", ul(d["conclusions"])+p(d["scope"])))
rows=[]
for x in d["datasets"]:
    rows.append([esc(x["name"]),esc(x["resolution"]),esc(x["entities"]),esc(x["target"]),esc(x["coverage"]),esc(x["similarity"])+"<br><small>"+esc(x["limits"])+"</small><br>"+cite(x["sources"])])
parts.append(section("유사 오픈 데이터 비교",table(["데이터","해상도","설비·변수","target","기간","프로젝트 적용·한계"],rows)))
rows=[]
for x in d["hipe_studies"]:
    horizon=x["horizon_duration"]
    horizon=", ".join(horizon) if isinstance(horizon,list) else horizon
    rows.append([esc(x["name"])+"<br>"+esc(x["year"]),esc(x["task"]),esc(x["resolution"]),esc("미확인/미설정" if x["input_steps"] is None else x["input_steps"]),esc(horizon),esc(x["result"])+"<br><small>"+esc(x.get("limits",""))+"</small><br>"+cite(x["sources"])])
parts.append(section("HIPE 실제 연구 task와 예측 범위",table(["연구","task","해상도","input steps","forecast horizon","근거·해석"],rows)+p("Sanni et al.의 보고 수치는 원고 인덱스에서 확인했지만 전체 원고 직접 접근이 실패하여 단위를 확인하지 못했다. 이 수치를 프로젝트 kWh MAE와 직접 비교하지 않는다.")))
rows=[]
for x in d["ecl_benchmark"]:
    for i,h in enumerate(x["horizons"]):
        rows.append([esc(x["model"]),esc(x["lookback"]),esc(h),f'{x["MSE"][i]:.3f}',f'{x["MAE"][i]:.3f}',esc(x["table"])+"<br>"+cite([x["paper"]])])
parts.append(section("ECL 논문 결과: 원문별 protocol을 함께 확인",p("동일한 hour horizon이라도 lookback, 전처리와 학습 조건이 다르다. 아래 정규화 오차는 이 프로젝트의 물리 단위 MAE가 아니다. PatchTST L512와 iTransformer L96을 동일조건 우승 비교로 해석하지 않는다.")+table(["모델","lookback","H","MSE","MAE","표·원문"],rows)))
rows=[]
for x in d["candidate_models"]:
    rows.append([esc(x["name"])+"<br><small>"+esc(x["release"])+"</small>",esc(x["method"]),esc(x["role"]),esc(x["cpu_feasibility"]),esc(x["configuration"]),esc(x["evidence"])+"<br>"+cite(x["sources"])])
parts.append(section("모델 선정과 CPU 실현 조건",table(["모델·발표","방법","비교 역할","실행 판단","초기 설정","성능 근거의 범위"],rows)))
parts.append(section("같은 H 값의 실제 시간",table(["해상도","H96","H192","H336","H720"],[[esc(x[k]) for k in ("resolution","H96","H192","H336","H720")] for x in d["horizon_mapping"]])+p("일반 논문에서 H96은 첫 미관측 시각 origin부터 0..95의 96개 값이다. 96이라는 숫자만 같아서는 실제 예측 시간이 같지 않다.")))
proto=d["proposed_protocol"]
rows=[]
for key,label in [("unit","단위"),("hierarchy","측정 범위"),("last_observed_hour","마지막 완전 학습 시간"),("forecast_origin","첫 미관측 origin"),("forecast_length","총 출력 수"),("offsets_zero_based","origin 기준 0-based"),("leads_from_last_observed_one_based","마지막 관측 기준 1-based"),("scored_offsets_zero_based","점수 offsets"),("scored_leads_one_based","점수 leads"),("scored_slice_python","Python slice"),("scored_labels","평가 label 수"),("scoring_range_inclusive","평가 범위"),("endpoint_rule","원시 끝점"),("primary_metric","지표"),("validation","검증 권고"),("lookback","입력 길이")]:
    rows.append([esc(label),esc(proto[key])])
parts.append(section("이 프로젝트의 예측 protocol 제안",table(["조건","값·설명"],rows)+ul(proto["leakage_rules"])+p(proto["project_inference"])))
diag=d["diagnostic_protocol"]
parts.append(section("분류와 Autoencoder 평가 해석",p(diag["classification_target"])+p(diag["classification_metrics"])+p(diag["autoencoder"])+p(diag["evaluation"])+ul(diag["cautions"])+cite(diag["sources"])))
rows=[]
for x in d["sources"]:
    rows.append([esc(x["id"]),'<a href="'+esc(x["url"])+'">'+esc(x["title"])+'</a>',esc(x["type"]),esc(x.get("license",""))+"<br><small>"+esc(x.get("access_note",""))+"</small>"])
parts.append(section("검토한 1차 자료",table(["ID","자료","유형","권리·접근 제한"],rows)))
css="""*{box-sizing:border-box}body{margin:0;background:#edf3f7;color:#172a3a;font-family:'Malgun Gothic','Segoe UI',sans-serif;line-height:1.65}header{background:#133447;color:white;padding:42px max(5vw,24px)}header .eyebrow{color:#8ee1d2;font-size:13px;letter-spacing:1px}h1{font-size:30px;line-height:1.35;max-width:1050px}header p{color:#d0e1e8;max-width:1050px}main{max-width:1500px;margin:26px auto;padding:0 24px}section{background:white;border:1px solid #dce7ed;border-radius:12px;margin:20px 0;padding:24px}h2{font-size:20px;margin:0 0 18px;color:#16465a}p{margin:12px 0}ul{padding-left:24px}li{margin:10px 0}a{color:#096d80;text-decoration:underline;text-underline-offset:3px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th{background:#e9f3f4;color:#16465a;text-align:left;font-weight:700;padding:11px;white-space:nowrap}td{border-bottom:1px solid #e2e9ed;padding:11px;vertical-align:top;min-width:90px}td:last-child{min-width:240px}tr:last-child td{border-bottom:0}small{color:#576d7b;font-size:12px}footer{max-width:1500px;padding:10px 24px 40px;margin:auto;color:#576d7b;font-size:12px}@media(max-width:750px){header{padding:28px 20px}h1{font-size:24px}main{padding:0 12px}section{padding:18px}}"""
out='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+esc(d["title"])+'</title><style>'+css+'</style></head><body><header><div class="eyebrow">RESEARCH / INDUSTRIAL POWER TIME SERIES</div><h1>'+esc(d["title"])+'</h1><p>검토일 '+esc(d["reviewed_at"])+' · 산업 데이터 구조, 장기 예측 protocol, 모델 최신성과 실험 가능성을 나누어 확인</p></header><main>'+"".join(parts)+'</main><footer>본 문서는 문헌 조사와 실험 설계 제안이다. 실행 모델, 실측 속도 및 로컬 성능은 별도 모델링 결과에서 확인한다. 원본 데이터 및 기존 분석 결과를 수정하지 않았다.</footer></body></html>'
(folder/"research_review.html").write_text(out,encoding="utf-8")
text_lines=[d["title"],"검토일: "+d["reviewed_at"],d["scope"],""]+d["conclusions"]+[""]
for key in ["datasets","hipe_studies","ecl_benchmark","candidate_models","horizon_mapping"]:
    text_lines += ["\n"+key+"\n",json.dumps(d[key],ensure_ascii=False,indent=2)]
text_lines += ["\nproposed_protocol\n",json.dumps(proto,ensure_ascii=False,indent=2),"\ndiagnostic_protocol\n",json.dumps(diag,ensure_ascii=False,indent=2),"\nsources\n"]
for x in d["sources"]: text_lines.append(x["id"]+" | "+x["title"]+" | "+x["url"])
(folder/"research_review.txt").write_text("\n".join(text_lines),encoding="utf-8")
assert len(d["ecl_benchmark"])==5
assert proto["forecast_length"]==745 and proto["scored_labels"]==505
print(json.dumps({"files":[{"name":f.name,"bytes":f.stat().st_size} for f in [folder/"literature_review.json",folder/"research_review.html",folder/"research_review.txt"]],"sources":len(d["sources"]),"dataset_rows":len(d["datasets"]),"hipe_studies":len(d["hipe_studies"]),"benchmark_rows":20},ensure_ascii=False))

