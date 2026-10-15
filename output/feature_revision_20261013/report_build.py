"""Build the completed feature audit and retraining report; no model fitting."""
from pathlib import Path
import base64,html,json,os

OUT=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def esc(x):return html.escape(str(x))
def fmt(x,d=3):return '—' if x is None else f'{float(x):,.{d}f}'
def rel(p):
    p=Path(p)
    if not p.is_absolute():p=OUT/p
    return os.path.relpath(p,OUT).replace('\\','/')
def link(p,label):return f'<a href="{esc(rel(p))}">{esc(label)}</a>'
def fig(p,caption=''):
    p=Path(p)
    if not p.is_absolute():p=OUT/p
    if not p.is_file():raise FileNotFoundError(p)
    return f'<figure><img loading="lazy" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"><figcaption>{esc(caption)}</figcaption></figure>'
def table(head,rows):
    return '<div class="tablewrap"><table><thead><tr>'+''.join(f'<th>{esc(s)}</th>' for s in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{s}</td>' for s in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def details(label,value):return '<details><summary>'+esc(label)+'</summary><pre>'+esc(json.dumps(value,ensure_ascii=False,indent=2))+'</pre></details>'

def main():
    comp=load(OUT/'comparison_results.json')
    analysis=load(OUT/'analysis/analysis_results.json')
    catalog=load(OUT/'data/feature_catalog.json')
    extraction=load(OUT/'data/extraction_verification.json')
    audit=load(OUT/'classical/previous_feature_audit.json')
    protocol=load(OUT/'experiment_protocol.json')
    refs=load(OUT/'method_references.json')
    classical=load(OUT/'classical/results.json')
    css='''*{box-sizing:border-box}body{margin:0;background:#edf3f7;color:#173348;font-family:"Malgun Gothic",sans-serif;line-height:1.75}main{max-width:1200px;margin:auto;padding:28px 22px 65px}header{background:#18374f;color:white;border-radius:14px;padding:30px}h1{font-size:28px;margin:8px 0 13px}h2{font-size:23px;margin:0 0 16px}h3{font-size:17px;margin:20px 0 10px}section{background:white;border:1px solid #d7e2eb;padding:26px;border-radius:12px;margin:22px 0}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:20px}.card{padding:20px;border:1px solid #d7e2eb;border-radius:10px;background:white}.value{font-size:25px;font-weight:700}.small,figcaption{font-size:12px;color:#647a8b}.note{padding:16px;border-left:4px solid #168577;background:#edf8f5;margin:18px 0}.caution{background:#fff3e6;border-color:#cb783a}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px;border-bottom:1px solid #dce5ed;vertical-align:top;text-align:left}th{background:#f0f5f8}img{width:100%;max-width:100%;height:auto}figure{margin:20px 0}a{color:#006b86}.links{display:flex;gap:15px;flex-wrap:wrap}details{background:#f4f8fb;padding:14px;border-radius:8px;margin:15px 0}summary{cursor:pointer;font-weight:bold;color:#067e74}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}select{padding:10px;border:1px solid #a9bac5;border-radius:6px;font-family:inherit;max-width:100%;font-size:14px}.controls{display:flex;flex-wrap:wrap;gap:18px;margin:15px 0}.gallery-img{min-height:100px}footer{font-size:12px;color:#647a8b}@media(max-width:780px){main{padding:15px 10px}header,section{padding:21px 16px}.cards{grid-template-columns:1fr}h1{font-size:23px}}'''
    body=['<header><div>2026-10-13 · 피처 재검증 및 모델 재학습</div><h1>센서 피처가 미래 전력량을 설명하는가</h1><p>상관관계·시간 지연·검증 중요도를 점검하고, 설비별 입력과 재선정 피처로 예측 모델을 다시 만들었습니다.</p></header>']
    body.append('<div class="cards"><div class="card"><div class="value">13설비 · 442피처</div><div class="small">원본 전체 재집계 · 3,600시간</div></div><div class="card"><div class="value">각 학습 구간에서 선택</div><div class="small">미래 센서·정답 파생 열 제외</div></div><div class="card"><div class="value">기준선과 함께 비교</div><div class="small">24시간 경로 및 lead1·6·24</div></div></div>')
    conclusions=comp.get('conclusions',[])
    if conclusions:body.append('<section><h2>결과에서 확인한 점</h2>'+''.join('<p>'+esc(x)+'</p>' for x in conclusions)+'</section>')
    body.append('<section><h2>기존 피처와 데이터 가용성 검증</h2><p>기존 XGBoost는 합계 전력량의 lag1·24·168, rolling mean24·168·720, 표준편차와 원점·예측 시간의 달력 정보를 사용했습니다. 기존 신경망은 합계 전력량 한 변수만 입력했으므로 설비 간 관계를 충분히 활용하지 못했습니다.</p><p>기존 코드에서 미래 센서가 입력에 섞이거나 정답 시간이 어긋난 오류는 발견되지 않았습니다. 원점 기준 고정 lag24와 각 예측 대상 시간에 대응하는 전날·전주 값은 다른 피처이므로 이번 실험에서 구분했습니다.</p><div class="note caution">같은 시간의 설비 전력량 합은 목표값 자체를 구성합니다. 같은 시간의 높은 상관계수를 근거로 미래 예측 입력에 넣으면 정답 누출이 됩니다. 모든 센서 피처는 예측 원점 이전에 완료된 시간에서만 가져옵니다. 5월 파일의 요금·탄소·전체 합계·hourly_pow는 입력으로 사용하지 않습니다.</div>'+table(['재검증 항목','결과'],[
        ['원본 처리',f'{extraction["rows"]:,}행 · 완전한 설비-시간 {extraction["complete_module_hours"]:,}개'],
        ['완전 시간별 표본 수',f'{extraction["counts_min"]}~{extraction["counts_max"]}개 / 설비'],
        ['새 설비 합계와 기존 목표값 차이',fmt(extraction['energy_sum_vs_existing_target_max_absdiff'],10)],
        ['후보 제외','상수 operation, 누적 카운터 절대값, 잘못된 Unix 시간키, 미래 정답 파생 열'],
        ['단위·계측 범위','가이드 단위를 따른 잠정 kWh; 계측기 계층과 물리 단위 확인 필요']
    ])+'<p class="links">'+link(OUT/'classical/previous_feature_audit.json','기존 코드 시간 정렬 감사')+link(OUT/'data/feature_catalog.json','442개 피처 정의·가용시점')+link(OUT/'data/extraction_verification.json','전체 재집계 검증')+'</p>'+details('기존 피처 감사 상세',audit)+'</section>')
    body.append('<section><h2>상관계수와 시간 관계성</h2><p>동시 상관과 미래 목표와의 지연 상관을 분리하고, Pearson·Spearman 계수, 다중검정 보정, 앞·뒤 학습 구간의 안정성을 확인했습니다. 자기상관·주기·요일 효과와 피처 중복도 함께 분석했습니다. 날짜 블록을 섞은 대조 결과는 여러 피처 중 큰 상관이 우연히 나타날 가능성을 확인하는 참고 자료입니다.</p>')
    sh=analysis['shuffle']
    body.append('<div class="note">공통 '+str(sh['hours'])+'시간의 '+str(sh['tested_pairs'])+'개 피처·지연 조합에서 최대 |r|='+fmt(sh['observed_max_abs_r'],4)+', 날짜 블록 대조의 최대통계량 p='+fmt(sh['max_stat_empirical_p'],3)+'였습니다. 이 비교에서는 우연 수준을 넘어서는 안정적인 지연 상관을 확인하지 못했습니다. 비선형 관계나 아직 관측하지 못한 외생 요인의 부재를 증명하는 결과는 아닙니다.</div>')
    corrrows=[]
    for item in analysis['top_lagged'][:10]:
        corrrows.append([esc(item['feature']),str(item['lag_hours']),fmt(item['r'],4),fmt(item['q_BH_AR1'],4)])
    body.append(table(['과거 피처','지연 시간','Pearson r','BH 보정 q'],corrrows)+'<p class="links">'+link(OUT/'analysis/lag_correlations.csv','전체 피처 Pearson·Spearman 계수')+link(OUT/'analysis/correlation_stability.csv','구간별 상관 안정성')+link(OUT/'analysis/importance_individual_ridge.csv','개별 피처 중요도')+link(OUT/'analysis/importance_grouped.csv','그룹 중요도·제외 실험')+'</p>')
    plots=analysis.get('plots',{})
    plot_items=[(item['name'],item) for item in plots] if isinstance(plots,list) else list(plots.items())
    for key,item in plot_items:
        p=item.get('png') if isinstance(item,dict) else item
        if p:body.append('<h3>'+esc(item.get('title',key) if isinstance(item,dict) else key)+'</h3>'+fig(p,item.get('caption','') if isinstance(item,dict) else '')+(link(item['svg'],'분석 그래프 SVG 열기') if isinstance(item,dict) and item.get('svg') else ''))
    body.append('<p>낮은 검증 성능 모델의 작은 중요도가 피처의 고유 정보량이 없음을 증명하지는 않습니다. 서로 상관된 피처는 대체 정보를 제공할 수 있으므로 그룹별 제외 실험과 검증 중요도를 함께 판단했습니다.</p>'+details('상관·시간·중요도 분석 전체 결과',analysis)+'</section>')
    body.append('<section><h2>재학습 모델의 피처 중요도</h2><p>새 모델을 3월24일까지의 이력으로 학습하고 3월25~31일의 7개 일일 원점에서 피처를 섞어 경로 오차 변화를 측정했습니다. 이 진단으로 4월 점수를 보거나 모델 설정을 다시 선택하지 않았습니다. 학습 gain·표준화 계수와 검증 permutation 결과는 서로 다른 의미의 지표입니다.</p>')
    for name in ['xgboost','ridge','elasticnet']:
        m=classical['models'][name];d=OUT/'classical'/name
        body.append('<h3>'+esc(name)+'</h3>'+fig(m['plots']['importance_png'],'시간 순서의 내부 검증 중요도 · 센서 묶음과 원점 단위 순열')+'<p class="links">'+link(d/'chronological_group_permutation.csv','피처 그룹별 검증 중요도')+link(d/'chronological_feature_permutation.csv','학습 순위 상위 20개 피처의 검증 중요도')+link(d/('gain_final.csv' if name=='xgboost' else 'coefficients_final.csv'),'전체 최종 학습 gain·계수')+'</p>')
    body.append('<p>입력으로 제공한 모든 피처가 모델에서 실제로 사용된 것은 아닙니다. 높은 학습 gain만으로 미래 예측에 도움이 된다고 판단하지 않았습니다.</p></section>')
    body.append('<section><h2>재선정 피처와 학습·평가</h2><p>각 expanding fold의 학습 prefix에서만 센서 상관·안정성·중복 조건으로 후보를 선정했습니다. 선택된 이름과 정규화 통계는 해당 검증 기간에 고정했습니다. 4월 평가의 피처 선택은 3월31일 이전 자료로 고정하고, 최종 저장 모델의 가중치만 4월29일까지 다시 학습했습니다.</p><p>24시간 모델은 기존과 같은 5개 fold·70개 일일 원점에서 설정을 선택하고 4월29개 원점의 696개 값으로 평가했습니다. 향후 24시간 경로와 1·6·24시간 뒤의 단일 값 오차를 별도로 보고합니다. 신경망은 설비별 다변량 입력을 사용합니다.</p>')
    selectionrows=[]
    for item in analysis['selectorcounts']:
        selectionrows.append([esc(item['fold']),esc(item['origin']),str(item['training_rows']),str(item['selected_count']),esc(', '.join(item['selected_columns']) or '추가 센서 없음')])
    body.append(table(['선택 구간','예측 시작','학습 시간 수','선택 센서 수','선택 이름'],selectionrows)+'<p>일부 짧은 학습 구간에서 통과한 센서는 뒤 구간까지 유지되지 않았습니다. 3월 말 기준으로 추가 물리 센서 선택은 0개입니다. 그렇다고 센서를 미리 모두 제외하고 결론을 정하지는 않았으며, 전체 센서 입력과 설비 전력량 입력을 별도 후보로 학습해 검증했습니다.</p>')
    body.append('<p class="small">선정 조건: 과거 센서와 미래 목표의 |r|≥0.08, BH 보정 q≤0.05, 앞·뒤 학습 구간에서 같은 부호와 최소 효과 크기 유지, 이미 선택한 피처와 |상관|≤0.95. 후보가 없을 수 있으며, 약한 비선형 신호를 놓칠 가능성은 전체 센서 모델의 비교와 함께 판단했습니다.</p>')
    feature_rows=[]
    for key,m in comp['scenarios']['dayahead24']['models'].items():
        if m.get('previous_control'):continue
        descriptions={'all_sensor_calendar':'과거 합계·설비13 전력량·센서403 + 대상 시간에 맞춘 전날 값·달력',
            'selected_sensor_calendar':'과거 합계·설비13 전력량 + 학습 구간에서 선정한 센서·달력',
            'calendar':'시간대·요일·주말의 알려진 달력 + ARIMA 상태',
            'ema720':'합계 전력량의 EMA(720시간)'}
        desc=descriptions.get(m.get('input_description',''),m.get('input_description',''))
        note=m.get('feature_selection_note','');count=m.get('input_feature_count','—')
        if m['model_family']=='itransformer':desc='설비13 전력량 + 합계 목표의 14변수';note='변수 토큰 사이 attention · 합계 채널만 감독 · 최종 추가 센서 0개'
        elif m['model_family']=='patchtst':desc='설비13 전력량 + 합계 목표의 14변수';note='채널별 공유 파라미터 학습 · 13설비 예측만 합산 · 추론 시 채널 사이 attention 없음'
        elif m['model_family']=='baseline':count=1;note='합계 전력량의 과거 이력에서 계산'
        feature_rows.append([esc(m['display_name']),esc(desc),str(count),esc(note)])
    body.append(table(['모델','입력 구조','피처 수','선정·학습 방식'],feature_rows)+'<p class="small">신경망은 최대 16epoch의 소형 후보 탐색이며 일부 내부 검증의 최저 epoch가 상한에 도달했습니다. 수렴이나 전역 최적을 보장하지 않습니다. 이번 결과로 피처의 모든 비선형 정보가 없다고 단정하지 않았습니다.</p><p class="links">'+link(OUT/'neural/results.json','신경망 학습·정규화·출력·CLI 상세')+link(OUT/'classical/feature_group_ablation.csv','피처 그룹별 후보 검증 결과')+'</p>'+details('시간 분리와 누출 방지 프로토콜',protocol)+'</section>')
    for scenario,section_title in [('dayahead24','24시간 모델의 재검증 결과'),('long745','장기 예측의 별도 검증')]:
        if scenario not in comp['scenarios']:continue
        s=comp['scenarios'][scenario];rows=[]
        for key,m in s['models'].items():
            v=m['validation_metrics'];t=m['test_metrics'];e=m.get('endpoint_metrics',{})
            rows.append([esc(m['display_name']),fmt(v['mae_kwh']),fmt(t['mae_kwh']),fmt(t['rmse_kwh']),fmt(t['r2'],5),fmt(e.get('24',{}).get('mae_kwh')),fmt(m.get('previous_test_mae_kwh'))])
        body.append('<section><h2>'+section_title+'</h2>'+table(['모델','검증 MAE','평가 MAE','평가 RMSE','R²','lead24 MAE','기존 동일모델 MAE'],rows)+'<p>검증 기준 선택: <strong>'+esc(s['models'][s['selected_model']]['display_name'])+'</strong>. 테스트 점수로 피처나 설정을 다시 선택하지 않았습니다.</p>')
        if scenario=='dayahead24':body.append('<div class="note caution">새 XGBoost의 4월 R²는 약 0.00091로 관측 변동의 설명력이 거의 없습니다. 기존 Chronos 혼합과의 MAE 차이(새 모델−기존)는 -0.0474 kWh, 날짜별 paired bootstrap 95% 구간은 [-0.2896, +0.1897]로 0을 포함합니다. 이번 피처 추가가 예측 품질을 확실하게 개선했다고 주장할 근거는 부족합니다.</div>')
        if scenario=='long745':body.append('<p>5월에는 미래 설비 센서와 중간 관측 이력이 없습니다. 4월30일 단일 원점에서 745시간 경로를 생성하고 5월10일~31일의 505개 정답만 비교합니다. 매일 갱신하는 24시간 모델의 현장 평가와 구분합니다.</p>')
        body.append(details('모델별 최적 설정·피처·지표',s)+'</section>')
    figures=comp.get('figures',{})
    if figures:
        options=''.join(f'<option value="{esc(k)}">{esc(v["label"])}</option>' for k,v in figures.items())
        encoded={k:{'label':v['label'],'png':'data:image/png;base64,'+base64.b64encode(Path(v['png']).read_bytes()).decode(),'svg':rel(v['svg'])} for k,v in figures.items()}
        first=next(iter(encoded.values()))
        body.append('<section><h2>모델별 실측과 예측</h2><p>동일한 시간의 실측·예측과 잔차를 확인할 수 있습니다. 그래프의 세로축을 확대했으며 오차와 변동 설명력을 함께 판단해야 합니다.</p><div class="controls"><label>모델·업무 선택 <select id="prediction-model">'+options+'</select></label></div><figure><img id="prediction-figure" class="gallery-img" src="'+first['png']+'"><figcaption id="prediction-caption">'+esc(first['label'])+'</figcaption></figure><a id="prediction-svg" href="'+esc(first['svg'])+'">선택한 그래프 SVG 열기</a></section>')
        script='<script>const predictionFigures='+json.dumps(encoded,ensure_ascii=False)+';document.getElementById("prediction-model").addEventListener("change",function(){const item=predictionFigures[this.value];document.getElementById("prediction-figure").src=item.png;document.getElementById("prediction-caption").textContent=item.label;document.getElementById("prediction-svg").href=item.svg;});</script>'
    else:script=''
    body.append('<section><h2>해석과 실행 자료</h2><p>MAE의 절대 크기뿐 아니라 단순 기준선 대비 개선과 R²를 확인해야 합니다. 평평한 예측이 관측 변동을 설명하지 못하면 낮은 MAPE만으로 높은 예측 정확도를 주장할 수 없습니다. 피처를 늘려도 미래에 재현되는 신호가 부족하면 성능이 좋아지지 않을 수 있습니다.</p><p>4월·5월은 이전 분석에서 확인한 기간이므로 이번 검증은 완전히 미관측인 prospective test가 아닙니다. 실제 운영 성능은 추가 기간과 예측 시점에 알 수 있는 생산 계획·부하 요인으로 검증해야 합니다.</p><p class="links">'+link(OUT/'README.txt','실행·산출물 안내')+link(OUT/'comparison_results.json','모델 비교 JSON')+link(OUT/'verification_results.json','최종 검증 결과')+'</p>'+''.join('<p><a href="'+esc(s['url'])+'">'+esc(s['title'])+'</a> · '+esc(s['supports'])+'</p>' for s in refs['sources'])+'</section><footer>원본 및 기존 EDA·학습 산출물은 보존했습니다. 이번 재집계·피처 분석·저장 모델·예측·그림은 별도 폴더에 저장했습니다.</footer>')
    page='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>피처 재검증과 예측 모델 재학습</title><style>'+css+'</style></head><body><main>'+''.join(body)+'</main>'+script+'</body></html>'
    (OUT/'feature_revision_report.html').write_text(page,encoding='utf8')
    print('REPORT_BYTES',len(page.encode('utf8')))
if __name__=='__main__':main()
