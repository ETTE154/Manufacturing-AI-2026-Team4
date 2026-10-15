"""Create offline research/model comparison report from verified experiment files."""
from pathlib import Path
import base64,html,json,os,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def esc(value):return html.escape(str(value))
def fmt(value,digits=3):
    if value is None:return '미확인'
    return f'{float(value):,.{digits}f}'
def rel(path):
    p=Path(path)
    if not p.is_absolute():p=OUT/p
    return os.path.relpath(p,OUT).replace('\\','/')
def link(path,label):return f'<a href="{esc(rel(path))}">{esc(label)}</a>'
def embed(path,caption=''):
    p=Path(path)
    if not p.is_absolute():p=OUT/p
    if not p.exists():raise FileNotFoundError(p)
    return f'<figure><img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" loading="lazy"><figcaption>{esc(caption)}</figcaption></figure>'
def table(headers,rows):
    return '<div class="tablewrap"><table><thead><tr>'+''.join(f'<th>{esc(h)}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{c}</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def main():
    audit=load(OUT/'source_audit/audit_report.json')
    source=load(OUT/'source_audit/byte_comparison.json')
    research=load(OUT/'research/presentation_basis.json')
    signal=load(OUT/'research/forecastability_audit.json')
    comp=load(OUT/'forecast/comparison_results.json')
    anomaly=load(OUT/'anomaly/anomaly_metrics.json')
    uncertainty=load(OUT/'research/paired_uncertainty.json') if (OUT/'research/paired_uncertainty.json').exists() else None
    old=load(ROOT/'output/modeling/forecast/comparison_results.json')
    model_refs=load(OUT/'research/model_references.json')
    if not source['all_byte_identical']:raise ValueError('Official and provided source differ; reconcile before reporting')
    css='''*{box-sizing:border-box}body{margin:0;background:#eff3f7;color:#173047;font-family:"Malgun Gothic",sans-serif;line-height:1.75}main{max-width:1180px;margin:auto;padding:28px 22px 70px}header{padding:32px;background:#18364e;color:white;border-radius:15px}h1{font-size:28px;margin:6px 0 14px}h2{font-size:22px;margin:0 0 17px}h3{font-size:17px;margin:22px 0 10px}section{padding:27px;background:white;border:1px solid #d9e3ec;border-radius:12px;margin:22px 0}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:18px}.card{padding:18px;background:white;border:1px solid #d9e3ec;border-radius:10px}.value{font-size:25px;font-weight:bold}.small,figcaption{font-size:12px;color:#607487}.note{background:#edf8f6;border-left:4px solid #148277;padding:16px;margin:18px 0}.caution{background:#fff4e8;border-color:#cc713c}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px;border-bottom:1px solid #dce5ee;text-align:left;vertical-align:top}th{background:#f1f5f9}img{max-width:100%;width:100%;height:auto}figure{margin:20px 0}a{color:#006a85}code,pre{background:#f1f5f9;font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:15px;border-radius:8px}details{background:#f8fafc;padding:14px;margin:15px 0;border-radius:8px}summary{cursor:pointer;color:#087477;font-weight:bold}.gallery{display:grid;grid-template-columns:1fr 1fr;gap:18px}.gallery figure{margin:0}.links{display:flex;flex-wrap:wrap;gap:15px}footer{font-size:12px;color:#607487}@media(max-width:780px){main{padding:16px 10px}header,section{padding:22px 16px}.cards,.gallery{grid-template-columns:1fr}h1{font-size:23px}}'''
    content=[]
    content.append('<header><div>2026-10-13 · 공개 원본 확인 및 추가 최적화</div><h1>전력 예측 최적화와 이상 후보 탐지</h1><p>공개 데이터 직접 다운로드 비교, 다음날 24시간 예측과 5월 장기 과제 평가, 연속 센서 구간의 모델별 탐지 결과를 정리했습니다.</p></header>')
    scenarios=comp['scenarios'];day=scenarios['dayahead24'];long=scenarios['long745']
    content.append(f'<div class="cards"><div class="card"><div class="value">공식 CSV 2개 동일</div><div class="small">직접 다운로드·크기·SHA-256 비교</div></div><div class="card"><div class="value">24시간 / 745시간</div><div class="small">업무 목적과 과제 평가를 구분</div></div><div class="card"><div class="value">검증 후 설정 고정</div><div class="small">4월·5월로 하이퍼파라미터 재선택 안 함</div></div></div>')
    rows=[]
    for item in source['files']:
        rows.append([esc(item['source_key']),f'{item["original"]["bytes"]:,}',f'{item["downloaded"]["bytes"]:,}',esc(item['original']['sha256']), '완전 동일' if item['sha256_match'] else '차이'])
    content.append('<section><h2>공식 다운로드와 제공 파일 비교</h2><p>사용자가 제공한 사이트에서 공개 목록을 확인하고 학습·5월 CSV를 실제로 내려받았습니다. 두 파일은 제공된 파일과 바이트가 완전히 같습니다. 발견된 해석·품질 문제는 다운로드나 복사 과정에서 생긴 손상이 아닙니다.</p>'+table(['파일','제공 bytes','다운로드 bytes','SHA-256','결론'],rows)+'<p class="links">'+link(OUT/'source_audit/audit_report.html','원본 비교 상세 보고서')+link(OUT/'source_audit/source_manifest.json','공개 목록·다운로드 출처')+'</p><div class="note caution">같은 파일이라는 사실은 모든 측정값이 물리적으로 정확하다는 뜻이 아닙니다. Unix 시간값과 localtime의 불일치, 전력·누적 에너지 단위, 상하위 계측 중복 가능성, 실제 삽입 정답 부재는 별도로 남아 있습니다. 원본과 기존 분석 결과는 보존했습니다.</div></section>')
    rows=[]
    for s in research['sources'][:3]:
        rows.append([esc(s['authors'])+' · '+str(s['year']),f'<a href="{esc(s["url"])}">{esc(s["title"])}</a>',esc(s['supports']),esc(s['transfer_limit'])])
    content.append('<section><h2>발표의 문제 정의와 예측 길이 근거</h2><div class="note">'+esc(research['presentation_script'])+'</div>'+table(['저자·연도','원문','설정 근거','현재 실험과 차이'],rows)+'<p>'+esc(research['reason_for_horizon'])+'</p><p>1 step은 1시간입니다. 업무 모델은 마지막 관측에서 1~24시간 뒤의 시간 전력량 24개를 예측하고 24시간 뒤 단일 값의 오차도 보고합니다. 과제 모델은 마지막 4/29 23시 관측 뒤 745개를 생성해 5/10~5/31의 505개 정답만 평가합니다.</p><div class="note caution">'+esc(research['may_limitation'])+'</div><p>'+link(OUT/'research/presentation_notes.txt','발표 문장·근거 TXT')+'</p></section>')
    content.append('<section><h2>전처리와 시간 순서 검증</h2><p>5초 원본의 연속성이 확인된 localtime을 시간 키로 사용했습니다. 설비별 한 시간에 720개 관측이 있고 13개 설비 모두 완전한 시간만 남겼습니다. 전력량은 가이드의 kW 표기에 따라 Σ activePower × 5/3600으로 계산하고 설비 합계를 만들었습니다. 4월30일 00시의 불완전한 마지막 시간은 제외하여 3,600시간을 사용했습니다. 단위와 계측기 계층이 확인되기 전까지 이 합계는 잠정 값입니다.</p><p>예측에는 과거 합계 전력량을 공통 입력으로 사용합니다. XGBoost는 과거 통계와 예측 시점에 이미 알려진 달력 정보를 후보로 비교했습니다. 평가 파일의 요금·탄소·전체 합계처럼 정답에서 파생된 열은 입력에서 제외했습니다. 신경망의 정규화 통계와 이상 탐지의 RobustScaler는 각 학습 구간에서만 추정했습니다. 결측 보간이나 원본 이상 후보 제거로 평가 값을 바꾸지 않았습니다.</p><p>24시간 업무 모델은 5개 expanding fold의 70개 일일 원점으로 설정을 선택하고, 4월의 29개 원점 696개 값으로 평가했습니다. 가중치는 각 fold 시작에 학습하고, 매일 입력 이력은 그날 원점 이전의 실측만 갱신합니다. 장기 모델은 2월25일 원점의 745시간 검증으로 설정을 고정하고, 3월29일 원점의 내부 평가와 5월 정답 평가를 분리했습니다.</p><p class="small">iTransformer는 이번 공통 입력에서 변수 토큰이 하나라 다변량 상호작용의 장점을 검증하지 못합니다. PatchTST·iTransformer의 소형 최적화 설정은 공개 벤치마크 결과의 재현이나 현재 절대 SOTA를 입증하는 실험이 아닙니다.</p></section>')
    content.append('<section><h2>유사 공개 데이터와 모델 선정 근거</h2><p>산업 설비별 스마트미터라는 구조가 유사한 공개 데이터는 KIT의 HIPE입니다. 실제 공장의 10대 설비와 주 계측을 포함하므로 계측기 계층과 생산 활동을 함께 고려하는 데 참고가 됩니다. 현재 데이터가 HIPE의 복사본이거나 같은 분포라는 뜻은 아닙니다. 이번 모델 최적화는 제공 데이터에서 수행했고, HIPE에 대한 벤치마크 재학습은 하지 않았습니다.</p>'+table(['근거','적용 이유','이 프로젝트의 제한'],[[f'<a href="{esc(s["url"])}">{esc(s["title"])}</a>',esc(s['supports']),esc(s['limit'])] for s in model_refs['sources']])+'</section>')
    rows=[]
    for name,scenario,metric_key in [('24시간 업무 예측',day,'test_metrics'),('745시간 과제 예측',long,'may_metrics')]:
        for key,item in scenario['models'].items():
            m=item[metric_key];v=item['validation_metrics']
            rows.append([esc(name),esc(item.get('display_name',key)),fmt(v.get('mae_kwh')),fmt(m.get('mae_kwh')),fmt(m.get('rmse_kwh')),fmt(m.get('mape_percent')),fmt(m.get('r2')),str(item.get('trials_count','미확인'))])
    content.append('<section><h2>최적화 결과와 남은 불확실성</h2>'+table(['실험','모델','검증 MAE','테스트 MAE','테스트 RMSE','MAPE %','R²','후보 설정 수'],rows)+'<p class="small">전력량 단위는 가이드와 정답의 단위·집계 범위가 일치한다는 가정하의 잠정 kWh입니다. 후보 설정 수와 실제 fit 수는 다릅니다. 여러 시간 원점으로 검증했어도 검증 예산 내의 최선이며 전역 최적해를 보장하지 않습니다.</p><p>24시간 검증 선택: <strong>'+esc(day['selected_model'])+'</strong>. 장기 검증 선택: <strong>'+esc(long['selected_model'])+'</strong>. 테스트 점수가 더 낮은 모델이 나오더라도 테스트로 다시 선택하지 않습니다.</p><details><summary>목적별 상세 지표·최적 설정·선택 기록</summary><pre>'+esc(json.dumps(comp,ensure_ascii=False,indent=2))+'</pre></details></section>')
    endpoint_rows=[]
    for key,item in day['models'].items():
        endpoint_rows.append([esc(item['display_name']),fmt(item['validation_endpoint_metrics']['mae_kwh']),fmt(item['test_endpoint_metrics']['mae_kwh']),str(item['test_endpoint_metrics']['count'])])
    content.append('<section><h2>24시간 뒤 단일 값과 최적화 범위</h2>'+table(['모델','lead24 검증 MAE','lead24 4월 MAE','4월 원점 수'],endpoint_rows)+'<p>ARIMA는 13개 차수, XGBoost는 깊이·나무 수·규제·잔차 축소·손실을 포함한 31개 설정, PatchTST·iTransformer는 각각 입력 길이·크기·층 수·dropout·MSE/L1 손실의 7개 설정을 비교했습니다. Chronos-2는 사전학습 가중치를 갱신하지 않고 입력 길이·평균 보정·혼합 비중 27개 설정을 비교했습니다. 기준선 17개 설정도 같은 검증으로 선택했습니다. 두 목적을 합쳐 총 204개 후보 설정입니다.</p><p>24시간 선택은 context 168시간의 Chronos-2 예측 25%와 과거 평균 75%를 혼합한 모델입니다. 4월 경로 MAE는 '+fmt(day['models']['chronos2']['test_metrics']['mae_kwh'])+'이고 EMA 기준선 '+fmt(day['models']['baseline']['test_metrics']['mae_kwh'])+' 대비 개선은 '+fmt(100*(1-day['models']['chronos2']['test_metrics']['mae_kwh']/day['models']['baseline']['test_metrics']['mae_kwh']),3)+'%뿐입니다. 24시간 뒤 단일 값의 MAE는 '+fmt(day['models']['chronos2']['test_endpoint_metrics']['mae_kwh'])+'입니다. 장기 선택은 전체 과거를 입력한 원래 Chronos-2입니다.</p>')
    oldrows=[]
    for key,item in long['models'].items():
        olditem=old['models'].get(key)
        if olditem and 'may' in olditem:
            a=olditem['may']['P505']['mae_kwh'];b=item['may_metrics']['mae_kwh']
            oldrows.append([esc(item['display_name']),fmt(a),fmt(b),fmt(a-b)])
    content.append('<h3>기존 실험과 새 실험의 5월 관찰 비교</h3>'+table(['모델','기존 5월 MAE','새 5월 MAE','기존−새 MAE'],oldrows)+'<p>검증 기준의 최적화가 모든 미사용 기간의 개선을 보장하지는 않습니다. 새 5월의 관찰 최저 MAE는 XGBoost '+fmt(long['models']['xgboost']['may_metrics']['mae_kwh'])+'이지만, 기존 XGBoost의 5월 MAE보다 낮아지지는 않았습니다. 기존 iTransformer는 설비별 다변량, 새 실험은 공통 단변량 입력이므로 설정 변경만의 효과로 해석할 수 없습니다. 장기 모델 전체의 5월 R²는 음수입니다. 작은 MAPE를 100−MAPE의 예측 정확도로 표현하면 안 됩니다.</p>')
    if uncertainty:
        urows=[]
        for key,item in uncertainty['comparisons'].items():
            lo,hi=item['uncertainty']['paired_bootstrap_95_interval_kwh']
            urows.append([esc(key),fmt(item['baseline_minus_model_mae_kwh'],4),f'[{fmt(lo,4)}, {fmt(hi,4)}]'])
        content.append('<h3>기준선 대비 개선은 얼마나 확실한가</h3>'+table(['비교','기준선−모델 MAE','조건부 95% 구간'],urows)+'<p>양수는 모델의 개선입니다. 4월은 29개 일일 원점, 5월은 하루 블록을 짝지어 10,000회 재표집했습니다. 4월의 개선 구간은 양수이지만 효과는 약 0.16%로 매우 작습니다. 5월 선택 Chronos는 평균 기준선보다 나쁩니다. 5월 XGBoost 구간은 한 장기 예측 경로에 조건부인 사후 관찰이며, 다른 월로의 일반화·모델 선택 불확실성·사후 순위 선택 편향을 반영하지 않습니다.</p><details><summary>날짜 단위 paired bootstrap 상세</summary><pre>'+esc(json.dumps(uncertainty,ensure_ascii=False,indent=2))+'</pre></details>')
    content.append('</section>')
    content.append('<section><h2>모델별 5월 실측과 예측</h2><p>모든 모델은 동일한 4월30일 원점에서 미래 745시간을 생성했습니다. 아래 그림은 정답이 있는 505시간만 비교합니다. 센서의 미래 실측으로 중간 경로를 채우지 않았습니다.</p>')
    for key,figs in comp.get('figures',{}).items():
        p=figs.get('may_long_png')
        if p:content.append('<h3>'+esc(key)+'</h3>'+embed(p,'5월 실측·예측·잔차 · 세로축 확대와 전체 규모를 함께 확인')+link(figs.get('may_long_svg',p),'개별 그래프 열기'))
    content.append('</section><section><h2>모델별 다음날 24시간 예측</h2><p>4월의 매일 예측 원점에서 그 전까지 관측한 이력만 사용합니다. 전체 24시간 경로와 lead24의 단일 값 결과를 구분합니다. 5월 장기 그래프와 평가 조건이 다릅니다.</p>')
    for key,figs in comp.get('figures',{}).items():
        p=figs.get('april_day_png')
        if p:content.append('<h3>'+esc(key)+'</h3>'+embed(p,'4월 29개 일일 원점의 24시간 예측')+link(figs.get('april_day_svg',p),'개별 그래프 열기'))
    content.append('</section>')
    content.append('<section><h2>데이터에 예측 신호가 충분한가</h2><p>4월 평가 전인 12~3월 2,904시간에서 변동계수는 '+fmt(signal['cv_percent'])+'%였습니다. 24시간 자기상관은 '+fmt(signal['acf_selected']['24'],4)+', 168시간 자기상관은 '+fmt(signal['acf_selected']['168'],4)+'입니다. Ljung–Box 검정은 이 범위에서 뚜렷한 자기상관을 검출하지 못했습니다. 이것은 독립성의 증명이 아니며, 단순 기준선 대비 테스트 개선 여부와 함께 판단해야 합니다.</p>'+embed(OUT/'research/forecastability_audit.png','시간 신호와 조건부 전력 단위 진단')+'<div class="note caution">같은 회로의 상전압·전류·역률이라는 가정 아래 위상 전력 공식으로 계산하면, 가이드의 kW 표기를 적용한 activePower 비율의 표본 중앙값은 '+fmt(signal['phase_power_diagnostic']['reported_activePower_if_guide_kw_ratio_quantiles']['median'])+'배입니다. 이는 단위·배율·계측 범위를 공급자에게 확인할 근거이며, 임의로 1,000으로 나누어 수정하지 않았습니다. 16,849행의 체계 표본에 대한 조건부 진단입니다.</div></section>')
    content.append('<section><h2>원본 이상 후보와 합성 이상 실험</h2><p>원본의 삽입 정답 마스크는 공개 목록에도 없습니다. 원본 센서 규칙 일치도와 새로 주입한 합성 정답의 성능을 분리합니다. 합성 테스트를 잘 탐지해도 실제 고장 성능을 입증하지 않습니다.</p><p class="links">'+link(OUT/'anomaly/anomaly_report.html','모델별 연속 구간 탐지·합성 평가 갤러리')+link(OUT/'anomaly/anomaly_metrics.json','진단 지표·주입 설계')+'</p>')
    # The dedicated anomaly report owns its detailed model metrics and all figures.
    models=anomaly.get('models',{});arows=[]
    for key,item in models.items():
        met=item.get('original_weak_agreement',{});syn=item.get('synthetic_heldout_metrics',{})
        arows.append([esc(item.get('display_name',key)),fmt(met.get('average_precision')),fmt(met.get('precision')),fmt(met.get('recall')),fmt(syn.get('precision')),fmt(syn.get('recall'))])
    if arows:content.append(table(['모델','원본 규칙 AP','원본 규칙 정밀도','원본 규칙 재현율','합성 정밀도','합성 재현율'],arows))
    content.append('<p>원본 규칙 후보는 63,963행(전체의 약 0.190%)이며, 3호기의 저전압과 예비건조기·6호기의 저역률 구간에 집중됩니다. 연속 후보 사건의 99.14%가 5초 관측 하나입니다. 이것은 원본 삽입 고장의 정답 개수가 아닙니다. 원본 연속 5초 구간을 추출하여 센서 값, 각 모델의 점수·임계값·경보를 개별 그림으로 저장했습니다. rolling median으로 신호를 평활화하면 중요한 후보를 지울 수 있습니다.</p><p>candidate-free 정상 대리 구간에 spike·dip·drift·level shift를 주입했습니다. 학습 624개, 검증 312개, 테스트 312개 합성 사건을 시간으로 분리했습니다. 테스트 112,320행 중 주입된 3,978행의 마스크를 정답으로 사용했습니다. 증강 XGBoost와 Denoising AE는 학습 증강만 사용하고, 임계값은 3월 원본의 정상 대리 구간에서 고정했습니다. Denoising AE는 손상 입력을 주입 전 같은 관측의 값으로 복원하도록 학습했습니다.</p><p>증강 Denoising AE의 합성 점별 재현율은 31.47%로 기존 비지도 AE의 29.94%보다 소폭 높습니다. 원본의 규칙 후보 정밀도는 약 22.18%이므로 이 값만으로 실제 고장 진단을 신뢰할 수 없습니다. 현재는 전기 품질 규칙으로 명확한 후보를 알리고 AE 점수를 추가 검토 신호로 병행하는 구성이 합리적입니다. 실제 고장 분류기를 확정하려면 정비 이력과 고장 정답이 필요합니다.</p><details><summary>이상 탐지 프로토콜과 합성 데이터 상세</summary><pre>'+esc(json.dumps(anomaly,ensure_ascii=False,indent=2))+'</pre></details></section>')
    refs=''.join('<p><a href="'+esc(s['url'])+'">'+esc(s['title'])+'</a> · '+esc(s.get('supports',''))+'</p>' for s in research['sources'][3:])
    content.append('<section><h2>실행과 검증 자료</h2><p class="links">'+link(OUT/'README.txt','실행·산출물 안내')+link(OUT/'verification_results.json','최종 검증 결과')+link(OUT/'forecast/comparison_results.json','최적화 결과 JSON')+'</p><p>모델 선택에 4월·5월 점수를 사용하지 않았습니다. 다만 이전 작업에서 이 기간의 데이터를 분석한 이력이 있어 완전히 미관측인 prospective 검증은 아닙니다. 실제 운영 전에는 추가 기간과 실제 고장 정답으로 검증해야 합니다.</p>'+refs+'</section><footer>원본 가이드·CSV와 기존 분석·학습 산출물은 보존했습니다. 공개 CSV 사본과 이번 추가 실험은 별도 폴더에 저장했습니다.</footer>')
    page='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>공개 원본 감사와 모델 최적화</title><style>'+css+'</style></head><body><main>'+''.join(content)+'</main></body></html>'
    (OUT/'optimization_report.html').write_text(page,encoding='utf-8')
    (OUT/'report_build_metadata.json').write_text(json.dumps({'images':page.count('data:image/png;base64'),'scenarios':list(scenarios),'all_official_files_identical':source['all_byte_identical'],'report_bytes':len(page.encode())},indent=2),encoding='utf-8')
    print('Report built:',OUT/'optimization_report.html')

if __name__=='__main__':main()
