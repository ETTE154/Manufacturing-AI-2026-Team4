"""Build self-contained Korean anomaly report and per-model true-window figures."""
from pathlib import Path
import argparse, base64, html, io, json, os, sys
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'output'/'analysis'/'.plot_deps'))
os.environ['MPLCONFIGDIR']=str(OUT/'tmp'/'mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,dates as mdates
from matplotlib.ticker import FuncFormatter,NullFormatter
font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,
                     'figure.facecolor':'#f3f6fa','axes.spines.top':False,'axes.spines.right':False,
                     'font.size':10,'svg.fonttype':'none'})
KEY='module(equipment)'
COLORS=['#007c83','#dc692d','#3868b2','#7952ac','#61768e','#ba448a','#a88b19','#24a36f']

def save(fig,name):
    fig.savefig(OUT/'figures'/f'{name}.png',dpi=135,bbox_inches='tight')
    fig.savefig(OUT/'figures'/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)

def embed(path):
    from PIL import Image
    with Image.open(path) as im:
        im=im.convert('RGB');im.thumbnail((1600,1600),Image.Resampling.LANCZOS)
        stream=io.BytesIO();im.save(stream,format='WEBP',quality=88,method=4)
    return 'data:image/webp;base64,'+base64.b64encode(stream.getvalue()).decode()
def esc(s):return html.escape(str(s))

def window_figure(w,d,s,model,info):
    fig,axs=plt.subplots(6,1,figsize=(14.5,13.5),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[1,1,1,.85,1.2,.5]})
    x=d.dt
    for k,f in enumerate(['voltageR','voltageS','voltageT']):axs[0].plot(x,d[f],lw=.7,color=COLORS[k],label=f[-1]+'상')
    axs[0].axhline(205,color='#b24542',ls='--',lw=1,label='205/225V 후보 기준');axs[0].axhline(225,color='#b24542',ls='--',lw=1)
    axs[0].set_ylabel('상전압 (V)');axs[0].legend(ncol=4,fontsize=8,loc='upper right')
    for k,f in enumerate(['powerFactorR','powerFactorS','powerFactorT']):axs[1].plot(x,d[f],lw=.7,color=COLORS[k],label=f[-1]+'상')
    axs[1].axhline(80,color='#b24542',ls='--',lw=1);axs[1].axhline(100,color='#b24542',ls='--',lw=1)
    axs[1].set_ylabel('상역률 (%)')
    for k,f in enumerate(['currentR','currentS','currentT']):axs[2].plot(x,d[f],lw=.7,color=COLORS[k])
    axs[2].set_ylabel('상전류 (A)')
    axs[3].plot(x,d.activePower,color=COLORS[0],lw=.8);axs[3].set_ylabel('유효전력 (kW)\n가이드 단위')
    score=s[f'{model}_score'].to_numpy(float);threshold=info['threshold']
    axs[4].plot(x,score,color=COLORS[3],lw=1,label='저장 모델 점수')
    axs[4].axhline(threshold,color='#b24542',ls='--',lw=1.2,label=f'고정 기준 {threshold:.6g}')
    if model.startswith('ae') or model.startswith('xgb'):
        axs[4].set_yscale('log');axs[4].yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'));axs[4].yaxis.set_minor_formatter(NullFormatter())
        axs[4].set_ylabel('모델 점수\n로그 축')
    else:axs[4].set_ylabel('모델 점수')
    axs[4].legend(loc='upper right',fontsize=9)
    candidate=d.weak_candidate.to_numpy().astype(bool);alarm=s[f'{model}_alarm'].to_numpy().astype(bool)
    axs[5].scatter(x[candidate],np.ones(candidate.sum()),marker='|',s=180,color='#1d567c',label='약한 규칙 후보')
    axs[5].scatter(x[alarm],np.repeat(2,alarm.sum()),marker='|',s=180,color='#dc692d',label='모델 경보')
    axs[5].set_yticks([1,2],['규칙 후보','모델 경보']);axs[5].set_ylim(.5,2.5)
    for ax in axs[:-1]:
        ax.grid(alpha=.17)
        for t in x[candidate]:ax.axvspan(t,t+pd.Timedelta(seconds=5),color='#dc692d',alpha=.13)
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    axs[-1].set_xlabel(f'{d.dt.min():%Y-%m-%d %H:%M:%S} ~ {d.dt.max():%Y-%m-%d %H:%M:%S} localtime · 원본 연속 5초 · 센서 세로축 확대')
    role='후보 주변' if w['role']=='candidate_visual' else '약한 규칙 정상 비교'
    fig.suptitle(f'{info["display_name"]} · {w["module"]} · {role}',fontsize=17,fontweight='bold')
    fig.supxlabel(f'360개 원본 관측 · 규칙 후보 {candidate.sum()}개 / 경보 {alarm.sum()}개 · 색칠된 관측은 탐색 후보이며 실제 고장 정답이 아님',fontsize=10)
    name=f'window_{model}_{w["window_id"]}';save(fig,name)
    return {'model':model,'window_id':w['window_id'],'png':f'figures/{name}.png','svg':f'figures/{name}.svg'}

def summary_figures(report,windows,scores):
    models=report['models'];names=[d['display_name'] for d in models.values()];xx=np.arange(len(models))
    fig,axs=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    original=[d['original_weak_agreement'] for d in models.values()]
    axs[0,0].bar(xx-.18,[d['precision'] for d in original],width=.35,color=COLORS[0],label='정밀도')
    axs[0,0].bar(xx+.18,[d['recall'] for d in original],width=.35,color=COLORS[1],label='재현율')
    axs[0,0].set_ylim(0,1.06);axs[0,0].set_xticks(xx,names,rotation=25,ha='right',fontsize=8);axs[0,0].legend()
    axs[0,0].set_title('원래 분포의 4월 표본 · 약한 후보 규칙 일치도')
    axs[0,1].bar(xx,[d['false_positive_rate']*100 for d in original],color=COLORS[2]);axs[0,1].axhline(.5,color=COLORS[1],ls='--',label='검증 목표 0.5%')
    axs[0,1].set_xticks(xx,names,rotation=25,ha='right',fontsize=8);axs[0,1].set_ylabel('규칙 정상에 대한 경보율 (%)');axs[0,1].legend();axs[0,1].set_title('0.5% 목표는 3월에서 고정 · 4월 달성 보장 없음')
    types=['spike','dip','gradual_drift','level_shift']
    heat=np.array([[d['synthetic_by_type'][t]['recall'] for t in types] for d in models.values()])
    im=axs[1,0].imshow(heat,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto');axs[1,0].set_xticks(range(4),['Spike','Dip','Gradual drift','Level shift'])
    axs[1,0].set_yticks(range(len(names)),names,fontsize=9)
    for i in range(len(names)):
        for j in range(4):axs[1,0].text(j,i,f'{heat[i,j]:.2f}',ha='center',va='center',color='white' if heat[i,j]>.55 else '#162c43')
    fig.colorbar(im,ax=axs[1,0],shrink=.7);axs[1,0].set_title('합성 주입 관측 재현율 · 실제 고장 성능과 분리')
    es=report['event_summary'];bars=es['by_equipment']
    axs[1,1].bar([r['equipment'] for r in bars],[r['rows'] for r in bars],color=COLORS[:len(bars)])
    axs[1,1].set_title(f'전체 원본의 탐색 후보 {es["candidate_rows"]:,}개');axs[1,1].set_ylabel('후보 관측 수');axs[1,1].tick_params(axis='x',rotation=15)
    axs[1,1].text(.03,.93,f'후보 연속 구간 {es["events"]:,}개\n단일 5초 구간 {es["single_record_event_fraction"]*100:.2f}%',transform=axs[1,1].transAxes,va='top')
    fig.suptitle('이상 후보를 여러 탐지 방법으로 확인한 결과',fontsize=18,fontweight='bold')
    save(fig,'anomaly_comparison_overview')
    # Same source observations before/after a causal median filter.
    w=next(w for w in report['windows'] if w['window_id']=='candidate_1')
    d=windows[windows.window_id==w['window_id']].reset_index(drop=True)
    voltage=d[['voltageR','voltageS','voltageT']]
    fig,axs=plt.subplots(2,1,figsize=(14,7),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[3,1]})
    axs[0].plot(d.dt,voltage.min(axis=1),lw=.8,color=COLORS[0],label='원본 최소 상전압')
    axs[0].plot(d.dt,voltage.rolling(5,min_periods=5).median().min(axis=1),lw=1.3,color=COLORS[1],label='상별 25초 trailing median의 최소값')
    axs[0].axhline(205,color='#b24542',ls='--',label='205V 후보 기준');axs[0].legend(ncol=3);axs[0].set_ylabel('상전압 (V) · 확대 축')
    raw=d.weak_candidate.to_numpy().astype(bool);axs[1].scatter(d.dt[raw],np.ones(raw.sum()),marker='|',s=180,color=COLORS[0]);axs[1].set_ylim(.5,1.5);axs[1].set_yticks([1],['원본 후보'])
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'));axs[1].set_xlabel(f'{d.dt.min():%Y-%m-%d %H:%M:%S} ~ {d.dt.max():%Y-%m-%d %H:%M:%S} localtime · 실제 연속 5초 관측')
    fig.suptitle('단발 전압 저하 후보는 평활화 후 사라질 수 있습니다',fontsize=17,fontweight='bold')
    fig.supxlabel('노이즈 제거 효과와 이상 후보 삭제를 구분해야 합니다. 원본 관측과 평활화 관측의 경보를 별도로 보존합니다.',fontsize=10)
    save(fig,'median_suppression')

def synthetic_figures(report):
    manifest=json.loads((OUT/'augmented_sample_manifest.json').read_text(encoding='utf-8'))
    synt=pd.read_csv(OUT/'synthetic_observations.csv.gz',parse_dates=['dt'])
    scores=pd.read_csv(OUT/'synthetic_test_scores.csv.gz',parse_dates=['dt'],float_precision='round_trip')
    illustrated=[]
    for typ in ['spike','dip','gradual_drift','level_shift']:
        pool=[e for e in manifest['events'] if e['split']=='test' and e['type']==typ]
        e=next((e for e in pool if e['feature'].startswith('voltage') and abs(e['signed_amplitude_iqr'])>=2),pool[0])
        d=synt[synt.event_id==e['event_id']].reset_index(drop=True);s=scores[scores.event_id==e['event_id']].reset_index(drop=True)
        fig,axs=plt.subplots(3,1,figsize=(14,9),sharex=True,layout='constrained')
        axs[0].plot(d.dt,d.base_value,color=COLORS[0],lw=.9,label='같은 원본 관측')
        axs[0].plot(d.dt,d[e['feature']],color=COLORS[1],lw=1.3,label='합성 주입 후 관측');axs[0].set_ylabel(e['feature']+'\n원 단위, 확대 축');axs[0].legend()
        axs[1].plot(d.dt,d.injected_delta,color=COLORS[1],lw=1.2);axs[1].axhline(0,color='#61768e',ls=':');axs[1].set_ylabel('기록된 주입량\n원 단위')
        for color,(name,info) in zip(COLORS,report['models'].items()):
            ratio=s[f'{name}_score']/info['threshold'];axs[2].plot(d.dt,ratio,color=color,lw=.9,label=info['display_name'])
        axs[2].axhline(1,color='#26374c',ls='--',lw=1.2,label='각 모델의 고정 경보 기준')
        axs[2].set_yscale('log');axs[2].yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'));axs[2].yaxis.set_minor_formatter(NullFormatter())
        axs[2].set_ylabel('점수 / 경보 기준\n로그 축');axs[2].legend(ncol=3,fontsize=8)
        for ax in axs:
            ax.grid(alpha=.15);ax.axvspan(d.loc[e['start_offset'],'dt'],d.loc[e['stop_offset']-1,'dt']+pd.Timedelta(seconds=5),color=COLORS[1],alpha=.13)
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        axs[-1].set_xlabel(f'{e["module"]} · {d.dt.min():%Y-%m-%d} localtime · 검증/학습과 분리한 합성 테스트')
        fig.suptitle(f'합성 {typ} · {e["feature"]} · {e["duration_seconds"]}초 · {e["signed_amplitude_iqr"]:+g} training IQR',fontsize=17,fontweight='bold')
        fig.supxlabel('색칠한 구간만 알려진 인위적 주입 mask입니다. 원본 데이터에 삽입된 실제 이상치 위치를 뜻하지 않습니다.',fontsize=10)
        name='synthetic_'+typ;save(fig,name);illustrated.append({'type':typ,'event_id':e['event_id'],'png':f'figures/{name}.png','svg':f'figures/{name}.svg'})
    return illustrated

def build_html(report,figures,synthfigs):
    models=report['models'];table='';stable=''
    for name,info in models.items():
        m=info['original_weak_agreement'];s=info['synthetic_heldout_metrics'];em=info['synthetic_event_detection']
        table+=f'<tr><td>{esc(info["display_name"])}</td><td>{m["precision"]:.3f}</td><td>{m["recall"]:.3f}</td><td>{m["average_precision"]:.3f}</td><td>{m["false_positive_rate"]*100:.3f}%</td><td>{m["true_positive"]}/{m["positive_rows"]}</td><td>{m["false_positive"]}</td></tr>'
        stable+=f'<tr><td>{esc(info["display_name"])}</td><td>{s["recall"]:.3f}</td><td>{s["precision"]:.3f}</td><td>{s["false_positive_rate"]*100:.3f}%</td><td>{em["detected_any_fraction"]:.3f}</td><td>{em["paired_original_control_any_alarm_fraction"]:.3f}</td></tr>'
    visual=[w for w in report['windows'] if w['role'] in ['candidate_visual','normal_visual']]
    modeloptions=''.join(f'<option value="{name}">{esc(info["display_name"])}</option>' for name,info in models.items())
    windowoptions=''.join(f'<option value="{w["window_id"]}">{esc(w["module"])} {"후보 주변" if w["role"]=="candidate_visual" else "정상 비교"} · {esc(w["start"])}</option>' for w in visual)
    images={f['model']+'|'+f['window_id']:embed(OUT/f['png']) for f in figures}
    sfhtml=''.join(f'<figure><img src="{embed(OUT/f["png"])}" alt="합성 {f["type"]} 테스트"><figcaption>예시 event ID: {esc(f["event_id"])}</figcaption></figure>' for f in synthfigs)
    es=report['event_summary'];single=es['single_record_event_fraction']*100
    med=[r for r in report['median_suppression'] if r['window_id'].startswith('candidate_')]
    medtext=' / '.join(f'{r["window_id"]}: 원본 {r["raw_candidate_rows"]}개 중 {r["raw_candidates_removed"]}개 제거' for r in med)
    sample=next(iter(models.values()))['original_weak_agreement']
    firstimage=next(iter(images.values()))
    details=''.join(f'<details><summary>{esc(info["display_name"])} 학습·임계값</summary><p>{esc(info["training"])}</p><p>특성: {esc(", ".join(info["features"]))}</p><p>고정 기준 {info["threshold"]:.17g}. {esc(info["threshold_basis"])}</p></details>' for info in models.values())
    page=rf'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>이상 후보 확인과 합성 증강 비교</title>
<style>body{{margin:0;background:#edf2f7;color:#193148;font-family:"Malgun Gothic",sans-serif;line-height:1.75}}main{{max-width:1180px;margin:auto;padding:24px}}header,section{{background:white;border:1px solid #d8e2ec;border-radius:14px;padding:28px;margin:20px 0}}header{{background:#16344c;color:white}}h1{{font-size:30px;line-height:1.45;margin:0 0 12px}}h2{{font-size:22px;line-height:1.5;margin:0 0 15px}}h3{{font-size:17px}}p{{margin:12px 0}}.note{{background:#fff2e4;border-left:4px solid #ce682f;padding:15px 18px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.card{{background:#f2f6fb;padding:18px;border-radius:9px}}.card strong{{font-size:24px;display:block}}img{{width:100%;height:auto;display:block}}figure{{margin:22px 0}}figcaption,.muted{{color:#617387;font-size:12px}}.tablewrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:11px 9px;border-bottom:1px solid #dce5ee;text-align:right;white-space:nowrap}}td:first-child,th:first-child{{text-align:left}}th{{background:#eef4fa}}select{{max-width:100%;padding:10px;font:inherit;border:1px solid #b5c4d2;border-radius:6px;background:white;margin:7px 10px 12px 0}}label{{display:inline-block}}a{{color:#006e86}}details{{border-top:1px solid #dce5ee;padding:14px 0}}summary{{cursor:pointer;font-weight:bold}}code{{font-family:Consolas,monospace;font-size:12px;overflow-wrap:anywhere}}.files{{overflow-wrap:anywhere}}@media(max-width:780px){{main{{padding:10px}}header,section{{padding:20px 15px}}h1{{font-size:25px}}.grid{{grid-template-columns:1fr}}}}</style></head>
<body><main><header><h1>이상 후보 확인과 합성 증강 비교</h1><p>전압·역률 탐색 후보를 실제 연속 5초 관측에서 확인하고, 기존 모델 4개와 새로운 탐지 방법 3개를 비교했습니다.</p></header>
<section><h2>전체 원본에서 확인한 관측 후보</h2><div class="grid"><div class="card">전기품질 탐색 후보<strong>{es['candidate_rows']:,}개</strong>33,696,013개 관측 중 약 0.190%</div><div class="card">후보의 연속 구간<strong>{es['events']:,}개</strong>같은 설비의 연속 5초 후보를 묶음</div><div class="card">단일 5초 구간 비율<strong>{single:.2f}%</strong>연속 구간 수 기준</div></div><p>후보는 3호기 저전압, 예비건조기·6호기 저역률에 집중됩니다. 다른 탐지 방법의 점수도 함께 제시했으며, 센서별 비정상 분포와 모델 경보가 같은 관측을 가리키는지 확인할 수 있습니다.</p><div class="note">실제 고장 기록과 가이드가 언급한 원본 삽입 이상치의 정답 위치는 제공되지 않았습니다. 아래의 원본 후보는 상전압 205–225V 밖, 상역률 80–100% 밖 등의 탐색 규칙으로 정했습니다. 고장 또는 삽입 이상치를 확정한 결과로 해석할 수 없습니다.</div></section>
<section><h2>모델 비교의 결과와 해석</h2><img src="{embed(OUT/'figures'/'anomaly_comparison_overview.png')}" alt="원본 약한 후보 일치도와 합성 테스트 비교"><p>원래 분포를 유지한 4월 표본 {sample['rows']:,}개에는 약한 후보 {sample['positive_rows']}개({sample['positive_rate']*100:.3f}%)가 포함됩니다. 후보가 드물기 때문에 정상 관측에 약 0.5%만 경보를 내도 경보의 상당 부분이 규칙 정상일 수 있습니다. 이 낮은 정밀도를 실제 고장 오탐률로 바꾸어 표현하지 않았습니다.</p><div class="tablewrap"><table><thead><tr><th>모델</th><th>약한 정밀도</th><th>약한 재현율</th><th>AP</th><th>규칙 정상 경보율</th><th>포착 후보</th><th>규칙 정상 경보</th></tr></thead><tbody>{table}</tbody></table></div><p>규칙 모사 XGB는 라벨을 정의한 전압·역률을 그대로 보므로 독립적인 진단 검증이 아닙니다. 규칙 센서 제외 XGB는 전류·전력에 후보 정보를 전달할 수 있는지 확인하는 비교입니다. Isolation Forest와 robust-z는 정상 proxy의 다변량 희소성·센서별 극단값을 찾습니다. 증강 XGB는 알려진 인위적 주입만 양성으로 학습했으며 원본의 후보 규칙 양성은 학습하지 않았습니다.</p></section>
<section><h2>각 모델의 후보 구간과 정상 비교 구간</h2><p>아래는 원본에서 추출한 연속 30분, 360개 5초 관측입니다. 시간당 4개씩 뽑은 불연속 학습 표본을 연결한 그림이 아닙니다. 모델과 구간을 선택하면 해당 모델의 센서·점수·기준·경보 그림으로 바뀝니다. 센서 세로축은 작은 변화를 볼 수 있도록 확대했습니다.</p><label for="model">모델</label><select id="model">{modeloptions}</select><label for="window">구간</label><select id="window">{windowoptions}</select><img id="windowfigure" src="{firstimage}" alt="모델별 원본 연속 구간"><p id="windowcaption" class="muted"></p><p class="muted">전체 후보 63,963개도 모든 센서로 다시 추론했습니다. 후보만 모은 파일의 포착률에는 정상 분모가 없으므로 정밀도 계산에 사용하지 않았습니다.</p></section>
<section><h2>단발 후보와 평활화의 관계</h2><img src="{embed(OUT/'figures'/'median_suppression.png')}" alt="원본 단발 전압 후보와 trailing median 비교"><p>5개 관측, 25초 trailing median은 주변 정상값으로 단일 저하값을 대체할 수 있습니다. {esc(medtext)}. 이상 후보를 확인하는 작업에서는 평활화가 후보를 제거하는 효과도 함께 측정해야 합니다. 지속시간 조건은 별도의 경보 정책이며 원본 점수와 관측을 남기는 것이 좋습니다.</p></section>
<section><h2>Spike·dip·drift·shift 증강 실험</h2><p>학습·검증·테스트의 날짜를 먼저 분리하고 각 구간의 약한 규칙 정상 30분 원본 창에 인위적인 단일 센서 변화를 더했습니다. Spike/dip은 5초, gradual drift/level shift는 60–180초입니다. 주입량은 학습 정상 분포의 IQR로 지정했으며 검증·테스트에는 0.5/1/2/4 IQR 강도를 사용했습니다. 음수 센서값은 0으로 제한하고 제한한 행 수를 manifest에 기록했습니다.</p><p><strong>합성 증강 학습은 XGB에만 적용했습니다.</strong> AE 두 개는 기존 저장 모델을 그대로 재추론했으며 증강 AE를 학습한 실험은 아닙니다. 전체 학습 AE의 합성 관측 재현율은 {models['ae_unsupervised_all_training']['synthetic_heldout_metrics']['recall']*100:.2f}%, 합성 증강 XGB는 {models['xgb_synthetic_augmented']['synthetic_heldout_metrics']['recall']*100:.2f}%로, 이 실험에서 일관된 개선을 확인하지 못했습니다. 두 모델은 구조와 학습 목표가 다르며, 기존 규칙 XGB와 합성 XGB도 양성 정의가 달라 증강만의 효과를 분리한 비교가 아닙니다.</p><p>기존·새 모델 모두 3월 원본 정상 proxy의 99.5분위 기준을 고정했습니다. 합성 테스트 {report['synthetic']['test_rows']:,}개 관측 중 주입 관측은 {report['synthetic']['injected_test_rows']:,}개입니다. 구성된 주입 빈도와 강도는 실제 이상 발생 분포가 아닙니다. 전압·역률·전류·전력별 결과와 강도별 결과도 JSON에 저장했습니다.</p><div class="tablewrap"><table><thead><tr><th>모델</th><th>합성 관측 재현율</th><th>합성 관측 정밀도</th><th>주입 밖 관측 경보율</th><th>주입 구간 한 번 이상 경보</th><th>동일 원본 구간 경보</th></tr></thead><tbody>{stable}</tbody></table></div><p>긴 구간은 우연히 한 번 경보가 날 기회도 많습니다. 따라서 같은 시간 구간의 주입 전 원본 경보를 함께 비교했습니다. 합성의 단발과 지속 구간은 같은 수지만 관측 재현율에는 긴 구간의 관측이 더 많이 반영됩니다. 첫 경보 지연은 포착한 구간에 대해서만 계산하고 누락은 구간 포착률에 그대로 남겼습니다. point-adjustment로 한 관측의 경보를 구간 전체의 정답으로 확장하지 않았습니다. 센서별 행 모델은 시간 순서를 입력하지 않으므로 gradual drift의 원인을 학습했다는 주장 대신 주입 ramp의 각 관측 진폭에 반응한 정도를 평가했습니다. 다른 센서는 물리적으로 재계산하지 않아 검증된 설비 고장 시뮬레이션으로 볼 수 없습니다.</p>{sfhtml}</section>
<section><h2>학습·임계값 잠금과 재현</h2><p>학습은 2024년 12월–2025년 2월, 검증은 3월, 테스트는 4월 1–29일입니다. 스케일·정상 분포 통계는 학습 데이터에서만 계산했습니다. 새로운 합성 주입은 split별 원본 timestamp·설비 키와 서로 다른 seed를 갖습니다. 기존 저장 모델은 그대로 복사하여 재추론했고 새 폴더에서만 결과를 저장했습니다.</p>{details}<div class="files"><p><a href="anomaly_metrics.json">전체 결과 JSON</a> · <a href="threshold_locks.json">고정 기준</a> · <a href="window_manifest.json">연속 원본 구간</a> · <a href="augmented_sample_manifest.json">합성 주입 manifest</a> · <a href="anomaly_study.py">실험 코드</a> · <a href="build_anomaly_report.py">보고서 코드</a> · <a href="predict_anomaly.py">저장 모델 추론</a></p><p>추론: <code>output\modeling\.venv\Scripts\python.exe output\optimization_20261013\anomaly\predict_anomaly.py --input observations.csv --output new_scores.csv</code></p></div><p class="muted">이 후속 실험은 기존 EDA에서 이미 본 4월 후보를 재검토합니다. 새 운영 데이터에서 경보량·조치 결과·실제 고장 라벨을 수집한 후 실효성을 확인할 수 있습니다. 원본 계측 계층과 전력 단위의 기존 불확실성도 유지됩니다.</p></section></main>
<script>const figureData={json.dumps(images,ensure_ascii=False)};const modelNames={json.dumps({n:d['display_name'] for n,d in models.items()},ensure_ascii=False)};const windows={json.dumps({w['window_id']:w for w in visual},ensure_ascii=False)};function updateFigure(){{const m=document.getElementById('model').value;const w=document.getElementById('window').value;document.getElementById('windowfigure').src=figureData[m+'|'+w];document.getElementById('windowfigure').alt=modelNames[m]+' '+windows[w].module+' 원본 연속 구간';document.getElementById('windowcaption').textContent=modelNames[m]+' · '+windows[w].module+' · '+windows[w].start+'–'+windows[w].end_exclusive+' · 임계값은 테스트 전 고정';}}document.getElementById('model').addEventListener('change',updateFigure);document.getElementById('window').addEventListener('change',updateFigure);updateFigure();</script></body></html>'''
    if 'ae_synthetic_denoising' in models:
        ae=models['ae_synthetic_denoising'];am=ae['synthetic_heldout_metrics'];an=ae['original_weak_agreement']
        start=page.index('<p><strong>합성 증강 학습은 XGB에만 적용했습니다.</strong>')
        stop=page.index('</p>',start)+4
        paragraph=f'<p><strong>합성 증강을 XGB와 새 denoising AE에 각각 적용했습니다.</strong> 기존 AE 두 개는 그대로 재추론했습니다. 새 AE는 오염된 입력을 같은 원본 timestamp의 주입 전 정상 proxy 센서값으로 복원하도록 학습했으며, 오염된 입력 자체를 복원 목표로 사용하지 않았습니다. scaler는 12월–2월 원본 정상 proxy만으로 fit하고, 3월 원본 정상 재구성 오차로 checkpoint와 99.5분위 기준을 고정했습니다. 주입 전 목표값을 아는 준지도 학습입니다.</p><p>합성 관측 재현율은 기존 전체 AE {models["ae_unsupervised_all_training"]["synthetic_heldout_metrics"]["recall"]*100:.2f}%, 기존 약한 정상 AE {models["ae_weak_normal_proxy"]["synthetic_heldout_metrics"]["recall"]*100:.2f}%, 증강 XGB {models["xgb_synthetic_augmented"]["synthetic_heldout_metrics"]["recall"]*100:.2f}%, denoising AE {am["recall"]*100:.2f}%입니다. Denoising AE는 이 인위적 주입 시험에서 더 많은 주입 관측을 포착했지만 원본 약한 정밀도는 {an["precision"]*100:.2f}%입니다. 실제 고장 진단의 개선을 입증한 결과는 아닙니다. 학습 데이터·목표·학습 횟수가 달라 증강 하나의 효과를 분리한 비교도 아닙니다.</p>'
        page=page[:start]+paragraph+page[stop:]
        page=page.replace('기존 모델 4개와 새로운 탐지 방법 3개','기존 모델 4개와 새로운 탐지 방법 4개')
        page=page.replace('<a href="anomaly_study.py">실험 코드</a>','<a href="anomaly_study.py">실험 코드</a> · <a href="add_denoising_ae.py">증강 AE 학습 코드</a>')
    temporary=OUT/'tmp'/'anomaly_report.tmp.html';temporary.write_text(page,encoding='utf-8');temporary.replace(OUT/'anomaly_report.html')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--html-only',action='store_true');parser.add_argument('--new-only',action='store_true');args=parser.parse_args()
    report=json.loads((OUT/'anomaly_metrics.json').read_text(encoding='utf-8'))
    synthetic_scores=pd.read_csv(OUT/'synthetic_test_scores.csv.gz',float_precision='round_trip')
    from anomaly_study import metrics
    for name,info in report['models'].items():
        e=pd.read_csv(OUT/f'synthetic_event_results_{name}.csv')
        info['synthetic_by_feature']={feature:metrics(d.synthetic_label,d[name+'_score'],info['threshold'])
                                      for feature,d in synthetic_scores.groupby('perturbed_feature')}
        info['synthetic_event_detection']['macro_mean_event_injected_point_recall']=float(e.injected_point_recall.mean())
        delays=e.loc[e.detected_any,'first_alarm_delay_seconds'].dropna()
        info['synthetic_event_detection']['detected_events_first_alarm_delay_seconds']={
            'detected_events':int(e.detected_any.sum()),
            'median':float(delays.median()) if len(delays) else None,
            'mean':float(delays.mean()) if len(delays) else None,
            'definition':'Seconds from first injected observation to first alarm within the known injection mask; detected events only, misses remain in event recall',
            'by_type':[{ 'type':typ,'detected_events':int(d.detected_any.sum()),
                        'median_seconds':float(d.loc[d.detected_any,'first_alarm_delay_seconds'].median()) if d.detected_any.any() else None}
                       for typ,d in e.groupby('type')]}
    if args.html_only:
        (OUT/'anomaly_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        build_html(report,report['figures']['per_model_original_windows'],report['figures']['synthetic_examples'])
        print('Updated offline HTML and per-feature/delay JSON, existing figures preserved');return
    windows=pd.read_csv(OUT/'continuous_source_windows.csv.gz',parse_dates=['dt'])
    scores=pd.read_csv(OUT/'continuous_window_scores.csv.gz',parse_dates=['dt'],float_precision='round_trip')
    assert len(windows)==len(scores) and windows[['window_id','localtime']].equals(scores[['window_id','localtime']])
    figures=[]
    for name,info in report['models'].items():
        for w in report['windows']:
            if w['role']=='synthetic_base':continue
            d=windows[windows.window_id==w['window_id']].reset_index(drop=True);s=scores[scores.window_id==w['window_id']].reset_index(drop=True)
            stem=f'window_{name}_{w["window_id"]}'
            if args.new_only and (OUT/'figures'/f'{stem}.png').exists() and (OUT/'figures'/f'{stem}.svg').exists():
                figures.append({'model':name,'window_id':w['window_id'],'png':f'figures/{stem}.png','svg':f'figures/{stem}.svg'})
            else:figures.append(window_figure(w,d,s,name,info))
    summary_figures(report,windows,scores);synthfigs=synthetic_figures(report)
    report['figures']={'per_model_original_windows':figures,'synthetic_examples':synthfigs,'overview':'figures/anomaly_comparison_overview.png','median':'figures/median_suppression.png'}
    (OUT/'anomaly_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    build_html(report,figures,synthfigs)
    print(f'Built {len(figures)} per-model original window figures + 6 summary/synthetic PNG and SVG, offline interactive HTML',flush=True)

if __name__=='__main__':main()
