"""Reproducible EDA figures and an offline Korean report; input files stay unchanged."""
from pathlib import Path
import os, sys, json, base64, html

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT / '.plot_deps'))
os.environ['MPLCONFIGDIR'] = str(OUT / 'tmp' / 'mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib import font_manager

FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,
                     'figure.facecolor':'#f7f9fc','axes.facecolor':'#ffffff',
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.edgecolor':'#c7d2df','axes.labelcolor':'#334155',
                     'text.color':'#172a43','xtick.color':'#475569',
                     'ytick.color':'#475569','font.size':10,
                     'axes.titleweight':'bold','axes.titlesize':13,
                     'savefig.facecolor':'#f7f9fc','svg.fonttype':'none'})
TEAL, ORANGE, BLUE, GREY = '#007f86','#dc6b26','#3467b7','#61758e'
comma = FuncFormatter(lambda x,p: f'{x:,.0f}')
h = pd.read_csv(OUT / 'hourly_by_equipment.csv', parse_dates=['hour'])
e = pd.read_csv(OUT / 'equipment_summary.csv')
s = pd.read_csv(OUT / 'sampled_sensor_rows.csv', parse_dates=['dt'])
m = pd.read_csv(ROOT / 'may-dataset' / 'rtu_ground_truth_may.csv', parse_dates=['id'])
guide = json.loads((OUT / 'guide_summary.json').read_text(encoding='utf-8'))
hc = h[h['complete_720'].astype(str).str.lower() == 'true'].copy()
module_count = e.shape[0]
g = hc.groupby('hour').agg(energy=('energy_kwh','sum'), modules=('module(equipment)','nunique'))
g = g[g.modules == module_count].copy()
g['dow'],g['hod'] = g.index.dayofweek,g.index.hour
daily = g.energy.resample('D').agg(['mean','count'])
daily = daily[daily['count'] == 24]
md = m.set_index('id').hourly_pow.resample('D').agg(['mean','count'])
md_full = md[md['count'] == 24]
heat = g.pivot_table(index='dow', columns='hod', values='energy', aggfunc='mean')
heat_pct = (heat / g.energy.mean() - 1)*100
eq_full = hc.groupby('module(equipment)').energy_kwh.sum().reindex(e.equipment)
eq_means = hc.groupby('module(equipment)').energy_kwh.mean().reindex(e.equipment)
eq_dev = (eq_means / eq_means.mean() - 1)*100
may_cv = float(m.hourly_pow.std()/m.hourly_pow.mean()*100)
train_cv = float(g.energy.std()/g.energy.mean()*100)
delta_pct = float((m.hourly_pow.mean()/g.energy.mean()-1)*100)
sample_n = len(s)

def tidy(ax, yfmt=True):
    ax.grid(axis='y',alpha=.20,color='#9aacc0'); ax.set_axisbelow(True)
    if yfmt: ax.yaxis.set_major_formatter(comma)

def dates(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m월'))
    ax.tick_params(axis='x',rotation=0)

def save(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=160,bbox_inches='tight')
    fig.savefig(FIG/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)

def daily_panel(ax, zero=False):
    ax.plot(daily.index,daily['mean'],color=TEAL,lw=1.3,label='학습: 완전한 하루의 시간당 평균')
    ax.plot(md_full.index,md_full['mean'],color=ORANGE,lw=1.5,marker='o',markersize=3,label='5월 정답: 완전한 하루의 시간당 평균')
    ax.axhline(g.energy.mean(),color=TEAL,ls=':',alpha=.65,lw=1)
    ax.set_ylabel('모듈 합계 시간당 에너지 (kWh)')
    ax.set_title('전체 수준은 안정적 · 5월 평균 변화 '+f'{delta_pct:+.3f}%')
    dates(ax); tidy(ax)
    ax.set_xlim(daily.index.min()-pd.Timedelta(days=2),m.id.max()+pd.Timedelta(hours=6))
    if zero: ax.set_ylim(0,max(g.energy.max(),m.hourly_pow.max())*1.12)
    else:
        values=np.r_[daily['mean'].values,md_full['mean'].values]
        span=values.max()-values.min()
        ax.set_ylim(values.min()-span*.12,values.max()+span*.16)
    return ax

fig, axs=plt.subplots(2,1,figsize=(13,8),height_ratios=[2.2,1],layout='constrained')
daily_panel(axs[0]); axs[0].legend(loc='upper right',fontsize=9)
axs[0].text(.01,.02,'세로축 확대 · 점 사이의 빈 구간은 관측 없음',transform=axs[0].transAxes,color=GREY,fontsize=9)
daily_panel(axs[1],True); axs[1].set_title('0부터 시작하는 축으로 확인한 절대 수준',fontsize=11)
fig.suptitle('학습 기간과 5월 검증 정답의 일평균 비교',fontsize=17,fontweight='bold')
fig.supxlabel('localtime 기준 · 4/30 부분시간 및 5/31 부분일 제외 · 모듈 계층 미확인',fontsize=9)
save(fig,'01_daily_train_may')

fig,axs=plt.subplots(1,2,figsize=(13,7),width_ratios=[1.5,1],layout='constrained')
y=np.arange(len(e)); colors=[TEAL]*len(e)
axs[0].barh(y,eq_full.values/1e6,color=colors,height=.66)
axs[0].set_yticks(y,e.equipment); axs[0].invert_yaxis()
axs[0].set_xlabel('12/1–4/29 완전시간 관측 에너지 (백만 kWh)')
axs[0].set_title('설비별 관측 에너지 · 0축'); axs[0].grid(axis='x',alpha=.2)
axs[0].set_axisbelow(True); axs[0].set_xlim(0,eq_full.max()/1e6*1.17)
for yy,v in zip(y,eq_full.values/1e6): axs[0].text(v+.06,yy,f'{v:.4f}',va='center',fontsize=9)
axs[1].barh(y,eq_dev.values,color=[ORANGE if x>0 else BLUE for x in eq_dev.values],height=.66)
axs[1].set_yticks(y,[]); axs[1].invert_yaxis(); axs[1].axvline(0,color=GREY,lw=1)
axs[1].set_xlabel('설비 평균 대비 시간당 에너지 차이 (%)')
axs[1].set_title('차이를 확대해서 본 상대편차')
for yy,v in zip(y,eq_dev): axs[1].text(v+(0.003 if v>=0 else -.003),yy,f'{v:+.3f}%',va='center',ha='left' if v>=0 else 'right',fontsize=9)
axs[1].set_xlim(min(eq_dev.min()*1.9,-.06),max(eq_dev.max()*1.9,.06)); axs[1].grid(axis='x',alpha=.2)
fig.suptitle('설비 간 소비 규모가 매우 유사함',fontsize=17,fontweight='bold')
fig.supxlabel('모듈 합계는 계층 미확인 · 합계를 공장 사용량으로 확정하지 않음 · 가이드 kW 단위 적용',fontsize=9)
save(fig,'02_equipment_energy')

fig,ax=plt.subplots(figsize=(13,5.5),layout='constrained')
vlim=max(abs(heat_pct.values).max(),.05)
im=ax.imshow(heat_pct.values,aspect='auto',cmap='RdBu_r',vmin=-vlim,vmax=vlim)
ax.set_yticks(np.arange(7),['월','화','수','목','금','토','일']); ax.set_xticks(np.arange(24),[f'{x:02d}' for x in range(24)])
ax.set_xlabel('localtime 시간대'); ax.set_ylabel('요일'); ax.set_title('요일 × 시간대의 상대편차: 관측 평균 대비 (%)',pad=15)
cb=fig.colorbar(im,ax=ax,shrink=.85,pad=.025); cb.set_label('학습 전체 평균 대비 편차 (%)')
for yy in range(7):
    for xx in range(24): ax.text(xx,yy,f'{heat_pct.iloc[yy,xx]:+.2f}',ha='center',va='center',fontsize=7.3,color='white' if abs(heat_pct.iloc[yy,xx])>.65*vlim else '#24364b')
fig.supxlabel('색 범위는 작은 차이를 보기 위해 확대됨 · 명확한 생산 주기 또는 절감 가능성으로 해석하지 않음',fontsize=9)
save(fig,'03_weekday_hour_heatmap')

fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
spec=[('activePower','유효전력', 'kW',TEAL),('pf_mean','3상 평균 역률','%',ORANGE),
      ('currentR','R상 전류','A',BLUE),('voltageR','R상 전압','V',TEAL),
      ('current_unbalance_pct','3상 전류 불균형 지표','%',ORANGE),
      ('voltage_unbalance_pct','3상 전압 불균형 지표','%',BLUE)]
for ax,(col,title,unit,color) in zip(axs.flat,spec):
    ax.hist(s[col].dropna(),bins=42,color=color,alpha=.84,edgecolor='white',linewidth=.35)
    ax.set_title(title,fontsize=12); ax.set_xlabel(f'{col} ({unit})',fontsize=9); ax.set_ylabel('표본 관측 수'); tidy(ax)
    ax.text(.97,.93,f'평균 {s[col].mean():,.2f}\n표준편차 {s[col].std():,.2f}',ha='right',va='top',transform=ax.transAxes,fontsize=9,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
fig.suptitle(f'원시 센서 표본 분포 · {sample_n:,}행',fontsize=17,fontweight='bold')
fig.supxlabel('전수 통계와 구분: 30만 행 청크에서 2,000행 간격으로 추출한 체계적 표본 · 추론용 무작위 표본 아님',fontsize=9)
save(fig,'04_sample_sensor_distributions')

fig,axs=plt.subplots(2,1,figsize=(13,8),height_ratios=[2,1],layout='constrained')
for ax,zero in zip(axs,[False,True]):
    ax.plot(m.id,m.hourly_pow,color=ORANGE,lw=.9)
    ax.axhline(m.hourly_pow.mean(),ls='--',color=GREY,lw=1,label=f'평균 {m.hourly_pow.mean():,.1f}')
    ax.set_ylabel('hourly_pow (가이드 해석: kWh)'); tidy(ax)
    ax.set_xticks(pd.date_range(m.id.min(),m.id.max(),freq='4D')); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    if zero: ax.set_ylim(0,m.hourly_pow.max()*1.1); ax.set_title('동일 데이터 · 0부터 시작하는 축',fontsize=11)
    else: ax.set_title(f'5월 정답 505시간 · 변동계수 {may_cv:.3f}% · 세로축 확대'); ax.legend(fontsize=9)
fig.suptitle('작은 변동이 확대 축에서 크게 보일 수 있음',fontsize=17,fontweight='bold')
fig.supxlabel('2025-05-10 00:00–05-31 00:00 · 끝 시각 포함 · hourly_pow는 요금/탄소 선형 환산과 대응하나 원시 5월 자료 없음',fontsize=9)
save(fig,'05_may_zoom_zero')

fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained')
daily_panel(axs[0,0]); axs[0,0].set_title('일평균 추이 · 세로축 확대',fontsize=12)
axs[0,0].set_ylabel('시간당 에너지 (kWh)'); axs[0,0].tick_params(labelsize=9)
axs[0,1].bar(np.arange(len(e)),eq_full/1e6,color=TEAL)
axs[0,1].set_xticks(np.arange(len(e)),[x.split('(')[0] for x in e.equipment]); axs[0,1].set_xlabel('모듈 번호');axs[0,1].set_ylabel('관측 에너지 (백만 kWh)')
axs[0,1].set_title('설비별 관측 에너지 · 0축',fontsize=12);tidy(axs[0,1],False)
axs[1,0].imshow(heat_pct.values,aspect='auto',cmap='RdBu_r',vmin=-vlim,vmax=vlim)
axs[1,0].set_yticks(np.arange(7),['월','화','수','목','금','토','일']);axs[1,0].set_xticks(np.arange(0,24,3));axs[1,0].set_xlabel('시간대');axs[1,0].set_title(f'요일 × 시간 상대편차 · ±{vlim:.3f}% 범위',fontsize=12)
axs[1,1].hist(s.activePower,bins=40,color=ORANGE,edgecolor='white',linewidth=.4)
axs[1,1].set_title('원시 표본 유효전력 분포',fontsize=12);axs[1,1].set_xlabel('activePower (가이드 단위 kW)');axs[1,1].set_ylabel('표본 수');tidy(axs[1,1])
fig.suptitle('RTU 프로젝트 · 기초 데이터 분석',fontsize=20,fontweight='bold')
fig.supxlabel('학습 3,369.6만 행 · 13개 모듈 · localtime 기준 · 모듈 계층·전력 단위·Unix 시간 대응 검증 필요',fontsize=10)
save(fig,'overview')

# Derived descriptive summaries also support reproducibility and report review.
descriptive={'training_full_hours':len(g),'training_full_days':len(daily),'modules':module_count,
             'training_hourly_module_sum_mean_kwh':float(g.energy.mean()),
             'training_hourly_module_sum_std_kwh':float(g.energy.std()),
             'training_hourly_module_sum_cv_pct':train_cv,
             'may_hourly_pow_mean':float(m.hourly_pow.mean()),'may_hourly_pow_cv_pct':may_cv,
             'may_vs_training_mean_pct':delta_pct,
             'equipment_mean_relative_min_pct':float(eq_dev.min()),
             'equipment_mean_relative_max_pct':float(eq_dev.max()),
             'weekday_hour_relative_min_pct':float(np.nanmin(heat_pct)),
             'weekday_hour_relative_max_pct':float(np.nanmax(heat_pct)),
             'sample_rows':sample_n,
             'complete_time_observed_energy_kwh_module_sum_unconfirmed_hierarchy':float(g.energy.sum())}
(FIG/'figure_descriptive_metrics.json').write_text(json.dumps(descriptive,ensure_ascii=False,indent=2),encoding='utf-8')

def img(name,alt):
    data=base64.b64encode((FIG/f'{name}.png').read_bytes()).decode()
    return f'<img loading="lazy" src="data:image/png;base64,{data}" alt="{html.escape(alt)}">'

def num(x,n=0): return f'{x:,.{n}f}'
def readjson(name):
    p=OUT/name
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}

metrics=readjson('analysis_metrics.json')
anomaly_section=''
quality=metrics.get('quality',{})
if metrics.get('anomalies'):
    anomaly=metrics['anomalies']
    ac=pd.read_csv(OUT/'anomaly_candidate_rows.csv',parse_dates=['dt'])
    by=anomaly['by_equipment'];labels=[q['module(equipment)'] for q in by]
    fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    yy=np.arange(len(by));vv=np.array([q['voltage_candidate_rows'] for q in by]);pp=np.array([q['pf_candidate_rows'] for q in by])
    axs[0,0].barh(yy,vv,color=BLUE,label='상전압 후보');axs[0,0].barh(yy,pp,left=vv,color=ORANGE,label='상역률 후보')
    axs[0,0].set_yticks(yy,labels);axs[0,0].invert_yaxis();axs[0,0].legend(fontsize=9,loc='lower right')
    axs[0,0].set_title('후보는 3개 모듈에 한정',fontsize=12);axs[0,0].set_xlabel('후보 행 수 · 전수 기준')
    axs[0,0].set_xlim(0,max(vv+pp)*1.2);axs[0,0].xaxis.set_major_formatter(comma)
    for i,q in enumerate(by):axs[0,0].text(q['candidate_rows']+400,i,f'{q["candidate_rows"]:,} ({q["rate_of_equipment_percent"]:.3f}%)',va='center',fontsize=9)
    for label,color in zip(labels,[BLUE,ORANGE,TEAL]):
        ad=ac[ac['module(equipment)']==label].set_index('dt').resample('D').size()
        axs[0,1].plot(ad.index,ad.values,lw=1,color=color,label=label)
    axs[0,1].set_title('후보는 전체 기간에 산발적으로 분포',fontsize=12);axs[0,1].set_ylabel('일별 후보 행 수');dates(axs[0,1]);tidy(axs[0,1]);axs[0,1].legend(fontsize=8,ncol=2)
    voltage_ac=ac[ac.voltage_candidate.astype(str).str.lower()=='true'][['voltageR','voltageS','voltageT']].to_numpy().ravel()
    voltage_s=s[s['module(equipment)']=='13(3호기)'][['voltageR','voltageS','voltageT']].to_numpy().ravel()
    pf_ac=ac[ac.pf_candidate.astype(str).str.lower()=='true'][['powerFactorR','powerFactorS','powerFactorT']].to_numpy().ravel()
    pf_s=s[['powerFactorR','powerFactorS','powerFactorT']].to_numpy().ravel()
    for ax,cand,raw,lo,hi,title,xlabel,color in [
        (axs[1,0],voltage_ac,voltage_s,205,225,'상전압 후보 분포와 원시 표본','상전압 (V)',BLUE),
        (axs[1,1],pf_ac,pf_s,80,100,'상역률 후보 분포와 원시 표본','상역률 (%)',ORANGE)]:
        edges=np.linspace(min(cand.min(),raw.min())-1,max(cand.max(),raw.max())+1,50)
        ax.hist(raw,bins=edges,density=True,color=GREY,alpha=.4,label='원시 표본의 상별 값')
        ax.hist(cand,bins=edges,density=True,histtype='step',color=color,lw=2,label='전수 후보 행의 상별 값')
        ax.axvline(lo,color=ORANGE,ls='--',lw=1);ax.axvline(hi,color=ORANGE,ls='--',lw=1)
        ax.set_title(title,fontsize=12);ax.set_xlabel(xlabel);ax.set_ylabel('정규화 밀도');tidy(ax,False);ax.legend(fontsize=8)
    fig.suptitle(f'이상 후보 {anomaly["candidate_rows"]:,}행 · 전체의 {anomaly["candidate_rate_percent"]:.3f}%',fontsize=17,fontweight='bold')
    fig.supxlabel('상전압 205–225V / 상역률 80–100% 범위 밖 탐색 기준 · 현장 범위·정답 라벨 미확인 · 확정 고장으로 판정하지 않음',fontsize=9)
    save(fig,'07_anomaly_candidates')
    ar=''.join(f'<tr><td>{q["module(equipment)"]}</td><td>{q["candidate_rows"]:,}</td><td>{q["rate_of_equipment_percent"]:.3f}%</td><td>'+('상전압' if q['voltage_candidate_rows'] else '상역률')+'</td></tr>' for q in by)
    anomaly_section=f'''<section><h2>7. 이상 후보는 3개 설비의 전압·역률에 집중</h2><p>전형적인 학습 분포를 참고하여 <strong>상전압 205–225 V 밖, 상역률 80–100% 밖, 음수 전력·전류 또는 잘못된 가동 값</strong>을 탐색했습니다. 전체 {anomaly['candidate_rows']:,}행(<strong>{anomaly['candidate_rate_percent']:.3f}%</strong>)이 후보이며, 아래 3개 설비에만 나타납니다. 이 기준은 현장 정상 범위나 가이드의 삽입 라벨로 검증된 고장 판정이 아닙니다.</p><div class="tablewrap"><table><tr><th>설비</th><th>후보 행</th><th>설비 관측 대비</th><th>후보 원인</th></tr>{ar}</table></div>{img('07_anomaly_candidates','세 설비의 전수 이상 후보 수, 일별 발생, 후보와 원시표본 전압 역률 분포')}<p>후보는 학습 기간 전반에 산발적으로 분포하며, 연속된 후보 행을 단순 묶어도 <strong>{anomaly['consecutive_candidate_run_count']:,}개</strong>의 구간이 생깁니다. 모든 행을 개별 알림으로 보내면 사용자가 처리하기 어렵습니다. 설비·유형·기간별 묶음, 지속 시간·심각도와 재알림 제한을 설계해야 합니다.</p><p>우선 13(3호기)의 낮은 상전압과 15(예비건조기)·17(6호기)의 역률 이탈을 원시 시계열에서 확인하고 센서 교정·실제 부하 사건·삽입 이상치 중 어떤 경우인지 라벨을 확보하는 것이 좋습니다. 후보를 자동 삭제하면 전력량·예측 학습도 달라지므로 원시 관측을 보존합니다.</p><p class="caption">분포 비교는 전수 후보 행과 체계적 원시 표본의 정규화 밀도입니다. 후보 행의 모든 상별 값이 포함되어 정상 범위 값도 같이 나타날 수 있습니다.</p></section>'''
baseline=readjson('baseline_audit/baseline_summary.json')
if not baseline: baseline=readjson('baseline_audit.json')
baseline_section=''
if baseline:
    names={'historical_mean':'과거 전체 평균','historical_median':'과거 전체 중앙값',
           'last_7_days_mean':'최근 7일 평균','hour_of_day_mean':'시간대 평균',
           'weekday_hour_mean':'요일 × 시간 평균','seasonal_naive_24h_fixed':'마지막 하루 반복',
           'seasonal_naive_168h_fixed':'마지막 주 반복'}
    valid=baseline['validation']; result=baseline['may_selected_baseline']; scores=valid['scores']; chosen=valid['selected_model']
    score_items=sorted(scores.items(),key=lambda x:x[1]['mae'])
    fig,axs=plt.subplots(2,1,figsize=(13,9),height_ratios=[1.1,1.5],layout='constrained')
    maev=[v['mae'] for k,v in score_items]; labels=[names.get(k,k) for k,v in score_items]
    axs[0].barh(np.arange(len(labels)),maev,color=[TEAL if k==chosen else BLUE for k,v in score_items],height=.6)
    axs[0].set_yticks(np.arange(len(labels)),labels);axs[0].invert_yaxis();axs[0].set_xlim(0,max(maev)*1.18)
    for i,v in enumerate(maev):axs[0].text(v+1,i,f'{v:.3f}',va='center',fontsize=9)
    axs[0].set_title('4/16–4/29 학습 내부 검증 · MAE 낮을수록 좋음',fontsize=12)
    axs[0].set_xlabel('MAE (모듈 합계 kWh · 계층 미확인)');axs[0].grid(axis='x',alpha=.2);axs[0].set_axisbelow(True)
    pred_path=OUT/'baseline_audit'/'may_selected_baseline_predictions.csv'
    pred=pd.read_csv(pred_path,parse_dates=['id'])
    axs[1].plot(pred.id,pred.actual_hourly_pow,color=ORANGE,lw=.8,alpha=.85,label='5월 정답')
    axs[1].plot(pred.id,pred.forecast_provisional_sum_kwh,color=TEAL,lw=1,label='선택·고정된 요일 × 시간 평균')
    axs[1].set_title(f'5월 최종 평가 · MAE {result["scores"]["mae"]:.3f} · 세로축 확대',fontsize=12)
    axs[1].set_ylabel('시간당 값 (가정 단위 kWh)');tidy(axs[1]);axs[1].legend(fontsize=9)
    axs[1].set_xticks(pd.date_range(pred.id.min(),pred.id.max(),freq='4D'));axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    fig.suptitle('단순 기준선의 검증과 5월 평가',fontsize=17,fontweight='bold')
    fig.supxlabel('검증은 4/16 이전 학습만 사용 · 모델 선택 후 전체 학습으로 고정 · 예측에 미래 실측값을 사용하지 않음',fontsize=9)
    save(fig,'06_baseline_evaluation')
    improvement=(1-scores[chosen]['mae']/scores['historical_mean']['mae'])*100
    rows=''.join(f'<tr><td>{names.get(k,k)}'+(' · 선택' if k==chosen else '')+f'</td><td>{v["mae"]:.4f}</td><td>{v["rmse"]:.4f}</td><td>{v["mape_percent"]:.4f}%</td></tr>' for k,v in score_items)
    may_scores=result['scores']
    baseline_section=f'''<section><h2>6. 단순 기준선으로도 작은 오차 · 복잡화 근거는 약함</h2><p>4/16 00시를 예측 기준 시점으로 두고 4/29 23시까지 <strong>336시간</strong>을 검증했습니다. 기준선은 그 이전 학습으로만 만들었으며 하루·주 반복 방식도 미래의 실제 값을 읽지 않았습니다. MAE가 가장 낮은 요일×시간 평균({scores[chosen]['mae']:.4f})은 전체 평균({scores['historical_mean']['mae']:.4f})보다 <strong>{improvement:.4f}%</strong> 낮은 수준입니다. 이 작은 차이만으로 안정적인 계절 효과를 주장하기 어렵습니다.</p><div class="tablewrap"><table><tr><th>기준선</th><th>검증 MAE</th><th>검증 RMSE</th><th>검증 MAPE</th></tr>{rows}</table></div><p>학습 내부에서 선택한 기준선을 4/30 이전의 전체 완전시간 학습으로 다시 계산한 후 고정했습니다. 5월 정답으로 모델·특성·임계값을 선택하지 않았습니다. 최종 505시간 평가는 아래와 같습니다.</p><div class="cards"><div class="card"><div class="value">{may_scores['mae']:.3f}</div><div class="label">MAE · 가정 단위 kWh</div></div><div class="card"><div class="value">{may_scores['rmse']:.3f}</div><div class="label">RMSE · 가정 단위 kWh</div></div><div class="card"><div class="value">{may_scores['mape_percent']:.3f}%</div><div class="label">MAPE</div></div><div class="card"><div class="value">{may_scores['mae']*180:,.0f}원</div><div class="label">동일 가정으로 환산한 요금 MAE</div></div></div>{img('06_baseline_evaluation','학습 내부 단순 기준선 MAE 비교와 5월 고정 예측 대 실제')}<p class="caption">5월 예측은 기준 시점보다 240–744시간 뒤의 값입니다. 에너지·요금 MAE는 학습 모듈 합계와 정답의 범위·단위가 같다는 가정에 의존합니다. 현재 5월 관측 평균의 소폭 하락 때문에 예측 평균이 약간 높습니다.</p><div class="note">출발 모델은 전체 평균 기준선으로 충분합니다. 요일×시간 평균은 비교 대상으로 유지하되, 차이가 다른 시간 순서 검증 구간에서도 지속되는지 확인한 뒤 복잡한 모델로 확대합니다.</div></section>'''

# The final overview includes the confirmed anomaly review and baseline audit.
if metrics.get('anomalies') and baseline:
    fig,axs=plt.subplots(3,2,figsize=(15,13),layout='constrained')
    daily_panel(axs[0,0]);axs[0,0].set_title('일평균 추이 · 세로축 확대',fontsize=12);axs[0,0].set_ylabel('시간당 에너지 (kWh)')
    axs[0,1].bar(np.arange(len(e)),eq_full/1e6,color=TEAL)
    axs[0,1].set_xticks(np.arange(len(e)),[x.split('(')[0] for x in e.equipment]);axs[0,1].set_xlabel('모듈 번호');axs[0,1].set_ylabel('관측 에너지 (백만 kWh)');axs[0,1].set_title('설비별 에너지 · 0축',fontsize=12);tidy(axs[0,1],False)
    axs[1,0].imshow(heat_pct.values,aspect='auto',cmap='RdBu_r',vmin=-vlim,vmax=vlim)
    axs[1,0].set_yticks(np.arange(7),['월','화','수','목','금','토','일']);axs[1,0].set_xticks(np.arange(0,24,3));axs[1,0].set_xlabel('시간대');axs[1,0].set_title(f'요일 × 시간 편차 · ±{vlim:.3f}% 색 범위',fontsize=12)
    axs[1,1].hist(s.activePower,bins=40,color=ORANGE,edgecolor='white',linewidth=.4)
    axs[1,1].set_title('원시 표본 유효전력 분포',fontsize=12);axs[1,1].set_xlabel('activePower (가이드 단위 kW)');axs[1,1].set_ylabel('표본 수');tidy(axs[1,1])
    bvals=[q['candidate_rows'] for q in by]
    axs[2,0].barh(np.arange(len(by)),bvals,color=[BLUE,ORANGE,ORANGE],height=.6)
    axs[2,0].set_yticks(np.arange(len(by)),[q['module(equipment)'] for q in by]);axs[2,0].invert_yaxis();axs[2,0].set_xlim(0,max(bvals)*1.27)
    for i,q in enumerate(by):axs[2,0].text(q['candidate_rows']+300,i,f'{q["candidate_rows"]:,}',va='center',fontsize=9)
    axs[2,0].xaxis.set_major_formatter(comma);axs[2,0].set_title(f'전수 이상 후보 {anomaly["candidate_rate_percent"]:.3f}% · 3개 설비',fontsize=12);axs[2,0].set_xlabel('탐색 기준 밖 행 수 · 확정 고장 아님')
    axs[2,1].barh(np.arange(len(score_items)),[v['mae'] for k,v in score_items],color=[TEAL if k==chosen else BLUE for k,v in score_items],height=.6)
    axs[2,1].set_yticks(np.arange(len(score_items)),[names.get(k,k) for k,v in score_items]);axs[2,1].invert_yaxis();axs[2,1].set_title('학습 내부 검증 · 단순 평균과 시간 평균 유사',fontsize=12);axs[2,1].set_xlabel('검증 MAE (가정 단위 kWh)')
    fig.suptitle('RTU 프로젝트 · 기초 분석과 검증',fontsize=20,fontweight='bold')
    fig.supxlabel('학습 3,369.6만 행 · 13개 모듈 · localtime 기준 · 모듈 합계의 계층 미확인 · 단위·시간 대응 검증 필요',fontsize=10)
    save(fig,'overview')

eq_table=''.join('<tr><td>'+html.escape(str(r.equipment))+f'</td><td>{num(eq_full.loc[r.equipment])}</td><td>{num(eq_means.loc[r.equipment],3)}</td><td>{num(r.activePower_std,2)}</td><td>{100*r.low_pf_rate:.2f}%</td></tr>' for r in e.itertuples())
interactive={'모듈 합계 (계층 미확인)':[[d.strftime('%Y-%m-%d'),round(float(v),4)] for d,v in daily['mean'].items()]}
for name,grp in hc.groupby('module(equipment)',sort=False):
    daily_eq=grp.set_index('hour').energy_kwh.resample('D').agg(['mean','count'])
    interactive[name]=[[d.strftime('%Y-%m-%d'),round(float(v),4)] for d,v in daily_eq[daily_eq['count']==24]['mean'].items()]

report='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>RTU 프로젝트 기초 분석</title>
<style>
:root{--ink:#162b44;--muted:#53667e;--line:#d9e2ed;--teal:#007f86;--orange:#dc6b26}*{box-sizing:border-box}body{margin:0;background:#eef3f8;color:var(--ink);font-family:"Malgun Gothic","Apple SD Gothic Neo",sans-serif;line-height:1.65}main{max-width:1120px;margin:auto;padding:38px 24px 70px}header{background:#152e47;color:#fff;padding:34px 38px;border-radius:18px;margin-bottom:24px}h1{font-size:30px;margin:3px 0 12px;letter-spacing:-1px}header p{color:#d9e6f0;margin:8px 0}.eyebrow{color:#66d6ce;font-size:13px;letter-spacing:2px}h2{font-size:23px;margin:0 0 18px;letter-spacing:-.6px}h3{font-size:17px;margin:24px 0 8px}section{background:#fff;border:1px solid var(--line);border-radius:16px;padding:30px;margin:22px 0}p{margin:10px 0}strong{font-weight:700}img{width:100%;height:auto;display:block;margin-top:20px;border-radius:8px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{border:1px solid var(--line);border-radius:12px;padding:19px;background:#f8fafc}.card .value{font-size:27px;font-weight:700;line-height:1.3}.label{font-size:12px;color:var(--muted);margin-top:6px}.note{padding:15px 18px;border-left:4px solid var(--teal);background:#f0f8f8;color:#244b54;border-radius:4px;margin:20px 0}.caution{border-left-color:var(--orange);background:#fff5ec;color:#75502e}table{border-collapse:collapse;width:100%;font-size:13px;margin-top:12px}td,th{padding:11px 9px;border-bottom:1px solid var(--line);text-align:right}td:first-child,th:first-child{text-align:left}th{background:#f0f4f8;color:#334155}ul,ol{padding-left:23px}li{margin:8px 0}small,.caption{font-size:12px;color:var(--muted)}code{background:#edf3f8;border-radius:4px;padding:2px 5px}select{padding:10px 14px;border:1px solid #b9cadc;background:#fff;border-radius:8px;font-size:14px;color:var(--ink);max-width:100%}svg{display:block;width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#f6f8fc;border-radius:9px;padding:20px}details{margin-top:18px}summary{cursor:pointer;color:var(--teal);font-weight:700}footer{padding:5px 10px;color:var(--muted);font-size:12px}@media(max-width:760px){main{padding:18px 12px}header,section{padding:24px 18px}.cards{grid-template-columns:repeat(2,1fr)}h1{font-size:25px}h2{font-size:21px}.tablewrap{overflow:auto}}@media print{body{background:#fff}section,header{break-inside:avoid;border-radius:0}header{background:#fff;color:var(--ink)}header p{color:var(--muted)}main{max-width:none;padding:0}select{display:none}.interactive{display:none}}
</style></head><body><main>'''
report+=f'''<header><div class="eyebrow">PROJECT DATA REVIEW · 2026.10.11</div><h1>RTU 데이터에서 출발하는 예측·이상탐지</h1><p>프로젝트 가이드와 학습 데이터, 5월 정답을 함께 확인한 기초 분석입니다. 작은 변동을 실제 규모와 함께 읽고, 모델링 전에 확정해야 할 데이터 정의를 정리했습니다.</p><p class="caption" style="color:#b9ccdf">범위: 제공 파일의 기술통계·시각화·품질 감사·실행 인사이트. 원본 데이터를 변경하지 않았습니다.</p></header>
<div class="cards"><div class="card"><div class="value">{num(e.records.sum())}</div><div class="label">학습 원시 관측 행</div></div><div class="card"><div class="value">{module_count}개</div><div class="label">모듈 · 각 모듈 동일 관측 수</div></div><div class="card"><div class="value">{num(len(g))}시간</div><div class="label">학습 완전시간 · 720행/모듈/시간</div></div><div class="card"><div class="value">{num(len(m))}시간</div><div class="label">5월 정답 · 끝 시각 포함</div></div></div>
<section><h2>목표와 분석 기준</h2><p>가이드의 최종 목표는 <strong>전기요금·탄소배출 예측, 삽입 이상치 탐지, 분석 에이전트, Streamlit 대시보드</strong>입니다. 평가지표는 시간별 예측의 MAE이며, 대시보드는 시간·설비 필터와 이상 후보의 근거를 함께 보여주는 방향이 적합합니다.</p><div class="tablewrap"><table><tr><th>파일</th><th>역할 / 범위</th></tr><tr><td>traing-dataset/rtu_data_full.csv</td><td>학습 · 2024-12-01 00:00–2025-04-30 00:00</td></tr><tr><td>may-dataset/rtu_ground_truth_may.csv</td><td>정답·최종 검증 · 2025-05-10 00:00–05-31 00:00</td></tr><tr><td>프로젝트가이드.pdf</td><td>프로젝트 요구사항 및 단위·환산 가정</td></tr></table></div><div class="note"><strong>시간당 에너지 = Σ activePower × 5/3600.</strong> 가이드의 activePower 단위 kW와 5초 평균 가정을 적용했습니다. 요금은 kWh × 180원, 탄소는 kWh × 0.424 kgCO₂로 환산하는 프로젝트 가정입니다. 실제 계약 요금제·공식 계수 검증 결과가 아닙니다.</div><div class="note caution"><strong>모듈 합계는 계층 미확인.</strong> 공장 전체와 하위 설비 계측이 겹칠 수 있습니다. 아래 합계를 공장 전력량·절감액으로 확정하지 않습니다. 전력 단위와 물리적 정합성도 모델링 전에 확인해야 합니다.</div>{img('overview','전체 분석 핵심 차트: 일평균, 설비별 에너지, 요일 시간 편차, 원시 전력 분포, 이상 후보, 기준선 검증')}</section>
<section><h2>1. 전체 수준은 안정적이며 큰 생산 주기는 확인되지 않음</h2><p>학습의 모듈 합계 시간당 평균은 <strong>{num(g.energy.mean(),2)} kWh</strong>, 표준편차는 {num(g.energy.std(),2)} kWh, 변동계수는 <strong>{train_cv:.3f}%</strong>입니다. 5월 정답 평균은 {num(m.hourly_pow.mean(),2)}이고 학습 평균 대비 <strong>{delta_pct:+.3f}%</strong>입니다. 기간 차이는 기술적으로 관찰되지만 운영 변화의 원인이 확정된 것은 아닙니다.</p><p>4월 30일 00시에는 모듈별 관측이 1행씩만 있어 일평균·시간 패턴에서 제외했습니다. 5월 31일 00시는 정답에 포함된 평가 시간이며 일평균 차트에서는 부분일로 제외했습니다.</p>{img('01_daily_train_may','학습과 5월의 완전한 하루 평균 비교: 확대 축과 0축')}<p class="caption">상단의 세로축은 작은 변동을 보기 위해 확대했습니다. 하단은 0축으로 실제 수준을 확인합니다. 학습 끝과 5월 정답 시작 사이의 빈 구간을 연결·보간하지 않았습니다.</p></section>
<section class="interactive"><h2>설비를 선택해 일평균 확인</h2><p>설비별 완전한 하루의 시간당 평균 에너지를 확인합니다. 세로축은 선택한 설비의 관측 범위에 맞춰 확대됩니다.</p><label for="equipment">설비 </label><select id="equipment"></select><p class="caption" id="rangeNote"></p><svg id="equipmentChart" viewBox="0 0 940 300" role="img" aria-label="선택 설비 일평균 시계열 차트"></svg></section>
<section><h2>2. 설비 간 규모 차이가 매우 작음</h2><p>각 모듈의 전체 평균 유효전력은 약 {num(e.activePower_mean.mean(),1)} kW이며, 완전시간 에너지의 설비 평균 대비 편차는 <strong>{eq_dev.min():+.3f}%–{eq_dev.max():+.3f}%</strong>에 불과합니다. 설비 명칭만으로 부하·용량·가동 패턴이 다르다고 가정하기 어렵습니다. 에너지 총량 순위가 설비 개선 우선순위라는 결론도 지지되지 않습니다.</p>{img('02_equipment_energy','설비별 완전시간 관측 에너지와 평균 대비 상대편차')}<details><summary>설비별 요약표 보기</summary><div class="tablewrap"><table><tr><th>설비</th><th>관측 에너지 (kWh)</th><th>시간당 평균 (kWh)</th><th>원시 전력 표준편차 (kW)</th><th>3상 평균 PF&lt;90</th></tr>{eq_table}</table></div><p class="caption">에너지는 12/1–4/29의 완전시간 합계. 역률 비율은 전수 원시 관측 기준이며 임계값 90은 탐색용 기준입니다.</p></details></section>
<section><h2>3. 요일·시간 효과는 작음</h2><p>요일×시간 셀 평균의 전체 평균 대비 편차는 <strong>{np.nanmin(heat_pct):+.3f}%–{np.nanmax(heat_pct):+.3f}%</strong>입니다. 색상이 진한 셀도 절대 차이는 작습니다. 강한 출퇴근·주말·교대조 주기나 특정 시간대의 절감 효과를 주장하기보다, 일정한 평균 기준선보다 예측을 개선하는지 시간 순서 검증으로 확인해야 합니다.</p>{img('03_weekday_hour_heatmap','학습 모듈 합계 에너지의 요일 시간 상대편차 히트맵')}</section>
<section><h2>4. 원시 변동과 집계 변동을 구분</h2><p>일정 간격으로 추출한 {num(sample_n)}행 표본에서 유효전력·역률·전류·전압 분포를 살펴봤습니다. 원시 유효전력의 표준편차는 약 {num(e.activePower_std.mean(),1)} kW로, 시간 합계의 작은 변동보다 큽니다. 5초 변동을 시간 집계가 평균화하는 효과를 구분해야 합니다.</p><p>불균형 지표는 <code>max(|상별 값−3상 평균|)/3상 평균 × 100</code>입니다. 공학적 표준에 따른 이상 판정값으로 확정하지 않았습니다. 이상 후보는 시계열 변화, 가동 상태, 누적계·전류·전압과의 일관성을 함께 확인해야 합니다.</p>{img('04_sample_sensor_distributions','원시 센서 표본 여섯 분포: 유효전력 역률 전류 전압 전류불균형 전압불균형')}</section>
<section><h2>5. 5월 정답은 작은 변동과 선형 환산 구조</h2><p>정답의 <code>hourly_pow</code> 평균은 {num(m.hourly_pow.mean(),2)}, 범위는 {num(m.hourly_pow.min(),2)}–{num(m.hourly_pow.max(),2)}, 변동계수는 <strong>{may_cv:.3f}%</strong>입니다. <code>may_bill=hourly_pow×180</code>, <code>may_carbon=hourly_pow×0.424</code>가 수치 오차 범위에서 일치합니다. 요금·탄소를 각각 독립 모델로 예측할 필요성은 낮고 에너지 예측을 동일 가정으로 환산하는 구조가 적합합니다.</p><p><code>agg_pow</code>는 모든 행에서 같은 값이며 505시간 <code>hourly_pow</code> 합계와 일치합니다. 전체 미래 구간의 정답을 담으므로 예측 특성에 사용하면 누수가 발생합니다. 5월 원시 RTU가 없어 <code>hourly_pow</code>의 단위를 직접 대조하지 못했으며, 아래 kWh 해석은 가이드·환산식·학습 규모를 함께 본 추정입니다.</p>{img('05_may_zoom_zero','5월 정답 변동의 확대 축과 0축 비교')}</section>
'''
report+=baseline_section+anomaly_section
report+='''<section><h2>품질 이슈와 프로젝트 해석의 한계</h2><div class="tablewrap"><table><tr><th>확인 사항</th><th>분석에 미치는 영향 / 처리</th></tr><tr><td>timestamp와 localtime 불일치</td><td>첫 행에서 Unix UTC와 localtime 차이는 −8시간, Asia/Seoul 변환 후 차이는 −17시간입니다. 가이드가 지정한 localtime을 모든 날짜·요일·시간 경계에 사용했습니다. 원본 시간 의미를 확정한 뒤 변환 규칙을 고정해야 합니다.</td></tr><tr><td>원본 시간 정렬</td><td>모듈별 원본 순서에 역순 경계가 관측됩니다. 집계는 시간으로 정렬한 뒤 사용하며 동일 모듈·시간 중복과 결측을 별도로 감사합니다.</td></tr><tr><td>관측 범위 경계</td><td>각 모듈 3,600개 완전시간과 4/30 00시 1행. 부분시간을 정상시간과 혼합하면 말일의 가짜 급락이 생깁니다.</td></tr><tr><td>operation=1</td><td>모든 원시 관측에서 가동 값이 1입니다. 현재 파일에서는 비가동 전환·절감 운전 효과를 추정할 수 없습니다.</td></tr><tr><td>누적 에너지 단위 표기 불일치</td><td>가이드의 Wh/kWh 표기가 서로 다릅니다. 누적계 차분과 전력 적분의 비율을 검증하며 리셋·음수 차분을 소비로 직접 사용하지 않습니다.</td></tr><tr><td>설비별 분포가 매우 유사</td><td>현실 공정의 용량·주기 구조인지, 교육용 데이터 생성 방식인지 확인해야 합니다. 고도화 모델의 효과나 설비 간 최적화 가능성을 미리 가정하지 않습니다.</td></tr><tr><td>삽입 이상치 라벨 없음</td><td>분포·물리·변화량 기준에서 찾은 점은 이상 후보입니다. 라벨 없이 이상탐지 정확도·재현율을 계산하거나 임의 제거하지 않습니다.</td></tr></table></div></section>
<section><h2>프로젝트를 위한 인사이트와 실행 순서</h2><ol><li><strong>데이터 계약부터 확정.</strong> timestamp/localtime의 의미, activePower 스케일, 누적계 단위, 공장/하위 모듈 계층을 확인합니다. 정합성이 확정될 때까지 예측 에너지·요금·탄소에는 사용한 가정과 관측률을 표시합니다.</li><li><strong>작은 기준선으로 시작.</strong> 학습 내부에서 시간 순서를 보존한 검증 구간을 두고 전체 평균·최근 평균·요일×시간 평균을 비교합니다. 전처리·임계값은 학습 구간에서만 적합합니다. 5월 정답은 선택이 끝난 모델의 최종 평가에만 사용합니다.</li><li><strong>한 예측량을 환산.</strong> 에너지 예측 후 가이드의 180원/kWh·0.424 kgCO₂/kWh로 요금·탄소를 계산합니다. 같은 평가 시간에서는 MAE도 동일 계수로 환산할 수 있습니다. 계수는 대시보드 설정으로 분리합니다.</li><li><strong>이상 후보를 맥락과 함께 관리.</strong> 전력 변화·상별 역률·전류/전압·누적계 일관성을 함께 사용해 후보를 찾고 원시 관측을 조회합니다. 현재 operation은 상수이므로 가동 전환을 정상/이상 구분 근거로 사용할 수 없습니다. 탐지한 점을 자동으로 삭제하지 않습니다.</li><li><strong>대시보드는 해석 가능성 중심.</strong> 실제/예측, MAE, 관측률, 날짜·설비 필터, 이상 후보의 근거와 전력 단위·모듈 계층 주의 문구를 제공합니다. 확대 축과 0축을 전환할 수 있게 해 작은 편차가 큰 효과로 읽히는 것을 줄입니다.</li></ol><div class="note">현재 데이터는 단순한 평균 예측이 강한 기준선이 될 가능성이 있습니다. 데이터 정의를 확정하고 기준선 대비 검증 개선이 확인된 뒤 복잡한 모델·운영 최적화로 확장하는 것이 합리적입니다.</div></section>
<footer>재현: output/analysis/scripts/build_visual_report.py · PNG/SVG: output/analysis/figures · 원본 데이터/가이드 수정 없음 · 이미지와 설비별 시계열을 내장한 오프라인 HTML · 원문 인증값은 보고서에 포함하지 않음</footer>'''

payload=json.dumps(interactive,ensure_ascii=False).replace('</','<\\/')
report+='''<script>
const series=PAYLOAD;const select=document.getElementById('equipment');const svg=document.getElementById('equipmentChart');
Object.keys(series).forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=k;select.appendChild(o)});
const esc=s=>s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
function draw(){const name=select.value,data=series[name],values=data.map(v=>v[1]);let mn=Math.min(...values),mx=Math.max(...values);const pad=(mx-mn||1)*.16;mn-=pad;mx+=pad;const L=83,R=920,T=20,B=254;const sx=i=>L+i/(data.length-1)*(R-L),sy=v=>B-(v-mn)/(mx-mn)*(B-T);let out='';for(let j=0;j<=4;j++){const v=mn+j/4*(mx-mn),y=sy(v);out+=`<line x1="${L}" x2="${R}" y1="${y}" y2="${y}" stroke="#dce5ef"/><text x="${L-10}" y="${y+4}" text-anchor="end" font-size="12" fill="#53667e">${v.toLocaleString('ko-KR',{maximumFractionDigits:1})}</text>`;}out+=`<polyline points="${data.map((d,i)=>sx(i)+','+sy(d[1])).join(' ')}" fill="none" stroke="#007f86" stroke-width="2"/>`;[0,Math.floor(data.length/3),Math.floor(2*data.length/3),data.length-1].forEach(i=>{out+=`<text x="${sx(i)}" y="281" text-anchor="middle" font-size="12" fill="#53667e">${esc(data[i][0])}</text>`});svg.innerHTML=out;svg.setAttribute('aria-label',name+'의 일평균 시간당 에너지, 세로축 확대');document.getElementById('rangeNote').textContent=`${name} · 세로축 확대: ${mn.toLocaleString('ko-KR',{maximumFractionDigits:1})}–${mx.toLocaleString('ko-KR',{maximumFractionDigits:1})} kWh · 완전한 하루 ${data.length}일`}
select.addEventListener('change',draw);draw();</script></main></body></html>'''.replace('PAYLOAD',payload)
# Final exhaustive audit replaces provisional time descriptions with confirmed evidence.
report=report.replace('<section><h2>품질 이슈와 프로젝트 해석의 한계</h2><div class="tablewrap"><table>',
    '<section><h2>품질 이슈와 프로젝트 해석의 한계</h2><div class="tablewrap"><table class="quality-table">')
report=report.replace('</style>',
    '.quality-table th:first-child,.quality-table td:first-child{width:190px;min-width:190px;word-break:keep-all}.quality-table th:last-child,.quality-table td:last-child{text-align:left}</style>')
report=report.replace('분포·물리·변화량 기준에서 찾은 점','전압·역률 임계값 기준으로 찾은 점')
report=report.replace('첫 행에서 Unix UTC와 localtime 차이는 −8시간, Asia/Seoul 변환 후 차이는 −17시간입니다. 가이드가 지정한 localtime을 모든 날짜·요일·시간 경계에 사용했습니다. 원본 시간 의미를 확정한 뒤 변환 규칙을 고정해야 합니다.',
    'Unix timestamp는 가이드의 한국시간 기준과 일치하지 않습니다. localtime을 시간대 없는 값으로 읽었을 때 Unix UTC와 차이는 초기 −8시간이며 2025-03-09 03시부터 −7시간으로 바뀝니다. 모든 분석은 전수 검증한 localtime을 사용했습니다. 원본 생성·시간대 규칙을 확인해야 합니다.')
report=report.replace('<td>원본 시간 정렬</td><td>모듈별 원본 순서에 역순 경계가 관측됩니다. 집계는 시간으로 정렬한 뒤 사용하며 동일 모듈·시간 중복과 결측을 별도로 감사합니다.',
    '<td>Unix 중복·역순 경계</td><td>같은 모듈의 Unix timestamp 기준 추가 중복은 9,360행(13×720), 역순 전환은 13회입니다. 3/9 03시의 오프셋 전환과 연결되며 localtime 기준 중복은 0입니다. Unix 키로 중복 제거하면 유효한 관측을 잃을 수 있습니다.')
report=report.replace('<tr><td>관측 범위 경계</td>',
    '<tr><td>전수 결측·localtime 간격</td><td>결측 셀 0, 모듈별 localtime 중복 0, 5초가 아닌 간격 0입니다. 내부 3,600시간은 모든 13개 모듈에서 720개의 고유 관측이 확인되었습니다.</td></tr><tr><td>관측 범위 경계</td>')
report=report.replace('가이드의 Wh/kWh 표기가 서로 다릅니다. 누적계 차분과 전력 적분의 비율을 검증하며 리셋·음수 차분을 소비로 직접 사용하지 않습니다.',
    '가이드의 Wh/kWh 표기가 서로 다릅니다. 실제 누적계 증가량/가이드 kW를 적분한 kWh의 설비별 비율은 0.9999999–1.0000001로 일치하고 음수 차분은 0입니다. 파일상 누적계는 kWh 해석이 정합적이지만 계측기 스케일도 확인해야 합니다.')
report=report.replace('일정 간격으로 추출한 '+num(sample_n)+'행 표본에서',
    '30만 행 청크에서 2,000행 간격으로 추출한 '+num(sample_n)+'행 체계적 표본에서')
report=report.replace('5초 변동을 시간 집계가 평균화하는 효과를 구분해야 합니다.</p>',
    '5초 변동을 시간 집계가 평균화하는 효과를 구분해야 합니다.</p><p>표본에서 유효전력과 3상 전류 합계의 상관계수는 <strong>0.999365</strong>로 매우 높습니다. 이는 같은 시점의 센서 간 관계이며 미래의 실측 전류는 예측 시점에 알려져 있지 않습니다. 미래 전류를 에너지 예측 특성으로 사용하면 정답 정보가 섞입니다.</p>')
(OUT/'project_analysis.html').write_text(report,encoding='utf-8')
print(json.dumps({'figures':len(list(FIG.glob('*.png'))),'report':str(OUT/'project_analysis.html'),'descriptive':descriptive},ensure_ascii=True,indent=2))
