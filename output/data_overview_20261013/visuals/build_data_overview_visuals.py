"""Publication-ready descriptive H/D/W/MS EDA from audited existing artifacts.

All writes are under this visuals directory. Source/full resampling data and
earlier reports/models are preserved. No forecasting training or May-label read.
"""
from pathlib import Path
import sys,os,json,hashlib
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'output/analysis/.plot_deps'))
os.environ['MPLCONFIGDIR']=str(OUT/'tmp/mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import fontManager
from matplotlib.ticker import FuncFormatter
fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10,
    'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold',
    'figure.facecolor':'white','axes.facecolor':'white','svg.fonttype':'none','savefig.facecolor':'white'})
FIG=OUT/'figures';FIG.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'output/resampling_modeling_20261013/data'
TARGET=ROOT/'output/analysis/baseline_audit/complete_hour_module_sum.csv'
MODULE=ROOT/'output/analysis/hourly_by_equipment.csv'
EQUIPMENT=ROOT/'output/analysis/equipment_summary.csv'
SAMPLE=ROOT/'output/modeling/diagnosis/uniform_raw_sample.csv.gz'
SAMPLE_META=ROOT/'output/modeling/diagnosis/sample_summary.json'
NAMES={'hourly':'시간 H','daily':'일 D','weekly':'주 W-SUN','monthly':'월 MS'}
BLUE='#255e8c';TEAL='#197d79';ORANGE='#b95b22';DARK='#252b33';GRAY='#738393'
FORMAT=FuncFormatter(lambda x,p:f'{x:,.0f}')

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(Path,pd.Timestamp)):return str(x)
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def write(path,value):Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def stats(v):
    a=np.asarray(v,float)
    return {'n':len(a),'mean':float(a.mean()),'std_ddof1':float(a.std(ddof=1)),
            'cv_percent':float(a.std(ddof=1)/a.mean()*100),'minimum':float(a.min()),'maximum':float(a.max())}
def finish(fig,stem):
    paths={}
    for ext in ['png','svg']:
        path=FIG/f'{stem}.{ext}';fig.savefig(path,dpi=190,bbox_inches='tight');paths[ext]=str(path)
    plt.close(fig);return paths
def add_chart(captions,key,paths,title,caption,evidence,sources,notes):
    captions[key]={'title':title,'caption':caption,'paths':paths,'evidence':evidence,
                   'source_files':[str(p) for p in sources],'interpretation_limits':notes}
def note(fig,text):fig.text(.5,.016,text,ha='center',fontsize=9,color='#374151')
def axes_style(ax):ax.grid(alpha=.18);ax.yaxis.set_major_formatter(FORMAT)
def boolcol(series):
    if series.dtype==bool:return series
    values=series.astype(str).str.lower();assert values.isin(['true','false']).all();return values.eq('true')

def load_sources():
    assert sha(TARGET)==read(DATA/'data_verification.json')['target_sha256']
    f=pd.read_csv(TARGET,float_precision='round_trip');y=pd.Series(f.provisional_sum_energy_kwh.to_numpy(float),index=pd.to_datetime(f.hour))
    assert len(y)==3600 and y.index.equals(pd.date_range('2024-12-01','2025-04-29 23:00',freq='h'))
    tables={};checks={}
    for name in NAMES:
        path=DATA/f'{name}_dataset.csv';d=pd.read_csv(path,float_precision='round_trip')
        for c in ['period_start','period_end_exclusive','first_observed','last_observed','available_at']:d[c]=pd.to_datetime(d[c])
        d['complete']=boolcol(d.complete)
        assert int(d.observed_hours.sum())==3600
        energy_error=float(abs(d.energy_kwh.sum()-y.sum()));assert energy_error<1e-6
        rate_error=float(np.abs(d.energy_kwh/d.observed_hours-d.rate_kwh_per_hour).max());assert rate_error<1e-8
        assert np.allclose(d.coverage,d.observed_hours/d.expected_hours,rtol=0,atol=1e-14)
        assert (d.last_observed<d.period_end_exclusive).all() and (d.first_observed>=d.period_start).all()
        assert (d.loc[d.complete,'observed_hours']==d.loc[d.complete,'expected_hours']).all()
        checks[name]={'energy_sum_vs3600h_error_kwh':energy_error,'energy_per_observed_hour_vs_rate_max_error':rate_error}
        tables[name]=d
    modules=pd.read_csv(MODULE,float_precision='round_trip');modules['hour']=pd.to_datetime(modules.hour)
    modules['complete_720']=boolcol(modules.complete_720);complete=modules[modules.complete_720].copy()
    assert len(complete)==46800 and complete['module(equipment)'].nunique()==13
    check=complete.groupby('hour').energy_kwh.sum().reindex(y.index)
    assert np.abs(check.to_numpy()-y.to_numpy()).max()<1e-7
    equipment=pd.read_csv(EQUIPMENT,float_precision='round_trip')
    sample=pd.read_csv(SAMPLE,float_precision='round_trip');sample['dt']=pd.to_datetime(sample.dt)
    meta=read(SAMPLE_META);assert len(sample)==meta['sample_rows']==187200
    sizes=sample.groupby(['module(equipment)',sample.dt.dt.floor('h')]).size()
    assert len(sizes)==46800 and sizes.eq(4).all() and sample.dt.max()<pd.Timestamp('2025-04-30')
    return y,tables,complete,equipment,sample,meta,checks

def period_charts(y,tables,captions):
    for name in NAMES:
        d=tables[name];complete=d[d.complete];partial=d[~d.complete]
        fig,axes=plt.subplots(2,1,figsize=(13,7.2),sharex=True)
        x=d.period_start
        if name=='monthly':
            colors=[BLUE if c else 'white' for c in d.complete]
            edge=[BLUE if c else DARK for c in d.complete]
            bars=axes[0].bar(x,d.energy_kwh,width=18,color=colors,edgecolor=edge,linewidth=1.5)
            for bar,(_,r) in zip(bars,d.iterrows()):
                if not r.complete:bar.set_hatch('////')
                axes[0].text(r.period_start,r.energy_kwh+max(d.energy_kwh)*.018,
                    f'{int(r.observed_hours/24)}일' if r.complete else f'{int(r.observed_hours/24)}/{int(r.expected_hours/24)}일 · 부분',ha='center',fontsize=10)
            axes[0].set_ylim(0,max(d.energy_kwh)*1.17)
        else:
            axes[0].plot(complete.period_start,complete.energy_kwh,color=BLUE,lw=.85 if name=='hourly' else 1.35,
                         marker='o' if name=='weekly' else None,ms=4,label='완전 기간')
            if len(partial):
                axes[0].bar(partial.period_start,partial.energy_kwh,width=4,color='white',edgecolor=DARK,hatch='////',linewidth=1.3,label='관측된 부분 기간')
                for _,r in partial.iterrows():
                    label=f'{r.first_observed:%m/%d}~{r.last_observed:%m/%d}\n{int(r.observed_hours)}/{int(r.expected_hours)}h'
                    alignment='left' if r.period_start==x.iloc[0] else 'right'
                    axes[0].annotate(label,(mdates.date2num(r.period_start),r.energy_kwh),xytext=(0,15),textcoords='offset points',ha=alignment,fontsize=9)
                axes[0].set_ylim(0,d.energy_kwh.max()*1.2);axes[0].legend(loc='lower center',ncol=2)
            else:axes[0].set_ylim(0,d.energy_kwh.max()*1.13)
        axes[1].plot(complete.period_start,complete.rate_kwh_per_hour,color=TEAL,lw=.85 if name=='hourly' else 1.3,
                     marker='o' if name in ['weekly','monthly'] else None,ms=4,label='완전 기간 평균')
        if len(partial):
            axes[1].scatter(partial.period_start,partial.rate_kwh_per_hour,s=70,facecolors='white',edgecolors=DARK,linewidths=1.5,label='부분 기간: 관측 시간으로 나눈 평균',zorder=4)
            axes[1].legend(loc='best',fontsize=9)
        axes[1].axhline(float(y.mean()),color=GRAY,ls='--',lw=.9,label='전체3600h 가중 평균')
        for ax in axes:axes_style(ax)
        axes[0].set_ylabel('관측 에너지 합계\n(잠정 kWh)');axes[1].set_ylabel('관측 평균 에너지율\n(잠정 kWh/h, 축 확대)')
        axes[0].set_title(f'{NAMES[name]} 집계: 에너지 합계와 관측 시간당 평균을 분리')
        axes[1].set_xlabel('기간 시작일 (기록된 localtime)')
        if name=='monthly':
            axes[1].set_xticks(x,[t.strftime('%Y-%m') for t in x]);axes[1].set_xlim(x.iloc[0]-pd.Timedelta(days=16),x.iloc[-1]+pd.Timedelta(days=16))
        else:
            axes[1].xaxis.set_major_locator(mdates.MonthLocator());axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            if name=='weekly':axes[1].set_xlim(x.iloc[0]-pd.Timedelta(days=2.2),x.iloc[-1]+pd.Timedelta(days=2.2))
            else:axes[1].set_xlim(x.iloc[0],x.iloc[-1])
        footer='합계는 관측 길이에 비례합니다. 평균 에너지율은 정격 대비 부하율(%)이 아니며 계측 계층이 미확정된 모듈 합입니다.'
        if name=='hourly':footer='1시간 버킷에서는 에너지와 시간당 평균의 수치가 같습니다. 상단0축·하단확대축을 사용했으며 생산 주기나 예측력의 증거가 아닙니다.'
        if name=='monthly':footer='2월28일과31일 월의 합계 차이는 주로 달 길이 효과입니다. 4월은29/30일 부분 월이며 남은24시간을 보간하지 않았습니다.'
        footer+='\n점선=전체3,600시간 가중 평균. 부하 정격에 대한 비율(%) 자료는 제공되지 않았습니다.'
        note(fig,footer);fig.tight_layout(rect=(0,.055,1,1))
        title=f'{NAMES[name]} 에너지 및 평균 에너지율'
        evidence={'period_count':len(d),'complete_count':len(complete),'partial_count':len(partial),
                  'observed_hours':int(d.observed_hours.sum()),'total_energy_kwh':float(d.energy_kwh.sum()),
                  'weighted_rate_kwh_per_hour':float(d.energy_kwh.sum()/d.observed_hours.sum()),
                  'complete_rate_statistics':stats(complete.rate_kwh_per_hour),
                  'partial_periods':partial[['period_start','period_end_exclusive','first_observed','last_observed','observed_hours','expected_hours','coverage','energy_kwh','rate_kwh_per_hour']].to_dict('records')}
        add_chart(captions,f'{name}_energy_rate',finish(fig,f'{name}_energy_rate'),title,footer,evidence,
                  [TARGET,DATA/f'{name}_dataset.csv'],['기본 기술통계 시각화이며 예측 모델 학습·검증이 아님','평균 에너지율 축은 확대됨','기록된 localtime 기준'])
    zoom=tables['hourly'].iloc[:72]
    fig,axes=plt.subplots(2,1,figsize=(13,6.3),sharex=True)
    axes[0].plot(zoom.period_start,zoom.energy_kwh,color=BLUE,lw=1.2,marker='o',ms=2);axes[0].set_ylim(0,zoom.energy_kwh.max()*1.1)
    axes[1].plot(zoom.period_start,zoom.rate_kwh_per_hour,color=TEAL,lw=1.2,marker='o',ms=2)
    for ax in axes:axes_style(ax)
    axes[0].set_ylabel('1시간 에너지\n(잠정 kWh, 0축)');axes[1].set_ylabel('평균 에너지율\n(잠정 kWh/h, 축 확대)')
    axes[0].set_title('연속72시간 확대 | 2024-12-01 00시~12-03 23시')
    axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=12));axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H시'))
    axes[1].set_xlim(zoom.period_start.iloc[0],zoom.period_start.iloc[-1]);axes[1].set_xlabel('관측 시간')
    footer='실제 연속 완전 시간 집계72개를 표시했습니다. 보간·이동평균·비연속 센서 표본 연결을 하지 않았습니다.'
    note(fig,footer);fig.tight_layout(rect=(0,.055,1,1))
    add_chart(captions,'hourly_72h',finish(fig,'hourly_72h'),'첫72시간 확대',footer,
              {'hours':72,'first':zoom.period_start.iloc[0],'last':zoom.period_start.iloc[-1],'statistics':stats(zoom.energy_kwh)},
              [TARGET,DATA/'hourly_dataset.csv'],['구간은 첫72시간으로 사전 고정한 예시이며 전 기간 대표 주기를 증명하지 않음'])

def temporal_pattern(y,captions):
    f=pd.DataFrame({'y':y,'weekday':y.index.dayofweek,'hour':y.index.hour})
    f.index.name=None
    grouped=f.groupby(['weekday','hour']).y.agg(['mean','count','std'])
    means=grouped['mean'].unstack().reindex(index=range(7),columns=range(24));counts=grouped['count'].unstack().reindex(index=range(7),columns=range(24))
    delta=(means/float(y.mean())-1)*100;vmax=float(abs(delta.to_numpy()).max())
    fig,axes=plt.subplots(2,1,figsize=(13,6.8),gridspec_kw={'height_ratios':[2,1.35]})
    im=axes[0].imshow(delta,aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax);fig.colorbar(im,ax=axes[0],pad=.012,label='전체 관측 평균 대비 상대편차 (%)')
    axes[1].imshow(counts,aspect='auto',cmap='Blues',vmin=0,vmax=counts.to_numpy().max())
    for row in range(7):
        for col in range(24):axes[1].text(col,row,str(int(counts.iloc[row,col])),ha='center',va='center',fontsize=8,color='white')
    for ax in axes:
        ax.set_yticks(range(7),['월','화','수','목','금','토','일']);ax.set_xticks(range(24),[str(h) for h in range(24)]);ax.set_ylabel('요일')
    axes[0].set_title(f'시간대×요일 관측 평균 | 상대편차 범위 {delta.to_numpy().min():.3f}%~{delta.to_numpy().max():.3f}% (색 축 확대)')
    axes[1].set_title('각 평균을 계산한 완전 시간 관측 수 n');axes[1].set_xlabel('시간대 (0~23시)')
    footer='셀 n은 서로 다른 날짜의 시간 집계 수입니다. 작은 탐색적 평균 차이를 안정된 생산 주기·절감 가능성·예측력으로 해석하지 않습니다.'
    note(fig,footer);fig.tight_layout(rect=(0,.065,1,1))
    rows=[{'weekday':i,'hour':j,'n':int(counts.loc[i,j]),'mean_kwh':float(means.loc[i,j]),'relative_deviation_percent':float(delta.loc[i,j]),
           'std_kwh':float(grouped.loc[(i,j),'std'])} for i in range(7) for j in range(24)]
    write(OUT/'weekday_hour_cells.json',{'cells':rows,'global_mean_kwh':float(y.mean()),'uses_training':False})
    add_chart(captions,'weekday_hour_pattern',finish(fig,'weekday_hour_pattern'),'요일·시간대 평균과 셀 관측 수',footer,
       {'cell_count':168,'cell_observation_n_min':int(counts.min().min()),'cell_observation_n_max':int(counts.max().max()),
        'mean_cell_observation_n':float(counts.to_numpy().mean()),'relative_min_percent':float(delta.to_numpy().min()),
        'relative_max_percent':float(delta.to_numpy().max()),'color_scale_absolute_limit_percent':vmax,'cell_data':str(OUT/'weekday_hour_cells.json')},
       [TARGET],['학습을 통한 미래 패턴이 아니라 전 관측 기간의 기술통계','요일·시간 셀 평균 차이에 대한 유의성 검정 미실시'])

def module_figures(modules,equipment,captions):
    names=sorted(modules['module(equipment)'].unique(),key=lambda s:int(s.split('(')[0]))
    vals=[modules.loc[modules['module(equipment)']==name,'energy_kwh'].to_numpy(float) for name in names]
    means=np.array([v.mean() for v in vals]);totals=np.array([v.sum() for v in vals]);n=[len(v) for v in vals]
    assert n==[3600]*13
    fig,ax=plt.subplots(figsize=(12.5,7.7));b=ax.boxplot(vals,orientation='horizontal',tick_labels=names,showfliers=False,widths=.65,patch_artist=True)
    for box in b['boxes']:box.set_facecolor('#d8e8ec');box.set_edgecolor(TEAL)
    for line in b['medians']:line.set_color(TEAL);line.set_linewidth(1.7)
    ax.set_title('13개 모듈의 완전 시간 에너지 분포 | 각 n=3,600')
    ax.set_xlabel('1시간 관측 에너지 (잠정 kWh)');ax.grid(axis='x',alpha=.2);ax.xaxis.set_major_formatter(FORMAT)
    footer='상자=25~75분위, 선=중앙값, 수염=1.5 IQR 범위. 개별 이상점은 생략했습니다. 계측 계층이 미확정되어 공장 총량으로 단정하지 않습니다.'
    note(fig,footer);fig.tight_layout(rect=(0,.055,1,1))
    module_stats=[{'module':name,'count_complete_hours':len(v),'total_energy_kwh':float(v.sum()),**stats(v),
                   'q25':float(np.quantile(v,.25)),'median':float(np.median(v)),'q75':float(np.quantile(v,.75))} for name,v in zip(names,vals)]
    add_chart(captions,'module_energy_distribution',finish(fig,'module_energy_distribution'),'모듈 시간 에너지 분포',footer,
              {'module_count':13,'complete_rows':len(modules),'modules':module_stats},[MODULE],['분포의 비슷함은 센서 동일성·설비 효율의 증명이 아님','고장 라벨을 사용하지 않은 기술통계'])
    fig,axes=plt.subplots(1,2,figsize=(14,7.8),sharey=True);positions=np.arange(13)
    axes[0].barh(positions,totals/1e6,color='#a1c5d5');axes[0].set_yticks(positions,names);axes[0].set_xlabel('완전3,600시간 누적 에너지 (백만 kWh)');axes[0].set_xlim(0,totals.max()/1e6*1.15)
    for i,total in enumerate(totals):axes[0].text(total/1e6+.05,i,f'{total/1e6:.4f}',va='center',fontsize=9)
    axes[1].scatter(means,positions,s=55,color=TEAL);axes[1].axvline(float(means.mean()),color=GRAY,ls='--',lw=.9)
    axes[1].set_xlabel('평균1시간 에너지 (kWh, 축 확대)');axes[1].set_xlim(means.min()-.6,means.max()+.6);axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:,.1f}'))
    for ax in axes:ax.grid(axis='x',alpha=.2)
    fig.suptitle('모듈 관측 에너지 비교 | 계층·중복 미확정',fontsize=15,fontweight='bold')
    spread=float((means.max()-means.min())/means.mean()*100)
    footer=f'평균 최고-최저 차이는 {means.max()-means.min():.3f} kWh ({spread:.4f}%)입니다. 이를 실제 설비별 생산성·전력 절감 우선순위로 해석하지 않습니다.'
    note(fig,footer);fig.tight_layout(rect=(0,.06,1,.95))
    rawmean=equipment.set_index('equipment').activePower_mean.reindex(names).to_numpy(float)
    add_chart(captions,'module_energy_comparison',finish(fig,'module_energy_comparison'),'모듈 누적·평균 에너지 비교',footer,
              {'module_means_kwh':dict(zip(names,means)),'module_totals_kwh':dict(zip(names,totals)),
               'mean_high_low_range_kwh':float(means.max()-means.min()),'relative_range_percent':spread,
               'total_complete_module_sum_kwh':float(totals.sum()),'raw_summary_power_mean_kw':dict(zip(names,rawmean)),
               'raw_summary_endpoint_note':'equipment_summary includes one additional5s observation/module at April30 00; chart uses complete3600h only'},
              [MODULE,EQUIPMENT],['누적합에 원본 마지막 부분 시간13레코드를 포함하지 않음','모듈 합은 계층 미확인으로 중복 가능'])

def resolution_distribution(tables,captions):
    order=list(NAMES);values=[d.loc[d.complete,'rate_kwh_per_hour'].to_numpy(float) for d in tables.values()]
    relative=[(v/v.mean()-1)*100 for v in values];cv=[v.std(ddof=1)/v.mean()*100 for v in values]
    fig,axes=plt.subplots(1,2,figsize=(13,5.8));b=axes[0].boxplot(relative,tick_labels=[f'{NAMES[key]}\nn={len(v):,}' for key,v in zip(order,values)],showfliers=False,patch_artist=True)
    for box in b['boxes']:box.set_facecolor('#d8e8ec')
    axes[0].axhline(0,color=GRAY,lw=.8);axes[0].set_ylabel('각 해상도 평균 대비 편차 (%)');axes[0].set_title('완전 기간 평균 에너지율 분포')
    bars=axes[1].bar([NAMES[key] for key in order],cv,color=[BLUE,TEAL,'#77a7a1','#bacdd1'])
    for bar,value in zip(bars,cv):axes[1].text(bar.get_x()+bar.get_width()/2,value+.006,f'{value:.4f}%',ha='center',fontsize=10)
    axes[1].set_ylim(0,max(cv)*1.2);axes[1].set_ylabel('CV = 표준편차 / 평균 (%)');axes[1].set_title('집계 평균의 변동계수')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    footer='집계가 커지면 평균의 변동이 작아집니다. 예측 성능 향상이 자동으로 따라오지는 않으며 월의 완전 표본은4개뿐입니다.'
    note(fig,footer);fig.tight_layout(rect=(0,.06,1,1))
    evidence={key:{'complete_period_n':len(v),'std_ddof1_kwh_per_hour':float(v.std(ddof=1)),
              'mean_unweighted_period_rate':float(v.mean()),'cv_percent':float(c)} for key,v,c in zip(order,values,cv)}
    add_chart(captions,'resolution_variability',finish(fig,'resolution_variability'),'리샘플링과 평균 변동 감소',footer,evidence,
       [DATA/f'{key}_dataset.csv' for key in order],['부분 주·부분 월은 통계에서 제외됨','해상도별 관측 포함 구간과 표본 수가 달라 동일한 독립 표본 실험이 아님','집계가 원시 센서 이상을 제거하거나 미래 변동을 학습했다는 의미가 아님'])

def sensor_descriptive(sample,meta,captions):
    s=pd.DataFrame({'전력':sample.activePower,'전류합':sample[['currentR','currentS','currentT']].sum(axis=1),
                    '역률최솟값':sample[['powerFactorR','powerFactorS','powerFactorT']].min(axis=1),
                    '전압최솟값':sample[['voltageR','voltageS','voltageT']].min(axis=1),
                    '무효전력':sample.reactivePowerLagging,
                    '평균역률':sample[['powerFactorR','powerFactorS','powerFactorT']].mean(axis=1)})
    assert np.isfinite(s.to_numpy()).all();corr=s.corr(method='pearson')
    fig,axes=plt.subplots(2,3,figsize=(15,8.8));flat=axes.flat
    items=[('전력','가이드 activePower (kW)',None),('전류합','R+S+T 전류 (원본 A)',None),
           ('역률최솟값','상 역률 최솟값 (원본 %)',80),('전압최솟값','상 전압 최솟값 (원본 V)',205)]
    for ax,(col,xlabel,threshold) in zip(flat,items):
        ax.hist(s[col],bins=70,color='#a5c8d7',edgecolor='white',linewidth=.25)
        if threshold is not None:
            ax.set_yscale('log');ax.axvline(threshold,color=ORANGE,ls='--',lw=1,label=f'탐색 기준 {threshold}')
            ax.legend(fontsize=8);ax.set_ylabel('표본 관측 수 (log)')
        else:ax.set_ylabel('표본 관측 수')
        ax.set_xlabel(xlabel);ax.set_title(col+' 분포');ax.grid(alpha=.17)
    scatter=axes[1,1];chosen=np.linspace(0,len(s)-1,3000,dtype=int)
    scatter.scatter(s['전류합'].iloc[chosen],s['전력'].iloc[chosen],s=5,alpha=.25,color=TEAL,rasterized=True)
    scatter.set_xlabel('동시 관측 전류합 (A)');scatter.set_ylabel('동시 관측 전력 (가이드 kW)')
    coefficient=float(corr.loc['전력','전류합']);scatter.set_title(f'표본 동시 상관 r={coefficient:.5f}');scatter.grid(alpha=.17)
    im=axes[1,2].imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');fig.colorbar(im,ax=axes[1,2],fraction=.045,pad=.02)
    axes[1,2].set_xticks(range(6),corr.columns,rotation=35,ha='right',fontsize=8);axes[1,2].set_yticks(range(6),corr.index,fontsize=8);axes[1,2].set_title('Pearson 동시 관측 상관')
    fig.suptitle('5초 원본 센서의 균등 층화 표본 기술통계 | n=187,200',fontsize=15,fontweight='bold')
    footer='모듈×완전시간마다 무작위4개를 추출한 표본입니다. 비연속 표본을 선으로 연결하지 않았고, 동시 상관을 미래 예측·고장 인과성으로 해석하지 않습니다.'
    note(fig,footer);fig.tight_layout(rect=(0,.075,1,.96))
    feature_stats={col:stats(s[col]) for col in s}
    source_details={'rows':len(sample),'seed':meta['seed'],'sampling':meta['sampling'],
                    'complete_module_hour_strata':46800,'per_stratum_unique_slots':4,
                    'per_full_hour_inclusion_probability':meta['inclusion_probability_per_full_hour'],
                    'first':sample.dt.min(),'last':sample.dt.max(),'candidate_oversampling':False,
                    'sample_min_voltage_lt205':int((s['전압최솟값']<205).sum()),'sample_min_pf_lt80':int((s['역률최솟값']<80).sum()),
                    'correlation_power_current_sum':coefficient,'sensor_statistics':feature_stats,
                    'correlations':corr.to_dict(),'scatter_plot_points':len(chosen)}
    write(OUT/'sensor_descriptive_statistics.json',source_details)
    add_chart(captions,'sensor_distribution_correlation',finish(fig,'sensor_distribution_correlation'),'균등 센서 표본의 분포·동시 상관',footer,
       source_details,[SAMPLE,SAMPLE_META],['표본 비율·탐색 임계값은 실제 고장률/고장 정답이 아님','현재 전류와 현재 전력의 상관은 미래 전력 예측 특성이 아님',
       '역률·전압 기준은 기존 전기 품질 탐색 기준이며 확정 고장 판단이 아님','히스토그램·상관은187200 표본, 산점도만3000개를 표시'])

def main():
    y,tables,modules,equipment,sample,meta,checks=load_sources();captions={}
    period_charts(y,tables,captions);temporal_pattern(y,captions);module_figures(modules,equipment,captions)
    resolution_distribution(tables,captions);sensor_descriptive(sample,meta,captions)
    assert len(captions)==10
    write(OUT/'chart_caption.json',{'charts':captions,'figure_count':10,'image_formats':['png','svg'],
          'language':'Korean','font':'Malgun Gothic','analysis_role':'Descriptive EDA; no forecast learning/training',
          'May_overlay':'Not used; no May-label file opened'})
    sources=[TARGET,MODULE,EQUIPMENT,SAMPLE,SAMPLE_META]+[DATA/f'{name}_dataset.csv' for name in NAMES]
    monthly=tables['monthly'];jan=monthly[monthly.period_start==pd.Timestamp('2025-01-01')].iloc[0];feb=monthly[monthly.period_start==pd.Timestamp('2025-02-01')].iloc[0]
    summary={'status':'complete','script':str(Path(__file__).resolve()),'figure_count':10,'caption_file':str(OUT/'chart_caption.json'),
      'figures':{key:value['paths'] for key,value in captions.items()},
      'sources':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sources],
      'source_first_hour':str(y.index[0]),'source_last_hour':str(y.index[-1]),'source_last_hour_available_at':str(y.index[-1]+pd.Timedelta(hours=1)),
      'complete_hour_count':3600,'complete_day_count':150,'complete_module_hour_count':46800,'module_count':13,
      'total_complete_hour_module_sum_energy_kwh':float(y.sum()),'hourly_statistics':stats(y),
      'period_summaries':{key:captions[f'{key}_energy_rate']['evidence'] for key in NAMES},
      'monthly_calendar_effect':{'january_energy_kwh':jan.energy_kwh,'february_energy_kwh':feb.energy_kwh,
        'february_vs_january_energy_change_percent':float((feb.energy_kwh/jan.energy_kwh-1)*100),
        'february_vs_january_mean_rate_change_percent':float((feb.rate_kwh_per_hour/jan.rate_kwh_per_hour-1)*100),
        'february_vs_january_duration_ratio':float(feb.observed_hours/jan.observed_hours),
        'interpretation':'Observed total energy difference is predominantly28-vs31day duration; rate difference is small; not an efficiency improvement claim'},
      'weekday_hour_pattern':captions['weekday_hour_pattern']['evidence'],
      'sensor_sample':{'rows':len(sample),'method':meta['sampling'],'seed':meta['seed'],'no_candidate_oversampling':True},
      'verification':checks,'original_files_modified':False,'May_labels_read':False,'model_training_performed':False,
      'limits':['Guide unit kWh assumption remains provisional','13-module sum has unresolved meter hierarchy and potential overlap',
        'Recorded localtime is used; Unix-key timezone inconsistency was audited in prior work',
        'Rate energy/observed-hour is average load, not a rated-capacity utilization ratio',
        'Partial weeks/month are shown explicitly; excluded from complete-period CV statistics',
        'Aggregation suppresses variability but is not evidence of forecast skill','Sample histograms do not establish full-data fault ground truth']}
    write(OUT/'results.json',summary)
    print(json.dumps({'figures':len(captions),'hourly_cv_percent':summary['hourly_statistics']['cv_percent'],
                       'weekly_complete':int(tables['weekly'].complete.sum()),'monthly_complete':int(tables['monthly'].complete.sum()),
                       'energy_kwh':float(y.sum()),'May_opened':False},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
