"""Read a bounded original prefix for raw cadence illustration, no imputation."""
from pathlib import Path
import sys,json,os
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(OUT/'tmp/matplotlib')
sys.path.insert(0,str(ROOT/'output/analysis/.plot_deps'))
import numpy as np
import pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
fp=FontProperties(fname='C:/Windows/Fonts/malgun.ttf');plt.rcParams['font.family']=fp.get_name();plt.rcParams['axes.unicode_minus']=False;plt.rcParams['axes.formatter.useoffset']=False

def main():
    out=OUT/'raw_views';out.mkdir(parents=True,exist_ok=True)
    f=pd.read_csv(ROOT/'traing-dataset/rtu_data_full.csv',nrows=17280,float_precision='round_trip');parser_dtypes={k:str(v)for k,v in f.dtypes.items()}
    f['localtime']=pd.to_datetime(f.localtime.astype(str),format='%Y%m%d%H%M%S')
    assert f['module(equipment)'].nunique()==1 and f['module(equipment)'].iloc[0]=='1(PM-3)'
    assert f.localtime.iloc[0]==pd.Timestamp('2024-12-01') and f.localtime.iloc[-1]==pd.Timestamp('2024-12-01 23:59:55')
    assert (f.localtime.diff().dropna()==pd.Timedelta(seconds=5)).all()
    fig,axs=plt.subplots(2,1,figsize=(11.4,6.3),layout='constrained')
    axs[0].plot(f.localtime,f.activePower,lw=.3,color='#0f766e');axs[0].set_title('PM-3: 첫 1일의 원본 5초 관측 17,280개')
    z=f.iloc[:25];axs[1].plot(z.localtime,z.activePower,'o-',ms=4,lw=1,color='#0f766e');axs[1].set_title('처음 120초 확대: 각 점 사이 간격 5초')
    for a in axs:a.set_ylabel('activePower (가이드상 kW)');a.grid(alpha=.2);a.tick_params(axis='x',labelsize=8)
    fig.suptitle('원자료 예시: PM-3의 2024-12-01 / 전체 데이터 분포의 대표 표본은 아님',fontsize=12)
    fig.savefig(out/'raw_5second_example.png',dpi=150);fig.savefig(out/'raw_5second_example.svg');plt.close(fig)
    # Clearly separate supplied raw sensors from future evaluation labels.
    fig,ax=plt.subplots(figsize=(11.4,3.4),layout='constrained')
    for start,end,y,label,color in [('2024-12-01','2025-04-30',2,'Full 원본 센서: 13모듈 / 5초','#0f766e'),('2024-12-01','2025-04-30',1,'완료된 시간 집계: 3,600시간','#2563eb'),('2025-05-10','2025-05-31',0,'May 정답: 505시간 / 센서 없음','#d97706')]:
        ax.plot(pd.to_datetime([start,end]),[y,y],lw=14,solid_capstyle='butt',color=color);ax.text(pd.Timestamp(start),y+.2,label,fontsize=10)
    ax.axvspan(pd.Timestamp('2025-04-30'),pd.Timestamp('2025-05-10'),alpha=.14,color='#94a3b8');ax.text(pd.Timestamp('2025-05-05'),1.2,'중간 실제 센서\n자료 없음',ha='center',fontsize=9)
    ax.set_ylim(-.4,2.8);ax.set_yticks([]);ax.grid(axis='x',alpha=.2);ax.set_title('제공 파일의 시간 범위: 표시 시각은 기록된 localtime, 실제 시간대는 미확정',fontsize=11)
    ax.set_xlim(pd.Timestamp('2024-11-28'),pd.Timestamp('2025-06-03'));ax.tick_params(axis='x',labelsize=9)
    fig.savefig(out/'dataset_timeline.png',dpi=150);fig.savefig(out/'dataset_timeline.svg');plt.close(fig)
    meta={'rows_read':len(f),'scope':'Original firstPM-3 day, bounded prefix only, not full-population sensor inference',
      'first':str(f.localtime.iloc[0]),'last':str(f.localtime.iloc[-1]),'cadence_seconds':5,'unique_modules':1,
      'inferred_parser_dtypes_from_prefix':parser_dtypes,'localtime_parser':'String conversion followed by explicitformat %Y%m%d%H%M%S',
      'figures':{'raw':'raw_views/raw_5second_example.png','timeline':'raw_views/dataset_timeline.png'},
      'original_modified':False,'no_imputation':True,'no_anomaly_removal':True}
    (out/'results.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(meta,ensure_ascii=True))
if __name__=='__main__':main()
