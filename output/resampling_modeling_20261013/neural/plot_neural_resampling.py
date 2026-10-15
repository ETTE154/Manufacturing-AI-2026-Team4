"""Static native-resolution diagnostics; no May-label access or model changes."""
from pathlib import Path
import sys,os
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'output/analysis/.plot_deps'))
os.environ['MPLCONFIGDIR']=str(OUT/'tmp/mpl')
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties,fontManager
from matplotlib.ticker import FuncFormatter
fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f7f9fc',
                     'axes.facecolor':'white','svg.fonttype':'none','axes.titleweight':'bold'})
FIG=OUT/'figures';FIG.mkdir(parents=True,exist_ok=True)
COLORS={'patchtst':'#238b8b','itransformer':'#c86925','chronos2':'#6753a4'}
NAME={'patchtst':'PatchTST','itransformer':'iTransformer','chronos2':'Chronos-2'}

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def save(fig,stem):
    paths={}
    for extension in ['png','svg']:
        p=FIG/f'{stem}.{extension}';fig.savefig(p,dpi=160,bbox_inches='tight',facecolor=fig.get_facecolor());paths[extension]=str(p)
    plt.close(fig);return paths
def date_axis(ax,start,end):
    ax.set_xlim(start,end);ax.xaxis.set_major_locator(mdates.DayLocator(interval=5));ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
def main():
    allresults=read(OUT/'results.json');rows=[];figures={}
    for resolution,data in allresults['scenarios'].items():
        for family,result in data['models'].items():
            label=('H 시간' if resolution=='hourly' else 'D 일')+' · '+NAME[family]
            rows.append({'label':label,'family':family,'resolution':resolution,'validation':result['validation']['mean_origin_late505_mae_kwh'],
                         'test':result['internal_test']['late505']['mae_kwh'],'result':result})
            f=pd.read_csv(result['fits']['internal_test']['prediction_csv'],float_precision='round_trip');f['id']=pd.to_datetime(f.id)
            fig,axes=plt.subplots(3,1,figsize=(14,8.4),gridspec_kw={'height_ratios':[2,1,1]})
            axes[0].plot(f.id,f.actual_kwh,color='#637180',lw=.65,alpha=.75,label='관측 시간 에너지')
            axes[0].plot(f.id,f.predicted_kwh,color=COLORS[family],lw=1.25,label='고정 선택 모델 예측')
            axes[0].set_title(label+' | 3월 29일 원점, 745시간 경로 (축 확대)')
            axes[0].set_ylabel('잠정 kWh / 시간');axes[0].legend(loc='upper right',ncol=2)
            axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:,.0f}'))
            axes[1].plot(f.id,f.predicted_kwh-f.actual_kwh,color=COLORS[family],lw=.6)
            axes[1].axhline(0,color='#333d4a',lw=.8);axes[1].set_ylabel('예측 - 관측\n(kWh)')
            for ax in axes[:2]:date_axis(ax,f.id.iloc[0],f.id.iloc[-1]);ax.grid(alpha=.18)
            axes[1].axvspan(f.id.iloc[240],f.id.iloc[-1],color='#e6edf7',alpha=.5,zorder=0)
            early=f.iloc[:72];axes[2].plot(early.id,early.actual_kwh,color='#637180',lw=.8,label='관측')
            axes[2].step(early.id,early.predicted_kwh,color=COLORS[family],where='post',lw=1.35,label='예측 시간 매핑')
            axes[2].set_xlim(early.id.iloc[0],early.id.iloc[-1]);axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H시'))
            axes[2].set_ylabel('잠정 kWh');axes[2].set_title('처음 72시간 확대: '+('일 평균 rate를 시간마다 균등 배분' if resolution=='daily' else '시간 단위 직접 745출력'))
            axes[2].yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:,.0f}'));axes[2].grid(alpha=.18)
            metric=result['internal_test']['late505'];note=f'내부 시험 Late505 MAE {metric["mae_kwh"]:.3f} kWh · R² {metric["r2"]:.4f} · 시험 점수로 모델을 재선택하지 않음'
            if family=='chronos2' and result['alpha0_baseline_equivalent']:note+=' · alpha=0: 과거 평균과 동일'
            fig.text(.5,.012,note,ha='center',fontsize=10,color='#334155');fig.tight_layout(rect=(0,.035,1,1))
            figures[f'{resolution}_{family}']=save(fig,f'{resolution}_{family}_internal_path')
    fig,ax=plt.subplots(figsize=(13,5.2));xs=np.arange(len(rows));width=.35
    ax.bar(xs-width/2,[r['validation'] for r in rows],width,color='#334d74',label='3개 검증 원점 평균')
    ax.bar(xs+width/2,[r['test'] for r in rows],width,color='#a8bbd6',label='3/29 내부 시험')
    ax.set_xticks(xs,[r['label'] for r in rows]);ax.set_ylabel('Late505 MAE (잠정 kWh)');ax.set_title('H/D 모델 비교 | 검증 Late505만으로 설정 선택')
    ax.set_ylim(0,max(r['test'] for r in rows)*1.16);ax.legend(ncol=2);ax.grid(axis='y',alpha=.2)
    for i,r in enumerate(rows):
        for offset,key in [(-width/2,'validation'),(width/2,'test')]:ax.text(i+offset,r[key]+.5,f'{r[key]:.3f}',ha='center',fontsize=8)
    fig.text(.5,.01,'검증 1,515개 오차는 고유 745개 정답 시간에서 나온 중첩 경로입니다. 작은 차이는 유의성이나 미래 우위를 뜻하지 않습니다.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));figures['resolution_comparison']=save(fig,'resolution_comparison')
    fig,axes=plt.subplots(2,2,figsize=(13,7.4));positions=[r for r in rows if r['family']!='chronos2']
    for ax,r in zip(axes.flat,positions):
        res=r['result'];lock=read(res['selection_lock']);candidate=lock['candidate_results'][lock['selected_candidate']]
        inner=candidate['folds'][0]['inner_early_stopping'];logs=inner['logs'];es=[v['epoch'] for v in logs]
        ax.plot(es,[v['target_data_loss_standardized'] for v in logs],color='#334d74',marker='o',ms=3,label='학습 손실(표준화)')
        ax.set_ylabel('표준화 target 손실');ax.set_xlabel('epoch');ax.set_title(r['label']+' · 첫 원점 내부검증')
        other=ax.twinx();other.spines['right'].set_visible(True)
        other.plot(es,[v['inner_late505_mae_kwh'] for v in logs],color='#c86925',marker='s',ms=3,label='내부 Late505 MAE')
        other.set_ylabel('내부검증 MAE(kWh)');ax.axvline(inner['best_epoch'],color='#697386',ls='--',lw=.9)
        ax.grid(alpha=.18)
        ax.text(.02,.92,f'선택 epoch {inner["best_epoch"]}'+(' / 상한 도달' if inner['best_at_epoch_cap'] else ''),transform=ax.transAxes,fontsize=9)
    fig.suptitle('학습 손실 감소와 내부검증 변화 | 외부 시험을 epoch 선택에 사용하지 않음',fontsize=13)
    fig.text(.5,.015,'일 단위 첫 내부 학습은 44일, 문맥7일+출력32일에서 겹치는 창6개입니다. 신뢰할 만한 계절·장기 관계를 학습하기에 매우 작습니다.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.95));figures['nested_training']=save(fig,'nested_training')
    (OUT/'plot_manifest.json').write_text(json.dumps({'figures':figures,'May_labels_used':False,'source':str(OUT/'results.json')},ensure_ascii=False,indent=2),encoding='utf8')
    print(f'{len(figures)} PNG/SVG diagnostic figures saved')
if __name__=='__main__':main()
