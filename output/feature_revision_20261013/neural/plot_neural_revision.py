"""Static forecast/loss diagnostics for completed neural revision artifacts only."""
from pathlib import Path
import argparse,json,os,sys
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'output'/'analysis'/'.plot_deps'))
# Read bundled plotting dependencies without modifying their installation.
os.environ['MPLCONFIGDIR']=str(OUT/'tmp'/'mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,dates as mdates
font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'figure.facecolor':'#f2f6fa',
                     'axes.spines.top':False,'axes.spines.right':False,'font.size':10,'svg.fonttype':'none'})

def save(fig,dest,name):
    dest.mkdir(parents=True,exist_ok=True);fig.savefig(dest/f'{name}.png',dpi=135,bbox_inches='tight');fig.savefig(dest/f'{name}.svg',bbox_inches='tight');plt.close(fig)

def plot(path):
    r=json.loads(path.read_text(encoding='utf-8'));dest=path.parent/'figures'
    d=pd.read_csv(r['prediction_csv']['test'],parse_dates=['id','origin'],float_precision='round_trip')
    names={'itransformer':'다변량 iTransformer','patchtst':'다채널 PatchTST'};name=names[r['family']]
    target='aggregate 출력 채널' if r['family']=='itransformer' else '13설비 출력 합계'
    fig,axs=plt.subplots(2,1,figsize=(14,8),sharex=True,layout='constrained')
    axs[0].plot(d.id,d.actual_kwh,lw=.8,color='#4b657f',alpha=.8,label='실측 모듈 합계')
    axs[0].plot(d.id,d.predicted_kwh,lw=1.1,color='#00828b',label='선택된 모델 예측')
    axs[0].set_ylabel('에너지 (provisional kWh)\n세로축 확대');axs[0].legend(ncol=2)
    residual=d.predicted_kwh-d.actual_kwh;axs[1].plot(d.id,residual,lw=.75,color='#bc6230')
    axs[1].axhline(0,color='#506880',ls='--');axs[1].set_ylabel('예측 - 실측 (kWh)');axs[1].grid(alpha=.15)
    axs[1].xaxis.set_major_locator(mdates.AutoDateLocator());axs[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[1].xaxis.get_major_locator()))
    axs[1].set_xlim(d.id.min(),d.id.max())
    axs[1].set_xlabel('localtime · 같은 데이터에서 시간 순서로 검증 · test 점수는 선택에 사용하지 않음')
    met=r['test_metrics'];fig.suptitle(f'{name} · {r["scenario"]} · 입력 {r["input_feature_count"]}채널 · {target}',fontsize=17,fontweight='bold')
    fig.supxlabel(f'MAE {met["mae_kwh"]:.3f}kWh · R² {met["r2"]:.4f} · 현재 계측 계층 미확인 · 미래 센서 입력 없음',fontsize=10)
    save(fig,dest,'test_forecast')
    chosen=next(c for c in r['candidate_results'] if c['candidate']==r['selected_candidate'])
    first=chosen['fold_fits'][0]['inner'];logs=pd.DataFrame(first['logs'])
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    axs[0].plot(logs.epoch,logs.data_loss_standardized,color='#00828b',marker='o',ms=3)
    axs[0].set_xlabel('학습 epoch');axs[0].set_ylabel(f'학습 {logs.loss.iloc[0]} · 학습 prefix 표준화 단위');axs[0].set_title('학습 오차 변화')
    axs[1].plot(logs.epoch,logs.inner_tail_mae_kwh,color='#bc6230',marker='o',ms=3)
    axs[1].axvline(first['best_epoch'],color='#4b657f',ls='--',label=f'선택 epoch {first["best_epoch"]}')
    axs[1].set_xlabel('학습 epoch');axs[1].set_ylabel('inner 시간 holdout MAE (kWh)');axs[1].set_title('학습 prefix 안의 후속 시간 검증');axs[1].legend()
    for ax in axs:ax.grid(alpha=.15)
    fig.suptitle(f'{name} · 선택 구성의 첫 outer fold · 학습/검증 오차의 단위를 분리',fontsize=15,fontweight='bold')
    fig.supxlabel('학습 손실 감소가 검증 개선을 보장하지 않음 · outer fold/April/May 평가 타깃으로 epoch를 고르지 않음',fontsize=10)
    save(fig,dest,'training_validation_loss')
    r['plots']={'test_forecast_png':str(dest/'test_forecast.png'),'test_forecast_svg':str(dest/'test_forecast.svg'),
                'training_validation_png':str(dest/'training_validation_loss.png'),'training_validation_svg':str(dest/'training_validation_loss.svg')}
    path.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(f'PLOTTED {r["family"]} {r["scenario"]}',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario',choices=['dayahead24','long745']);args=p.parse_args()
    scenarios=[args.scenario] if args.scenario else ['dayahead24','long745']
    for scenario in scenarios:
        for family in ['itransformer','patchtst']:
            path=OUT/scenario/family/'results.json'
            if path.exists():plot(path)

if __name__=='__main__':main()
