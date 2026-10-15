"""Build the offline Korean modeling report from frozen experiment artifacts.

This script reads results; it never trains a model or changes selection locks.
All reported measurements are loaded from JSON or calculated from prediction CSV.
"""
from pathlib import Path
import os,sys,json,base64,html,argparse
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'output'/'analysis'/'.plot_deps'))
os.environ['MPLCONFIGDIR']=str(OUT/'tmp'/'report_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10,
                     'figure.facecolor':'#f7f9fc','axes.facecolor':'white','axes.spines.top':False,
                     'axes.spines.right':False,'axes.titleweight':'bold','text.color':'#172a43',
                     'axes.labelcolor':'#475569','xtick.color':'#475569','ytick.color':'#475569',
                     'axes.edgecolor':'#c7d2df','svg.fonttype':'none'})
FIG=OUT/'report_figures'
FIG.mkdir(exist_ok=True)
PALETTE=['#007f86','#dc6b26','#3467b7','#8b5ca6','#aa4057','#769442','#9b7443','#687c8f','#2672a0','#df8996']
HORIZONS=['H96','H192','H336','H720','P745']
NAMES={'arima':'ARIMA','xgboost':'XGBoost','patchtst':'PatchTST compact','itransformer':'iTransformer compact',
       'chronos2':'Chronos-2 120M zero-shot','historical_mean':'과거 전체 평균','historical_median':'과거 전체 중앙값',
       'last_7_days_mean':'최근 7일 평균','hour_of_day_mean':'시간대 평균','weekday_hour_mean':'요일 × 시간 평균',
       'seasonal_naive_24h_fixed':'마지막 하루 반복','seasonal_naive_168h_fixed':'마지막 주 반복'}

def load(path,required=True):
    p=Path(path)
    if not p.exists():
        if required:raise FileNotFoundError(p)
        return {}
    return json.loads(p.read_text(encoding='utf-8'))

def esc(value):return html.escape(str(value))
def fmt(value,digits=3):return '—' if value is None or not np.isfinite(float(value)) else f'{float(value):,.{digits}f}'
def title(model,meta):return meta.get('display_name') or meta.get('name') or NAMES.get(model,model)
def metric(obj,key):
    keys={'mae_kwh':['mae_kwh','mae'],'rmse_kwh':['rmse_kwh','rmse'],
          'mape_percent':['mape_percent','mape'],'bias_pred_minus_actual_kwh':['bias_pred_minus_actual_kwh','bias'],
          'count':['count','rows','prediction_count']}
    for k in keys.get(key,[key]):
        if isinstance(obj,dict) and k in obj:return obj[k]
    return None

def normalize(comparison):
    """Accept model dictionaries or flat model/split/horizon rows."""
    models=comparison.get('models') or comparison.get('model_rows') or comparison.get('rows')
    if isinstance(models,dict):return models
    if isinstance(models,list):
        grouped={}
        for row in models:
            model=row.get('model') or row.get('model_id') or row.get('name')
            if not model:raise ValueError('Model row missing model ID')
            if 'split' in row:
                split=row['split'];horizon=row.get('horizon') or row.get('window') or ('P505' if split.startswith('may') else 'P745')
                grouped.setdefault(model,{})
                grouped[model].setdefault(split,{})[horizon]=row.get('metrics',row)
                for k in ['display_name','input_scope','training_type','runtime_seconds','config']:
                    if k in row:grouped[model][k]=row[k]
            else:grouped[model]=row
        return grouped
    raise ValueError('comparison_results.json must contain models dictionary or flat model rows')

def scores(meta,split,horizon='P745'):
    aliases={'internal_test':['internal_test','test'],'validation':['validation','valid'],
             'may':['may','may_holdout','may_evaluation','may_final','may_P505']}
    obj=next((meta[k] for k in aliases.get(split,[split]) if k in meta),{})
    if not isinstance(obj,dict):return {}
    if horizon in obj:return obj[horizon]
    if split=='may':
        for k in ['P505','scores','metrics']:
            if k in obj:return obj[k]
    if metric(obj,'mae_kwh') is not None:return obj
    return obj.get('scores',{})

def embed(path,alt):
    return '<img loading="lazy" src="data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode()+'" alt="'+esc(alt)+'">'

def save(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=150,bbox_inches='tight');fig.savefig(FIG/f'{name}.svg',bbox_inches='tight');plt.close(fig)
    return FIG/f'{name}.png'

def finish_axis(ax):
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:,.0f}'))

def prepare(comparison,predictions,lock):
    models=normalize(comparison)
    selected=lock.get('selected_model') or lock.get('selected_model_id') or lock.get('deployment_model')
    if isinstance(selected,dict):selected=selected.get('model') or selected.get('model_id') or selected.get('name')
    if selected is None:
        chosen=lock.get('selection') or {};selected=chosen.get('selected_model') or chosen.get('model')
    if selected not in models:raise ValueError(f'Deployment selected model missing from results: {selected}')
    columns=['model','split','id','actual_kwh','predicted_kwh']
    if any(c not in predictions for c in columns):raise ValueError('Prediction CSV missing '+str(columns))
    if predictions.duplicated(['model','split','id']).any():raise ValueError('Duplicate forecast target keys')
    ids=list(models)
    if any(k not in set(predictions.model) for k in ids):raise ValueError('Model in results missing prediction path')
    may=predictions[predictions.split.astype(str).str.lower().str.startswith('may')].copy()
    labelled=may[may.actual_kwh.notna()].copy()
    reference=labelled[labelled.model==selected].sort_values('id')
    computed={}
    for name,grp in labelled.groupby('model'):
        grp=grp.sort_values('id')
        if not grp.id.reset_index(drop=True).equals(reference.id.reset_index(drop=True)) or not np.allclose(grp.actual_kwh,reference.actual_kwh):
            raise ValueError(f'May scored target axis or actuals differ across models: {name}')
        actual=grp.actual_kwh.to_numpy();forecast=grp.predicted_kwh.to_numpy();residual=forecast-actual
        if len(grp)!=505:raise ValueError(f'May must score 505 labelled rows for {name}, got {len(grp)}')
        if not np.isfinite(forecast).all():raise ValueError('Nonfinite May predictions')
        sst=float(((actual-actual.mean())**2).sum())
        computed[name]={'count':len(grp),'mae_kwh':float(np.abs(residual).mean()),'rmse_kwh':float(np.sqrt(np.square(residual).mean())),
                        'mape_percent':float((np.abs(residual)/np.abs(actual)).mean()*100),
                        'bias_pred_minus_actual_kwh':float(residual.mean()),'r2':float(1-np.square(residual).sum()/sst)}
        claimed=scores(models[name],'may','P505')
        if metric(claimed,'mae_kwh') is not None and not np.isclose(metric(claimed,'mae_kwh'),computed[name]['mae_kwh'],rtol=1e-7,atol=1e-7):
            raise ValueError(f'May CSV/JSON MAE mismatch for {name}')
    if set(computed)!=set(models):raise ValueError('Not every model has final May scored predictions')
    return models,selected,ids,may,labelled,computed

def forecasting_charts(models,selected,ids,may,labelled,may_metrics):
    colors={k:PALETTE[i%len(PALETTE)] for i,k in enumerate(ids)}
    fig,axs=plt.subplots(1,2,figsize=(15,6),layout='constrained')
    for ax,split,heading in zip(axs,['validation','internal_test'],['선택에 사용한 시간 순서 검증','선택 후 내부 테스트']):
        for model in ids:
            vals=[metric(scores(models[model],split,h),'mae_kwh') for h in HORIZONS]
            if all(v is None for v in vals):continue
            ax.plot(range(len(HORIZONS)),[np.nan if v is None else v for v in vals],marker='o',markersize=4,
                    lw=2.4 if model==selected else 1.3,alpha=1 if model==selected else .8,
                    color=colors[model],label=title(model,models[model])+(' · 배포 후보' if model==selected else ''))
        ax.set_xticks(range(len(HORIZONS)),HORIZONS);ax.set_ylabel('MAE (잠정 모듈 합계 kWh)');ax.set_title(heading);finish_axis(ax)
        ax.set_ylim(bottom=0)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=min(4,len(ids)),fontsize=9)
    fig.suptitle('예측 구간 길이에 따른 MAE · 같은 fixed-origin 경로를 누적 평가',fontsize=17,fontweight='bold')
    hp=save(fig,'01_horizon_mae')

    grp=labelled[labelled.model==selected].sort_values('id');actual=grp.actual_kwh
    fig,axs=plt.subplots(2,1,figsize=(14,8),height_ratios=[1.4,1],layout='constrained')
    axs[0].plot(grp.id,actual,color='#8495a8',lw=.8,label='5월 실측 정답')
    for model in ids:
        z=labelled[labelled.model==model].sort_values('id')
        axs[0].plot(z.id,z.predicted_kwh,color=colors[model],lw=2.1 if model==selected else .9,
                    alpha=1 if model==selected else .62,label=title(model,models[model]))
    axs[0].set_title('5월 P505 평가 · 예측 곡선이 비슷하면 서로 겹쳐 보임');axs[0].set_ylabel('시간당 값 (가정 kWh)');finish_axis(axs[0]);axs[0].legend(ncol=3,fontsize=8)
    axs[1].plot(grp.id,grp.predicted_kwh-grp.actual_kwh,color=colors[selected],lw=.9)
    axs[1].axhline(0,color='#61758e',ls='--',lw=1);axs[1].set_title(title(selected,models[selected])+' · 예측 - 실측 잔차');axs[1].set_ylabel('잔차 (kWh)');finish_axis(axs[1])
    for ax in axs:
        ax.set_xticks(pd.date_range(grp.id.min(),grp.id.max(),freq='4D'));ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    fig.suptitle('4/30 한 번의 예측에서 나온 5/10–5/31 결과 · 세로축 확대',fontsize=17,fontweight='bold')
    fig.supxlabel('원점 이후 745시간을 생성하고 정답이 있는 offset 240–744의 505시간만 평가 · 미래 실제값으로 갱신하지 않음',fontsize=9)
    mp=save(fig,'02_may_prediction_residual')

    fig,axs=plt.subplots(1,2,figsize=(15,7),layout='constrained')
    order=sorted(ids,key=lambda k:metric(scores(models[k],'validation'),'mae_kwh'))
    y=np.arange(len(order));width=.36
    axs[0].barh(y-width/2,[metric(scores(models[k],'validation'),'mae_kwh') for k in order],height=width,color='#007f86',label='검증 P745')
    axs[0].barh(y+width/2,[metric(scores(models[k],'internal_test'),'mae_kwh') for k in order],height=width,color='#3467b7',label='내부 테스트 P745')
    axs[0].set_yticks(y,[title(k,models[k])+(' ★' if k==selected else '') for k in order]);axs[0].invert_yaxis();axs[0].set_title('검증으로 선택 · 내부 테스트는 사후 확인');axs[0].set_xlabel('MAE (kWh)');axs[0].legend(fontsize=9)
    axs[1].barh(y,[may_metrics[k]['mae_kwh'] for k in order],color=[colors[k] if k==selected else '#93a4b8' for k in order],height=.65)
    axs[1].set_yticks(y,[]);axs[1].invert_yaxis();axs[1].set_title('별도 5월 P505 평가 · 순위를 배포 선택에 쓰지 않음');axs[1].set_xlabel('MAE (가정 kWh)')
    for i,k in enumerate(order):axs[1].text(may_metrics[k]['mae_kwh']+max(v['mae_kwh'] for v in may_metrics.values())*.012,i,f'{may_metrics[k]["mae_kwh"]:.3f}',va='center',fontsize=9)
    axs[1].set_xlim(0,max(v['mae_kwh'] for v in may_metrics.values())*1.2)
    for ax in axs:ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('배포 결정과 사후 평가를 분리한 모델 비교',fontsize=17,fontweight='bold')
    fig.supxlabel('별표는 validation P745로 잠금한 배포 후보 · 내부 테스트와 5월의 최소 MAE 모델로 다시 선택하지 않음',fontsize=9)
    bp=save(fig,'03_locked_comparison')
    return [hp,mp,bp],colors

def build_report(comparison,predictions,lock,protocol,research,diagnosis):
    models,selected,ids,may,labelled,may_metrics=prepare(comparison,predictions,lock)
    charts,colors=forecasting_charts(models,selected,ids,may,labelled,may_metrics)
    select_name=title(selected,models[selected]);selection_metric=metric(scores(models[selected],'validation'),'mae_kwh')
    selected_may=may_metrics[selected]
    actual=labelled[labelled.model==selected].actual_kwh;cv=float(actual.std()/actual.mean()*100)
    fallback=lock.get('baseline_fallback','baseline_mean')
    mean_validation=metric(scores(models[fallback],'validation'),'mae_kwh') if fallback in models else None
    selection_delta=float(selection_metric-mean_validation) if mean_validation is not None else None
    internal_delta=float(metric(scores(models[selected],'internal_test'),'mae_kwh')-metric(scores(models[fallback],'internal_test'),'mae_kwh')) if fallback in models else None
    best_may=min(ids,key=lambda k:may_metrics[k]['mae_kwh'])
    best_may_improvement=may_metrics[fallback]['mae_kwh']-may_metrics[best_may]['mae_kwh'] if fallback in models else None
    practical_note=''
    if selection_delta is not None:
        practical_note=f'<div class="note"><strong>검증 최소값과 실질적 우월성은 구분합니다.</strong> 배포 후보의 검증 P745 MAE는 평균 기준선보다 {fmt(selection_delta,6)} kWh 차이({fmt(selection_delta/mean_validation*100,4)}%)이고 내부 테스트 P745의 차이는 {fmt(internal_delta,6)} kWh입니다. 이 정도의 차이만으로 복잡한 모델의 안정적인 우위를 입증하기 어렵습니다. 평균 기준선을 단순한 대체 경로로 유지합니다.</div>'
    may_note=f'<p>사후 5월 P505에서 가장 작은 MAE는 {esc(title(best_may,models[best_may]))}의 {fmt(may_metrics[best_may]["mae_kwh"])} kWh입니다.'
    if best_may_improvement is not None:
        may_note+=f' 평균 기준선 대비 {fmt(best_may_improvement,4)} kWh({fmt(best_may_improvement/may_metrics[fallback]["mae_kwh"]*100,3)}%) 차이입니다.'
    may_note+=' 이 순위는 배포 선택에 사용하지 않았고 검증으로 잠금한 후보를 유지합니다.</p>'
    biases=[may_metrics[k]['bias_pred_minus_actual_kwh'] for k in ids]
    level_note=f'<p>5월 예측−실측 평균 편향은 모델에 따라 {fmt(min(biases))}–{fmt(max(biases))} kWh입니다. 대부분 학습 기간의 수준에 가까운 예측을 유지하여 소폭 낮아진 5월 평균에 양의 편향이 생겼습니다. 미래 운영 변화의 원인은 이 파일만으로 확정할 수 없습니다.</p>'
    uncertainty_rows=''
    for k in ids:
        for split,pair in models[k].get('paired_error_analysis',{}).items():
            if split not in ['internal_test','may_labelled']:continue
            interval=pair.get('conditional_24h_block_bootstrap_95_interval_kwh')
            if not interval:continue
            horizon=pair.get('horizon') or pair.get('window') or ('P745' if split=='internal_test' else 'P505')
            window=('내부 테스트 ' if split=='internal_test' else '5월 ')+str(horizon)
            delta=pair.get('delta_mae_model_minus_mean_kwh')
            uncertainty_rows+=f'<tr><td>{esc(title(k,models[k]))}</td><td>{esc(window)}</td><td>{fmt(delta,6)}</td><td>[{fmt(interval[0],6)}, {fmt(interval[1],6)}]</td></tr>'
    uncertainty=''
    if uncertainty_rows:
        uncertainty=f'<section><h2>평균 기준선과의 차이 · 경로 내부의 조건부 불확실성</h2><div class="tablewrap"><table><tr><th>모델</th><th>평가 구간</th><th>모델 MAE−평균 MAE</th><th>조건부 95% 구간 (kWh)</th></tr>{uncertainty_rows}</table></div><p>음수는 평균 기준선보다 오차가 작은 방향입니다. 24시간 블록 bootstrap은 한 fixed-origin 경로 안의 오차 의존성을 일부 보존해 계산한 조건부 구간이며, 독립된 여러 예측 원점·연도·공장에서의 일반화 불확실성을 나타내지 않습니다. 내부 비교는 P745 전체 745시간이고 5월 비교는 정답이 있는 P505 505시간입니다.</p><p>ARIMA(0,0,0)와 평균 기준선의 극히 작은 차이는 수치 계산 오차 수준이므로 성능 우위로 해석하지 않습니다.</p></section>'
    interval_note=''
    if 'chronos2' in models:
        intr=models['chronos2'].get('may',{}).get('interval',{})
        if intr:
            interval_note=f'<div class="note caution">Chronos-2의 명목 {fmt(intr["nominal_coverage"]*100,0)}% 예측 구간은 이번 505시간에서 {fmt(intr["empirical_coverage"]*100,2)}%의 실제값을 포함했고 평균 폭은 {fmt(intr["mean_width_kwh"])} kWh였습니다. 로컬 보정을 하지 않은 구간이며 명목 포함률이 현장에서도 보장되는 것은 아닙니다.</div>'
    method_ids=[k for k in ids if any(q in k.lower() for q in ['arima','xgb','patch','itrans','chronos'])]
    sources={r['id']:r for r in research.get('sources',[])}
    def citations(keys):return ' · '.join('<a href="'+esc(sources[k]['url'])+'" target="_blank" rel="noopener">'+esc(sources[k]['title'])+'</a>' for k in keys if k in sources)
    datasets=''.join('<tr><td>'+esc(r['name'])+'</td><td>'+esc(r['resolution'])+'</td><td>'+esc(r['similarity'])+'</td><td>'+esc(r['limits'])+'<br>'+citations(r['sources'])+'</td></tr>' for r in research.get('datasets',[]) if r['name'] in ['HIPE','SPARK (KIT 2026 확장)','ECL 장기 예측 benchmark','Steel Industry Energy Consumption'])
    model_rows=''
    configs=''
    for k in ids:
        meta=models[k]
        if 'itrans' in k.lower():fallback_scope='과거 13개 모듈의 시간당 에너지 · 다변량 → 합계 출력'
        else:fallback_scope='과거 모듈 합계 1개 시계열 · XGBoost는 확정 달력 추가'
        scope=meta.get('input_scope',fallback_scope)
        training=meta.get('training_type') or ('공식 120M 사전학습 가중치 · zero-shot' if 'chronos' in k.lower() else '로컬 학습' if any(w in k.lower() for w in ['patch','itrans']) else '기준선/통계·트리 학습')
        vs=scores(meta,'validation');ts=scores(meta,'internal_test');ms=may_metrics[k]
        model_rows+=f'<tr><td>{esc(title(k,meta))}'+(' <span class="pill">배포 후보</span>' if k==selected else '')+f'</td><td>{fmt(metric(vs,"mae_kwh"))}</td><td>{fmt(metric(ts,"mae_kwh"))}</td><td>{fmt(ms["mae_kwh"])}</td><td>{fmt(ms["rmse_kwh"])}</td><td>{fmt(ms["mape_percent"])}</td><td>{fmt(ms["bias_pred_minus_actual_kwh"])}</td><td>{fmt(ms["r2"])}</td></tr>'
        config=meta.get('frozen_config') or meta.get('config') or meta.get('selected_config') or {}
        configs+=f'<details><summary>{esc(title(k,meta))} · 입력/학습/설정</summary><p>{esc(scope)}</p><p>{esc(training)}</p><pre>{esc(json.dumps(config,ensure_ascii=False,indent=2))}</pre></details>'
    diag_names={'xgb_rule_distillation':'규칙 모사 XGB','xgb_rule_sensor_excluded_transfer':'규칙 센서 제외 XGB','ae_unsupervised_all_training':'전체 학습 AE','ae_weak_normal_proxy':'약한 정상 proxy AE'}
    diag_rows=''
    for k,r in diagnosis['results'].items():
        ms=r.get('test',r.get('test_weak_proxy_agreement'))
        diag_rows+=f'<tr><td>{esc(diag_names.get(k,k))}</td><td>{fmt(ms["average_precision"],4)}</td><td>{fmt(ms["auroc"],4)}</td><td>{fmt(ms["precision"],4)}</td><td>{fmt(ms["recall"],4)}</td><td>{fmt(ms["f1"],4)}</td><td>{ms["true_positive"]}/{ms["positive_rows"]}</td><td>{ms["false_positive"]}</td></tr>'
    sample=diagnosis['sample_summary'];test_sample=next(r for r in sample['splits'] if r['split']=='test')
    selected_group=labelled[labelled.model==selected].sort_values('id')
    payload={'actual':[[r.id.strftime('%Y-%m-%d %H:%M'),round(r.actual_kwh,5)] for r in selected_group.itertuples()],
             'models':{k:{'name':title(k,models[k]),'color':colors[k],'data':[round(v,5) for v in labelled[labelled.model==k].sort_values('id').predicted_kwh]} for k in ids},'selected':selected}
    locked_summary=lock.get('selection_rule') or lock.get('selection_metric') or 'Lowest validation P745 MAE; test and May labels excluded from deployment selection'
    css='''*{box-sizing:border-box}body{margin:0;background:#eef3f8;color:#172a43;font-family:"Malgun Gothic","Apple SD Gothic Neo",sans-serif;line-height:1.7}main{max-width:1180px;margin:auto;padding:34px 24px 70px}header{background:#152e47;color:#fff;padding:35px 38px;border-radius:18px;margin-bottom:24px}h1{font-size:29px;margin:7px 0 12px;letter-spacing:-.6px}header p{color:#d7e5f2}h2{font-size:22px;margin:0 0 18px}h3{font-size:17px;margin:24px 0 8px}section{background:#fff;border:1px solid #dbe4ee;border-radius:15px;padding:28px 30px;margin:22px 0}.eyebrow{font-size:12px;letter-spacing:1.5px;color:#75dacf}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{background:#f8fafc;border:1px solid #dbe4ee;border-radius:11px;padding:18px}.value{font-size:26px;font-weight:bold;line-height:1.4}.caption,.label{font-size:12px;color:#61758e}.label{margin-top:8px}.note{padding:17px 19px;background:#eff9f8;border-left:4px solid #007f86;margin:20px 0}.caution{background:#fff4e9;border-left-color:#dc6b26}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:11px 9px;border-bottom:1px solid #dbe4ee;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#f0f5fa}table.prose td,table.prose th{text-align:left;vertical-align:top}table.prose td:first-child{min-width:155px}img{width:100%;height:auto;display:block;margin:17px 0}.pill{display:inline-block;background:#e0f3f0;color:#006b70;font-size:11px;border-radius:5px;padding:2px 6px}a{color:#006b83}code,pre{background:#eef3f8}code{padding:2px 5px;border-radius:4px}pre{white-space:pre-wrap;overflow-wrap:anywhere;border-radius:8px;padding:17px;font-size:12px}details{margin:14px 0;padding:11px 15px;background:#f7f9fc;border-radius:7px}summary{cursor:pointer;font-weight:bold;color:#007f86}select{font-size:14px;padding:9px 12px;border:1px solid #bacbdb;border-radius:7px;max-width:100%;color:#172a43;background:#fff}svg{width:100%;height:auto;display:block}.linkrow{display:flex;flex-wrap:wrap;gap:12px;font-size:13px}footer{font-size:12px;color:#61758e;padding:8px}li{margin:9px 0}.timeline{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.phase{background:#f5f8fb;border-top:3px solid #007f86;padding:16px;border-radius:5px}@media(max-width:780px){main{padding:18px 12px}section,header{padding:24px 18px}h1{font-size:24px}.cards,.timeline{grid-template-columns:repeat(2,1fr)}table{font-size:12px}}@media print{body{background:#fff}main{max-width:none;padding:0}header{background:#fff;color:#172a43}header p{color:#61758e}section,header{break-inside:avoid;border-radius:0}.interactive{display:none}}'''
    page='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>전력 예측·이상 후보 진단 비교</title><style>'+css+'</style></head><body><main>'
    page+=f'''<header><div class="eyebrow">LOCAL EXPERIMENTS · FROZEN TEMPORAL EVALUATION · {esc(research.get('reviewed_at',''))}</div><h1>전력 예측 모델 비교와 현재 이상 후보 진단</h1><p>유사 산업 계측 연구를 바탕으로 ARIMA·XGBoost, PatchTST·iTransformer의 축소 재학습 구현, 공식 Chronos-2 사전학습 모델을 비교했습니다. 실제 고장 라벨 없이 가능한 진단 실험의 범위도 분리했습니다.</p><p><strong>검증으로 잠금한 배포 후보: {esc(select_name)}</strong> · 검증 P745 MAE {fmt(selection_metric)} kWh · 별도 5월 P505 MAE {fmt(selected_may['mae_kwh'])} kWh.</p></header>
<div class="cards"><div class="card"><div class="value">{len(method_ids)}개</div><div class="label">주요 예측 모델 계열 · 기준선 별도</div></div><div class="card"><div class="value">P745</div><div class="label">한 번의 원점에서 745시간 예측</div></div><div class="card"><div class="value">P505</div><div class="label">5월 정답이 있는 505시간만 평가</div></div><div class="card"><div class="value">{fmt(cv)}%</div><div class="label">5월 시간당 정답 변동계수</div></div></div>
<section><h2>연구에서 가져온 비교 기준</h2><p>약 5초 간격의 설비별 전압·전류·전력이라는 구조는 HIPE와 SPARK 산업 계측 자료에 가깝습니다. 다만 확인된 HIPE 예측 연구의 단기 구간은 이번 월 단위 fixed-origin 평가와 다릅니다. 1시간 시계열에서 H96/H192/H336/H720을 평가하는 ECL 연구를 장기 예측 설계에 함께 참고했습니다.</p><div class="tablewrap"><table class="prose"><tr><th>자료</th><th>해상도</th><th>프로젝트와 관계</th><th>비교 한계 / 근거</th></tr>{datasets}</table></div><p>PatchTST는 시간 patch와 채널별 공유 Transformer, iTransformer는 변수 token attention을 비교합니다. 이번 모델은 CPU에서 재학습한 compact 설정이며 원 논문의 모든 설정·성능을 재현한 실험이 아닙니다. {citations(['PATCH_PAPER','PATCH_REPO','ITRANS_PAPER','ITRANS_REPO'])}</p><p>Chronos-2는 공식 <code>amazon/chronos-2</code> 120M 사전학습 가중치의 zero-shot 예측을 추가한 비교입니다. 학습 데이터로 새로 학습한 모델과 사전학습 모델은 자원·학습 조건이 다릅니다. {citations(['CHRONOS_PAPER','CHRONOS_CARD','CHRONOS_REPO'])}</p><div class="note caution">PatchTST(2023)·iTransformer(2024)는 문헌의 강한 장기 예측 계열을 선정한 것입니다. Chronos-2는 최근 foundation model 비교입니다. 이를 <strong>2026년 모든 전력 데이터에서의 절대 SOTA 순위</strong>로 해석하지 않습니다. 논문 정규화 MAE와 이번 kWh MAE를 숫자로 직접 비교하지 않습니다.</div><p><a href="research/research_review.html">세부 연구 조사 보고서</a></p></section>
<section><h2>시간 경계와 모델 선택을 먼저 고정</h2><div class="timeline"><div class="phase"><strong>검증 · 선택</strong><p>원점 {esc(protocol['validation_origin'])}<br>과거 {protocol['validation_history_hours']:,}시간 사용<br>미래 745시간 경로의 P745 MAE로 선택</p></div><div class="phase"><strong>내부 테스트</strong><p>원점 {esc(protocol['test_origin'])}<br>과거 {protocol['test_history_hours']:,}시간 사용<br>선택 후 설정을 고정해 평가</p></div><div class="phase"><strong>별도 5월 평가</strong><p>원점 {esc(protocol['may_origin'])}<br>전체 완전시간 학습 후 745시간 예측<br>5/10–5/31의 P505 정답으로만 점수</p></div></div><p>H96/H192/H336/H720은 첫 미관측 시점부터 각각 96/192/336/720개의 값이며, 프로젝트 전체 경로 P745는 offset 0–744입니다. 5월 P505는 offset 240–744를 선택하므로 중간 240시간을 미래 실제 관측으로 채우지 않았습니다. 미래 전류·전압·역률·가동 상태도 예측 특성으로 넣지 않았습니다.</p><div class="note caution"><strong>5월 정답은 앞선 EDA에서 이미 확인했습니다.</strong> 따라서 이번 평가는 완전히 한 번도 보지 않은 prospective holdout이라고 부를 수 없습니다. 이번 비교 전에 선택 규칙을 잠그고 5월 정답으로 하이퍼파라미터·모델 선택을 하지 않는 post-EDA 평가입니다. 검증과 내부 테스트의 두 fixed-origin만으로 계절·연도 전체의 일반화를 입증할 수 없습니다.</div><p>배포 후보는 validation P745 기준으로 잠금했고 내부 테스트·5월 순위로 다시 선택하지 않았습니다. <a href="forecast/deployment_selection_lock.json">선택 잠금 기록</a> · <a href="forecast/protocol.json">평가 프로토콜</a></p></section>
<section><h2>예측 결과 · 동일 목표와 평가 시점으로 비교</h2><div class="tablewrap"><table><tr><th>모델</th><th>검증 P745 MAE</th><th>내부 P745 MAE</th><th>5월 P505 MAE</th><th>5월 RMSE</th><th>5월 MAPE %</th><th>5월 예측 편향</th><th>5월 R²</th></tr>{model_rows}</table></div><p class="caption">MAE·RMSE·편향 단위는 잠정 모듈 합계 kWh. R²는 예측 CSV에서 1−SSE/SST로 계산했습니다. 5월 실제값의 표준편차는 {fmt(actual.std())}로 작아 R²가 평균 수준의 작은 차이에 민감하며, 낮은 MAPE를 ‘높은 고장/예측 정확도’로 바꾸어 표현하지 않습니다. 음수 R²는 평가 기간의 실제 평균을 상수로 예측한 경우보다 제곱오차가 크다는 뜻으로, 작은 시간별 변동을 잘 따라간다는 근거가 부족합니다.</p>{practical_note}{may_note}{embed(charts[0],'검증 및 내부 테스트의 H96 H192 H336 H720 P745 MAE 비교')}{embed(charts[2],'잠금한 배포 후보를 표시한 검증 내부테스트 및 별도5월 MAE 비교')}<h3>입력 정보와 학습 조건의 차이</h3><p>iTransformer는 13개 모듈의 과거 시계열을 다변량으로 입력하고 출력 합계를 같은 목표에 맞췄습니다. 나머지 주요 모델은 과거 모듈 합계 1개 시계열을 중심으로 사용하며 XGBoost는 예측 시점에 알 수 있는 달력을 추가합니다. 따라서 성능 차이를 architecture만의 효과로 분리할 수 없습니다. 모든 모델이 같은 정보량을 받은 통제 실험과 구분해야 합니다.</p>{configs}</section>{uncertainty}
<section><h2>5월 실측·예측과 잔차</h2><p>{esc(select_name)}의 5월 MAE는 <strong>{fmt(selected_may['mae_kwh'])} kWh</strong>, 예측−실측 평균 편향은 <strong>{fmt(selected_may['bias_pred_minus_actual_kwh'])} kWh</strong>입니다. 가이드 선형 환산을 적용하면 동일 표본의 요금 MAE는 {fmt(selected_may['mae_kwh']*180,0)}원, 탄소 MAE는 {fmt(selected_may['mae_kwh']*.424)} kgCO₂입니다.</p>{level_note}{embed(charts[1],'5월 실제 및 모델 예측 시계열과 선택모델의 예측빼기실측 잔차')}<p class="caption">작은 차이를 보기 위해 세로축을 확대했습니다. 정답 기간은 5월 전체 달이 아니라 5/10 00시–5/31 00시이며 끝 시각을 포함합니다. 요금 180원/kWh·탄소 0.424 kgCO₂/kWh는 프로젝트 가정값입니다.</p>{interval_note}</section>
<section class="interactive"><h2>모델을 선택해 같은 5월 구간 확인</h2><label for="forecastModel">모델 </label><select id="forecastModel"></select><p class="caption" id="forecastNote"></p><svg id="forecastChart" viewBox="0 0 1020 420" role="img" aria-label="선택 모델의5월예측과실측및잔차"></svg></section>
<section><h2>진단 실험 · 현재 전기품질 후보와 실제 고장을 구분</h2><div class="note caution"><strong>확정된 고장/삽입 이상치 정답 라벨이 없습니다.</strong> 전압·역률 임계값의 약한 후보 라벨과 모델 점수의 일치도를 평가했습니다. 이는 실제 고장 정밀도·재현율·조기 탐지 성능이 아닙니다.</div><p>원시 전체를 읽고 모듈×시간당 720개 5초 슬롯에서 무작위 4개를 선택해 {sample['sample_rows']:,}행을 추출했습니다. 모든 완전시간 관측의 포함 확률은 동일하며 후보를 추가 추출하지 않았습니다. 12–2월 학습 / 3월 검증 / 4월 테스트를 분리했고 4월 테스트 {test_sample['rows']:,}행의 약한 후보는 {test_sample['candidate_rows']}개입니다.</p><div class="tablewrap"><table><tr><th>모델</th><th>AP</th><th>AUROC</th><th>정밀도</th><th>재현율</th><th>F1</th><th>찾은 후보/전체</th><th>규칙 정상 경보</th></tr>{diag_rows}</table></div>{embed(OUT/'diagnosis'/'figures'/'diagnosis_overview.png','전기품질 약한후보분류 및 AE 재구성 이상점수 결과')}<p>규칙 모사 XGBoost는 라벨을 만드는 전압·역률을 입력으로 사용하므로 높은 점수가 독립적인 진단 근거가 아닙니다. 규칙 센서를 제외한 전달 테스트는 전류·전력·무효전력만으로 후보를 구분할 수 있는지 확인한 별도 비교입니다. 현재 관측 판단이며 장래 고장 예측이 아닙니다.</p><p>전체 학습 AE는 모든 학습 관측으로 비지도 재구성 학습을 하고, 약한 정상 proxy AE는 규칙 후보를 제외한 준지도 학습입니다. 두 모델의 기본 임계값은 3월 약한 정상 점수 99.5분위에서 고정하므로 임계값 보정은 준지도입니다. 후보 비율이 매우 낮아 좋은 점수 순위와 낮은 경보 정밀도가 동시에 나타날 수 있습니다. 테스트를 보고 임계값을 다시 조정하지 않았습니다.</p><p><a href="diagnosis/diagnosis_report.html">진단 세부 보고서</a> · <a href="diagnosis/diagnosis_model_metrics.json">진단 모델 지표</a></p></section>
<section><h2>프로젝트 판단과 다음 실행</h2><ol><li><strong>잠금한 검증 기준으로 배포 후보를 유지.</strong> {esc(select_name)}를 현재 비교의 후보로 정하고 내부·5월 결과는 위험과 성능을 확인하는 자료로 사용합니다. 더 복잡한 모델의 가치 판단은 기준선 대비 MAE, 실행 비용, 다른 기간의 반복 검증이 함께 필요합니다.</li><li><strong>계측 정의부터 확정.</strong> 13개 모듈의 계층이 확인되지 않아 공장 전체와 하위 계측이 중복될 수 있습니다. 5월 CSV에는 단위가 직접 적혀 있지 않아 학습의 kWh 합계와 같은 범위·스케일이라는 가정으로 비교했습니다. 이 합계를 공장 총량·절감액으로 확정하지 않습니다.</li><li><strong>예측 검증을 확대.</strong> 현재 두 fixed-origin 평가와 소수 검증 설정만으로 통계적 우월성을 주장하기 어렵습니다. 더 긴 기간과 여러 사전 정의 원점, 생산량·운전 계획처럼 origin 시점에 확보되는 설명 변수를 추가해야 합니다.</li><li><strong>실제 진단 라벨을 확보.</strong> 점검·교체·정지 이력과 삽입 위치를 연결해 실제 고장/정상 라벨을 만들고, 시간 순서와 미관측 설비 테스트를 분리합니다. 현재 후보·AE 점수는 근거를 확인할 탐색 목록으로 사용합니다.</li><li><strong>대시보드는 가정과 근거를 함께 표시.</strong> 실제/예측·MAE·잔차, 고정 예측 원점, 설비 필터, 이상 점수의 기여 센서와 후보 묶음을 보여줍니다. 행별 경보를 사건 수로 읽지 않도록 지속시간·재알림 제한을 설계합니다.</li></ol></section>
<section><h2>재현과 결과 파일</h2><div class="linkrow"><a href="build_model_report.py">통합 보고서 생성 코드</a><a href="forecast/run_forecasts.py">예측 실행 코드</a><a href="forecast/arima_xgb.py">ARIMA/XGBoost 코드</a><a href="forecast/itransformer_experiment.py">iTransformer 코드</a><a href="forecast/chronos2_experiment.py">Chronos-2 코드</a><a href="diagnosis/diagnosis_experiment.py">진단 실행 코드</a></div><pre>&amp; '{esc(str(OUT/'.venv'/'Scripts'/'python.exe'))}' '{esc(str(OUT/'build_model_report.py'))}'</pre><p>이 명령은 잠금된 결과 JSON·예측 CSV를 읽어 보고서와 그래프만 재생성합니다. 모델을 재학습하거나 배포 선택을 바꾸지 않습니다.</p><div class="linkrow"><a href="forecast/comparison_results.json">예측 비교 지표 JSON</a><a href="forecast/all_predictions.csv">전체 예측 CSV</a><a href="forecast/deployment_selection_lock.json">배포 선택 잠금 JSON</a><a href="diagnosis/diagnosis_report.json">진단 상세 JSON</a><a href="research/literature_review.json">연구 조사 JSON</a></div></section><footer>원본 데이터 수정 없음 · 모든 차트와 5월 상호작용 시계열은 HTML 안에 포함 · 연구 문헌의 성능과 이번 로컬 실험 성능을 분리</footer>'''
    script='''<script>const payload=PAYLOAD;const menu=document.getElementById('forecastModel'),chart=document.getElementById('forecastChart');Object.keys(payload.models).forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=payload.models[k].name;menu.appendChild(o)});menu.value=payload.selected;function draw(){const key=menu.value,m=payload.models[key],a=payload.actual.map(x=>x[1]),p=m.data,r=p.map((v,i)=>v-a[i]),L=90,R=995,n=a.length;let parts='';function panel(values,min,max,T,B,color){const pad=(max-min||1)*.12;min-=pad;max+=pad;const X=i=>L+i/(n-1)*(R-L),Y=v=>B-(v-min)/(max-min)*(B-T);for(let j=0;j<=3;j++){let v=min+(max-min)*j/3,y=Y(v);parts+=`<line x1="${L}" x2="${R}" y1="${y}" y2="${y}" stroke="#dce5ef"/><text x="${L-10}" y="${y+4}" text-anchor="end" font-size="12" fill="#61758e">${v.toLocaleString('ko-KR',{maximumFractionDigits:1})}</text>`}parts+=`<polyline points="${values.map((v,i)=>X(i)+','+Y(v)).join(' ')}" fill="none" stroke="${color}" stroke-width="1.1"/>`;return {X,Y}}const all=a.concat(p),top=panel(a,Math.min(...all),Math.max(...all),20,210,'#8495a8');parts+=`<polyline points="${p.map((v,i)=>top.X(i)+','+top.Y(v)).join(' ')}" fill="none" stroke="${m.color}" stroke-width="2"/>`;const bottom=panel(r,Math.min(...r,0),Math.max(...r,0),260,385,m.color);parts+=`<line x1="${L}" x2="${R}" y1="${bottom.Y(0)}" y2="${bottom.Y(0)}" stroke="#61758e" stroke-dasharray="4 4"/>`;[0,Math.floor(n/3),Math.floor(2*n/3),n-1].forEach(i=>{parts+=`<text x="${top.X(i)}" y="411" text-anchor="middle" font-size="12" fill="#61758e">${payload.actual[i][0].slice(5,10)}</text>`});parts+='<text x="10" y="112" font-size="12" fill="#61758e">kWh</text><text x="10" y="320" font-size="12" fill="#61758e">잔차</text>';chart.innerHTML=parts;chart.setAttribute('aria-label',m.name+'의5월실측과예측및잔차');document.getElementById('forecastNote').textContent=m.name+' · 회색=실측 / 색상=예측 · 아래=예측−실측 · 두 패널 세로축 확대 · 동일 505시간';}menu.addEventListener('change',draw);draw();</script>'''
    page=page.replace('학습 데이터로 새로 학습한 모델과 사전학습 모델은 자원·학습 조건이 다릅니다.',
        '점예측은 공식 predict_df 출력의 0.5분위 중앙값이며 사전에 고정했습니다. 학습 데이터로 새로 학습한 모델과 사전학습 모델은 자원·학습 조건이 다릅니다.')
    page=page.replace('평가 기간의 실제 평균을 상수로 예측한 경우보다','평가 기간을 사후적으로 확인한 실제 평균을 상수로 예측한 경우보다')
    page=page.replace('미래 전류·전압·역률·가동 상태도 예측 특성으로 넣지 않았습니다.</p>',
        '미래 전류·전압·역률·가동 상태도 예측 특성으로 넣지 않았습니다.</p><p>이번 H96/H192/H336/H720은 공통 P745 예측 경로의 앞부분(prefix)을 평가한 값이며, 각 길이의 출력 head를 따로 학습한 실험이 아닙니다. 논문의 horizon별 별도 학습 결과와 구분해야 합니다.</p>')
    page=page.replace('<a href="forecast/run_forecasts.py">예측 실행 코드</a>',
        '<a href="forecast/predict_forecast.py">잠금 모델 공통 추론 코드</a><a href="forecast/run_forecasts.py">예측 학습 실행 코드</a>')
    inference=f'''<h3>학습된 모델로 추론만 실행</h3><pre>&amp; '{esc(str(OUT/'.venv'/'Scripts'/'python.exe'))}' '{esc(str(OUT/'forecast'/'predict_forecast.py'))}' --model selected --steps 745</pre><p>기본 입력은 제공 학습의 마지막 완전시간까지입니다. 사용자 이력은 <code>--history</code>로 <code>timestamp,energy_kwh</code> 열의 연속·고유·완전한 1시간 CSV(최소 960행)를 넣습니다. iTransformer의 사용자 이력에는 <code>--module-history</code>로 동일 시간축의 13개 모듈 wide CSV도 필요합니다. 저장된 설정·파라미터를 사용하며 미래 정답을 읽거나 모델을 재학습하지 않습니다.</p>'''
    page=page.replace('<h2>재현과 결과 파일</h2>','<h2>재현과 결과 파일</h2>'+inference)
    page+=script.replace('PAYLOAD',json.dumps(payload,ensure_ascii=False).replace('</','<\\/'))+'</main></body></html>'
    (OUT/'project_modeling_report.html').write_text(page,encoding='utf-8')
    return {'report':str(OUT/'project_modeling_report.html'),'models':len(ids),'selected_model':selected,'embedded_figures':4,'may_rows_per_model':505,'generated_pngs':[str(p) for p in charts]}

if __name__=='__main__':
    comparison=load(OUT/'forecast'/'comparison_results.json');predictions=pd.read_csv(OUT/'forecast'/'all_predictions.csv',parse_dates=['id'])
    lock=load(OUT/'forecast'/'deployment_selection_lock.json');protocol=load(OUT/'forecast'/'protocol.json')
    research=load(OUT/'research'/'literature_review.json');diagnosis=load(OUT/'diagnosis'/'diagnosis_report.json')
    print(json.dumps(build_report(comparison,predictions,lock,protocol,research,diagnosis),ensure_ascii=True,indent=2))
