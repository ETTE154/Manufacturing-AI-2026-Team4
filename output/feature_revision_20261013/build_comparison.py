"""Merge frozen feature-revision experiments and score May without reselection."""
from pathlib import Path
import sys,os,hashlib,json
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'output/analysis/.plot_deps'))
os.environ['MPLCONFIGDIR']=str(OUT/'tmp/mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'font.size':11})

def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def metrics(a,p):
    a,p=np.asarray(a,float),np.asarray(p,float);e=p-a
    return {'count':len(a),'mae_kwh':float(np.abs(e).mean()),'rmse_kwh':float(np.sqrt((e*e).mean())),
        'mape_percent':float((np.abs(e)/np.abs(a)).mean()*100),'r2':float(1-(e*e).sum()/((a-a.mean())**2).sum()),'bias_pred_minus_actual_kwh':float(e.mean())}
def winner(models):
    best=min(models,key=lambda k:models[k]['validation_metrics']['mae_kwh'])
    for k in models:
        if 'baseline' in k and models[k]['validation_metrics']['mae_kwh']<=models[best]['validation_metrics']['mae_kwh']+1e-6:return k
    return best
def previous_value(old,name,scenario):
    item=old['scenarios'][scenario]['models'].get(name)
    return item['test_metrics']['mae_kwh'] if item and scenario=='dayahead24' else None

def classical_card(name,m,old,scenario):
    metric=m['validation_metrics'];test=m.get('test_metrics',m.get('internal_test_metrics'))
    if 'path_24_values' in metric:
        v=metric['path_24_values'];t=test['path_24_values'];ep={str(i):test[f'lead_{i}'] for i in [1,6,24]}
    else:
        v=metric.get('path_P745',metric.get('path_745_values',metric.get('path_metrics',metric)));t=test.get('path_P745',test.get('path_745_values',test.get('path_metrics',test)));ep={}
    labels={'xgboost':'새 XGBoost','ridge':'새 Ridge','elasticnet':'새 ElasticNet','arimax':'새 ARIMAX','baseline':'EMA720 기준선'}
    schema=m.get('final_feature_schema') or m.get('saved_model',{}).get('feature_schema') or []
    cfg=m['selected_config'];group=cfg.get('group',cfg.get('exog',cfg.get('kind','')))
    return {'display_name':labels[name],'model_family':name,'validation_metrics':v,'test_metrics':t,'endpoint_metrics':ep,
        'selected_config':cfg,'input_description':group,'input_feature_count':len(schema),'input_feature_schema':schema,
        'feature_selection_note':'각 fold 학습 prefix에서 선정; 최종 센서 schema는 3월 말 기준 고정',
        'feature_selection':m.get('feature_selection',m.get('feature_selection_paths')),'paths':m['paths'],'saved_model':m['saved_model'],
        'input_scope':m.get('input_scope'),'importance':m.get('importance'),'trials_count':m.get('trials_count'),
        'previous_test_mae_kwh':previous_value(old,name,scenario),'previous_control':False}

def neural_card(name,m,old,scenario):
    label='새 iTransformer 다변량' if name=='itransformer' else '새 PatchTST 설비별 합산'
    ep={str(i):m['test_lead_metrics'][f'lead{i}'] for i in [1,6,24]} if scenario=='dayahead24' else {}
    return {'display_name':label,'model_family':name,'validation_metrics':m['validation_metrics'],'test_metrics':m['test_metrics'],
        'endpoint_metrics':ep,'selected_config':m['selected_config'],'input_description':m['target_mode'],
        'input_feature_count':m['input_feature_count'],'input_features':m['input_features'],
        'feature_selection_note':f'3월 말 추가 센서 {m["selected_sensor_count"]}개 · {m["cross_channel_or_auxiliary"]}',
        'paths':{'validation_csv':m['prediction_csv']['validation'],'test_csv':m['prediction_csv']['test'],'future_csv':m['prediction_csv']['future']},
        'saved_model':m['saved_model'],'reload_evidence':m['reload_evidence'],'loss_definition':m['loss_definition'],
                'previous_test_mae_kwh':previous_value(old,name,scenario),'previous_control':False}

def plot_frame(key,label,df,kind):
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    df=df.sort_values('id');x=pd.to_datetime(df.id);a=df.actual_kwh.to_numpy();p=df.predicted_kwh.to_numpy()
    m=metrics(a,p)
    fig,axes=plt.subplots(3,1,figsize=(14.5,10.5),gridspec_kw={'height_ratios':[2,1,1.2]},constrained_layout=True)
    axes[0].plot(x,a,color='#127886',lw=1,label='실측');axes[0].plot(x,p,color='#e47c39',lw=1.4,label='예측')
    axes[0].set_title(f'{label} | {"April: 29 daily origins × 24h" if kind=="day" else "May: one origin, scored leads 241–745"}')
    axes[0].set_ylabel('잠정 전력량 kWh');axes[0].legend(loc='upper right');axes[0].grid(alpha=.2)
    axes[1].plot(x,p-a,color='#6855b7',lw=.8);axes[1].axhline(0,color='#557486',lw=.8)
    axes[1].set_ylabel('예측-실측 kWh');axes[1].grid(alpha=.2)
    if kind=='day':
        d=df[df.lead_hours==24];tx=pd.to_datetime(d.id)
        axes[2].plot(tx,d.actual_kwh,color='#127886',marker='o',ms=3,label='lead24 실측')
        axes[2].plot(tx,d.predicted_kwh,color='#e47c39',marker='o',ms=3,label='lead24 예측')
        axes[2].set_ylabel('lead24 kWh');axes[2].set_title(f'Path MAE {m["mae_kwh"]:.3f} | R² {m["r2"]:.5f} | lead24 MAE {np.abs(d.predicted_kwh-d.actual_kwh).mean():.3f}')
    else:
        axes[2].scatter(a,p,s=8,alpha=.6,color='#127886');lo=min(a.min(),p.min());hi=max(a.max(),p.max())
        axes[2].plot([lo,hi],[lo,hi],color='#748b9a',ls='--',label='실측=예측');axes[2].set_xlabel('실측 kWh');axes[2].set_ylabel('예측 kWh')
        axes[2].set_title(f'MAE {m["mae_kwh"]:.3f} | RMSE {m["rmse_kwh"]:.3f} | R² {m["r2"]:.5f}')
    axes[2].legend(loc='upper right');axes[2].grid(alpha=.2)
    png=dest/f'{key}.png';svg=dest/f'{key}.svg';fig.savefig(png,dpi=150);fig.savefig(svg);plt.close(fig)
    return {'label':label+' · '+('4월 24시간' if kind=='day' else '5월 장기'),'png':str(png),'svg':str(svg)}

def main():
    old=load(ROOT/'output/optimization_20261013/forecast/comparison_results.json')
    cls=load(OUT/'classical/results.json');lg=load(OUT/'classical/long745/results.json')
    cards={s:{} for s in ['dayahead24','long745']};locks={};figures={}
    for name,m in cls['models'].items():
        key='classical_'+name;cards['dayahead24'][key]=classical_card(name,m,old,'dayahead24')
        plots=m['plots'];figures[key+'_day']={'label':cards['dayahead24'][key]['display_name']+' · 4월 24시간','png':plots['april_day_png'],'svg':plots['april_day_svg']}
    locks['classical_day']=sha(OUT/'classical/selection_lock.json');locks['classical_long']=sha(OUT/'classical/long745/selection_lock.json')
    for name,m in lg['models'].items():cards['long745']['classical_'+name]=classical_card(name,m,old,'long745')
    for scenario in cards:
        for name in ['itransformer','patchtst']:
            d=OUT/'neural'/scenario/name;m=load(d/'results.json')
            assert m['status']=='complete' and not m['may_labels_read']
            key='neural_'+name;cards[scenario][key]=neural_card(name,m,old,scenario)
            locks[f'{scenario}/{name}']=sha(d/'selection_lock.json')
            if scenario=='dayahead24' and m.get('plots'):
                figures[key+'_day']={'label':cards[scenario][key]['display_name']+' · 4월 24시간','png':m['plots']['test_forecast_png'],'svg':m['plots']['test_forecast_svg']}
    for scenario in cards:
        for name in ['chronos2','baseline']:
            if scenario=='dayahead24' and name=='baseline':continue
            prev=old['scenarios'][scenario]['models'][name]
            t=prev['test_metrics'] if scenario=='dayahead24' else prev['internal_test_metrics']
            d=ROOT/'output/optimization_20261013/forecast'/scenario/('baselines' if name=='baseline' else name)
            cards[scenario]['previous_'+name]={'display_name':'기존 Chronos-2 혼합' if scenario=='dayahead24' else ('기존 Chronos-2' if name=='chronos2' else '과거 평균 기준선'),
                'model_family':name,'validation_metrics':prev['validation_metrics'],'test_metrics':t,'endpoint_metrics':{'24':prev['test_endpoint_metrics']} if scenario=='dayahead24' else {},
                'selected_config':prev['selected_config'],'paths':{'validation_csv':str(d/'validation_predictions.csv'),'test_csv':str(d/'test_predictions.csv'),'future_csv':str(d/'future_path.csv')},
                'previous_control':True,'input_description':'기존 공통 평가 기간의 합계 전력량 모델'}
            if scenario=='dayahead24':figures['previous_chronos2_day']={'label':'기존 Chronos-2 혼합 · 4월 24시간','png':old['figures']['chronos2']['april_day_png'],'svg':old['figures']['chronos2']['april_day_svg']}
    selected={s:winner(m) for s,m in cards.items()}
    lock={'selected_models':selected,'criterion':'Validation path MAE only; numerical tie prefers simple baseline',
        'new_model_selection_lock_sha256':locks,'April_or_May_scores_used_for_selection':False,
        'feature_sha256':sha(OUT/'data/hourly_features.csv'),'post_EDA_limit':'April/May were seen in previous experiments; no pristine prospective claim'}
    write(OUT/'comparison_selection_lock.json',lock)
    # All input schemas and hyperparameters are frozen before re-opening May labels.
    truth=pd.read_csv(ROOT/'may-dataset/rtu_ground_truth_may.csv');truth['id']=pd.to_datetime(truth['id']);assert len(truth)==505
    may_dir=OUT/'may_predictions';may_dir.mkdir(exist_ok=True)
    for key,m in cards['long745'].items():
        fut=pd.read_csv(m['paths']['future_csv']);fut['id']=pd.to_datetime(fut['id'])
        assert len(fut)==745
        if 'actual_kwh' in fut:assert fut.actual_kwh.isna().all()
        scored=fut.loc[fut.id.isin(truth.id)].sort_values('id').reset_index(drop=True)
        assert len(scored)==505 and scored.id.equals(truth.id.reset_index(drop=True))
        assert scored.lead_hours.min()==241 and scored.lead_hours.max()==745
        scored['actual_kwh']=truth.hourly_pow.to_numpy(dtype=np.float64)
        scored['absolute_error_kwh']=np.abs(scored.predicted_kwh-scored.actual_kwh)
        m['internal_test_metrics']=m['test_metrics'];m['test_metrics']=metrics(scored.actual_kwh,scored.predicted_kwh);m['may_metrics']=m['test_metrics']
        m['paths']['may_csv']=str(may_dir/f'{key}.csv');scored.to_csv(m['paths']['may_csv'],index=False,float_format='%.17g')
        figures[key+'_may']=plot_frame(key+'_may',m['display_name'],scored,'may')
    for key,m in cards['dayahead24'].items():
        if key+'_day' not in figures:figures[key+'_day']=plot_frame(key+'_day',m['display_name'],pd.read_csv(m['paths']['test_csv']),'day')
    assert selected=={s:winner(m) for s,m in cards.items()}
    conclusions=[
        '기존 시간 정렬·미래 입력 누출 오류는 발견되지 않았습니다. 문제는 코드 오류로만 설명되지 않습니다.',
        '442개 시간 통계 피처를 재검토했지만 지연 상관의 최대값은 날짜 블록 대조의 우연 범위 안에 있었고, 3월 말 기준 안정적 추가 센서는 0개였습니다.',
        '전압·전류·역률의 동시 물리식은 activePower와 높은 상대 상관을 보입니다. 그러나 같은 시간 센서는 미래에 알 수 없으므로 이 상관을 미래 예측 성능으로 해석할 수 없습니다.',
        '신규 XGBoost의 4월 MAE는 79.293으로 기존 XGBoost 79.350보다 소폭 낮지만, 기존 Chronos 혼합 대비 차이의 날짜 단위 불확실성 구간은 0을 포함합니다.',
        '설비별 다변량 모델까지 다시 학습했으며, 검증 기준의 우위와 실제 변동 설명력이 충분한지는 아래 표에서 기준선과 함께 판단해야 합니다.'
    ]
    result={'status':'complete','scenarios':{s:{'models':m,'selected_model':selected[s],'selection_basis':'Validation path MAE only'} for s,m in cards.items()},
        'comparison_selection_lock':str(OUT/'comparison_selection_lock.json'),'selection_lock_sha256':sha(OUT/'comparison_selection_lock.json'),
        'figures':figures,'conclusions':conclusions,'classical_paired_previous':cls.get('paired_previous_models'),
        'unit':'Provisional kWh under original guide and aggregate-scope assumptions','May_not_used_for_reselection':True}
    write(OUT/'comparison_results.json',result)
    print(json.dumps({'selected':selected,'day_models':len(cards['dayahead24']),'long_models':len(cards['long745']),'figure_pairs':len(figures)},ensure_ascii=True))
if __name__=='__main__':main()
