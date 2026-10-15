"""Combine independently locked model families, then score May and make figures.

No training occurs here. Global selection uses validation only and remains fixed.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import os
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from opt_common import OUT,ROOT,MAY_SOURCE,SEED,load_target,metrics,day_summary,write_json

MODEL_IDS=['arima','xgboost','patchtst','itransformer','chronos2','baseline']
DISPLAY={'arima':'ARIMA','xgboost':'XGBoost','patchtst':'PatchTST compact',
         'itransformer':'iTransformer univariate','chronos2':'Chronos-2 frozen zero-shot','baseline':'Selected simple baseline'}

def location(scenario,model):
    return OUT/scenario/('baselines' if model=='baseline' else model)

def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def primary(value,scenario):
    if 'mae_kwh' in value:return value
    if scenario=='dayahead24':return value['path_24_values']
    return value['path_P745'] if 'path_P745' in value else value['P745']

def read_model(scenario,model):
    directory=location(scenario,model);result=load_json(directory/'results.json');lock=load_json(directory/'selection_lock.json')
    selected=result.get('selected_config',result.get('frozen_config',lock.get('selected_config',lock.get('frozen_config'))))
    trials=result.get('attempted_configs',lock.get('configuration_count',len(lock.get('search_records',lock.get('candidate_results',lock.get('candidates',[]))))))
    if not trials:trials=result.get('trials_count',6 if model in ['patchtst','itransformer'] else None)
    card={'display_name':DISPLAY[model],'validation_metrics':primary(result['validation'],scenario),
          'selected_config':selected,'trials_count':trials,
          'trained_model_fit_count':0 if model=='chronos2' else None,'validation_refit_count':None,
          'fit_count_note':'Exact fit counts are not inferred from trial counts; inner early-stop fits, refits, test/final fits and intermediate30-trial runs differ. Native per-model logs preserve these details.',
          'input_scope':result.get('input_scope',result.get('input','Aggregate univariate past target; iTransformer has one variate token')),
          'saved_model':result.get('saved_model'),'result_path':str(directory/'results.json'),
          'selection_lock_path':str(directory/'selection_lock.json'),
          'model_card_notes':result.get('model_card_notes',result.get('limitations',[]))}
    if model=='baseline':
        names={'train_mean':'학습 전체 평균','train_median':'학습 전체 중앙값','expanding_mean':'확장 과거 평균',
               'recent_mean':'최근 평균','ema':'지수가중 평균','calendar_ridge':'달력 Ridge'}
        card['display_name']='기준선: '+names[selected['name']]+(f' ({selected["window"]}시간)' if 'window' in selected else '')
    if model=='chronos2' and selected.get('calibration')!='none':
        card['display_name']='Chronos-2 + 과거 평균 혼합' if selected['calibration']=='historical_mean' else 'Chronos-2 + 과거 EMA 혼합'
        card['blend_weights']={'frozen_Chronos_fraction':selected['alpha'],'past_level_fraction':1-selected['alpha'],'pretrained_weights_updated':False}
    if model=='chronos2':card['calibration_candidates']=trials
    if scenario=='dayahead24':
        card['test_metrics']=primary(result['test'],scenario)
        card['validation_endpoint_metrics']=result['validation']['endpoint_lead_24']
        card['test_endpoint_metrics']=result['test']['endpoint_lead_24']
        card['validation_origin_count']=result['validation']['origin_count']
        card['test_origin_count']=result['test']['origin_count']
    else:card['internal_test_metrics']=primary(result['test'],scenario)
    return card,lock

def freeze_selection():
    protocol=load_json(OUT/'protocol.json');scenarios={};hashes={}
    for scenario in ['dayahead24','long745']:
        models={}
        for model in MODEL_IDS:
            card,lock=read_model(scenario,model)
            if lock.get('selection_uses_May_labels',False) or lock.get('selection_uses_may_labels',False) or lock.get('may_labels_read',False):
                raise ValueError('A model lock indicates use of May labels')
            models[model]=card
            path=location(scenario,model)/'selection_lock.json'
            hashes[f'{scenario}/{model}']=hashlib.sha256(path.read_bytes()).hexdigest()
        winner=min(models,key=lambda model:models[model]['validation_metrics']['mae_kwh'])
        # Only numerical-equivalence ties prefer the simpler explicit baseline.
        if models['baseline']['validation_metrics']['mae_kwh']<=models[winner]['validation_metrics']['mae_kwh']+1e-6:winner='baseline'
        scenarios[scenario]={'models':models,'selected_model':winner,
                              'selection_basis':'Validation path MAE only; numerical tie within1e-6 kWh prefers simple baseline',
                              'point_count':24 if scenario=='dayahead24' else 745,
                              'test_period':'2025-04-01 00:00 through2025-04-29 23:00,29 day origins' if scenario=='dayahead24' else '2025-03-29 fixed origin through2025-04-29 00:00'}
    selection={'scenarios':{scenario:{'selected_model':card['selected_model'],'criterion':card['selection_basis']} for scenario,card in scenarios.items()},
               'selection_lock_sha256':hashes,'target_sha256':protocol['target_sha256'],'selection_uses_May_labels':False}
    write_json(OUT/'comparison_selection_lock.json',selection)
    combined={'status':'validation_selection_frozen_before_May','scenarios':scenarios,
              'limitations':["April and May were seen in earlier project EDA/evaluations; this optimization uses an explicit post-EDA lock, not a pristine prospective test.",
                             "May sensor context is unavailable; no May labels are used as rolling day-ahead inputs. April day-ahead and May single-origin long forecasts are different tasks.",
                             "All hourly values are a provisional13-module sum with unknown parent/child meter hierarchy; they are not certified plant electricity consumption.",
                             "Five chronological day blocks do not establish generalization across independent plants or years. Long HPO has only one validation origin.",
                             "The search is finite and budgeted; no claim of global optimum or paper-benchmark reproduction.",
                             "Additional search does not guarantee lower May error: a validation improvement can worsen the already fixed May evaluation.",
                             "Univariate iTransformer has one variate token, so its usual multivariate cross-variate attention benefit is absent."],
              'protocol':protocol,'figures':{},'selection_lock':str(OUT/'comparison_selection_lock.json')}
    write_json(OUT/'comparison_results.json',combined)
    print(json.dumps({'global_selections':selection['scenarios'],'May_labels_read':False},indent=2))

def verify_locks():
    locked=load_json(OUT/'comparison_selection_lock.json')
    for key,expected in locked['selection_lock_sha256'].items():
        scenario,model=key.split('/');path=location(scenario,model)/'selection_lock.json'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Model selection changed after global lock')
    target=load_target();protocol=load_json(OUT/'protocol.json')
    if locked['target_sha256']!=protocol['target_sha256']:raise ValueError('Target protocol changed')
    return locked

def paired_day_interval(model_frame,baseline_frame):
    model=model_frame.groupby('origin')['absolute_error_kwh'].mean().sort_index()
    base=baseline_frame.groupby('origin')['absolute_error_kwh'].mean().sort_index()
    if not model.index.equals(base.index):raise ValueError('Paired origins differ')
    delta=model.to_numpy()-base.to_numpy();rng=np.random.default_rng(SEED)
    boot=np.mean(rng.choice(delta,size=(2000,len(delta)),replace=True),axis=1)
    return {'delta_mae_model_minus_selected_baseline_kwh':float(delta.mean()),
            'conditional_origin_bootstrap_95_interval_kwh':np.quantile(boot,[.025,.975]).tolist(),
            'origins':len(delta),'resamples':2000,'seed':SEED,
            'limitation':'Conditional on these29 April origins and frozen fits; not a confidence bound for different months, plants, or HPO/model-training uncertainty'}

def score_may():
    locked=verify_locks();combined=load_json(OUT/'comparison_results.json')
    # All family and global choices have already been recorded before this read.
    labels=pd.read_csv(MAY_SOURCE,usecols=['id','hourly_pow']);labels['id']=pd.to_datetime(labels['id']);labels=labels.sort_values('id').set_index('id')['hourly_pow']
    if len(labels)!=505:raise ValueError('Unexpected May target length')
    dayframes={};allframes=[]
    for model in MODEL_IDS:
        directory=location('long745',model);path=pd.read_csv(directory/'future_path.csv')
        path['id']=pd.to_datetime(path['id']);path['origin']=pd.to_datetime(path['origin'])
        if len(path)!=745 or path['horizon_offset_hours'].min()!=0 or path['horizon_offset_hours'].max()!=744:
            raise ValueError('Wrong long fixed path')
        if not path['actual_kwh'].isna().all():raise ValueError('Future path was not unlabeled before May evaluation')
        selected=path.set_index('id').reindex(labels.index).reset_index()
        if selected.predicted_kwh.isna().any():raise ValueError('May timestamps are not covered')
        selected['model']=model;selected['split']='May_P505';selected['actual_kwh']=labels.to_numpy(dtype=np.float64)
        selected['predicted_kwh']=selected.predicted_kwh.to_numpy(dtype=np.float64)
        selected['absolute_error_kwh']=np.abs(selected.actual_kwh-selected.predicted_kwh).astype(np.float64)
        selected.to_csv(directory/'may_predictions.csv',index=False,encoding='utf-8-sig')
        card=combined['scenarios']['long745']['models'][model]
        card['may_metrics']=metrics(selected.actual_kwh,selected.predicted_kwh)
        allframes.append(selected)
        day=pd.read_csv(location('dayahead24',model)/'test_predictions.csv');day['model']=model
        day['actual_kwh']=day.actual_kwh.to_numpy(dtype=np.float64);day['predicted_kwh']=day.predicted_kwh.to_numpy(dtype=np.float64)
        day['absolute_error_kwh']=np.abs(day.actual_kwh-day.predicted_kwh)
        dayframes[model]=day
    for model,frame in dayframes.items():
        combined['scenarios']['dayahead24']['models'][model]['paired_vs_selected_baseline']=paired_day_interval(frame,dayframes['baseline'])
    combined['scenarios']['long745']['may_observed_best_not_selected']=min(MODEL_IDS,key=lambda m:combined['scenarios']['long745']['models'][m]['may_metrics']['mae_kwh'])
    combined['scenarios']['long745']['may_evaluation_rows']=505
    combined['scenarios']['long745']['may_offsets']='240..744';combined['scenarios']['long745']['may_leads']='241..745'
    combined['scenarios']['dayahead24']['April_test_rows']=696;combined['scenarios']['dayahead24']['endpoint_test_rows']=29
    combined['status']='all_locked_May_scored_selection_unchanged'
    pd.concat(allframes,ignore_index=True).to_csv(OUT/'all_may_predictions.csv',index=False,encoding='utf-8-sig')
    pd.concat(dayframes.values(),ignore_index=True).to_csv(OUT/'all_April_day_predictions.csv',index=False,encoding='utf-8-sig')
    write_json(OUT/'comparison_results.json',combined)
    print(json.dumps({scenario:{'selected_model':card['selected_model'],'scores':{
        model:{'validation':item['validation_metrics']['mae_kwh'],'test':item.get('test_metrics',item.get('internal_test_metrics'))['mae_kwh'],
               'May':item.get('may_metrics',{}).get('mae_kwh')} for model,item in card['models'].items()}} for scenario,card in combined['scenarios'].items()},indent=2))

def draw_figures():
    combined=load_json(OUT/'comparison_results.json')
    os.environ['MPLCONFIGDIR']=str(OUT/'figures'/'.mpl_cache')
    sys.path.insert(0,str(ROOT/'output/analysis/.plot_deps'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'svg.fonttype':'none','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    figure_dir=OUT/'figures';figure_dir.mkdir(parents=True,exist_ok=True)
    for model in MODEL_IDS:
        links={};day=pd.read_csv(location('dayahead24',model)/'test_predictions.csv');day['id']=pd.to_datetime(day['id']);day=day.sort_values('id')
        stat=combined['scenarios']['dayahead24']['models'][model]
        fig,axes=plt.subplots(3,1,figsize=(12,8.5),sharex=True,gridspec_kw={'height_ratios':[2,1,1]},layout='constrained')
        axes[0].plot(day.id,day.actual_kwh,label='실측',color='#0f6b78',lw=1.1);axes[0].plot(day.id,day.predicted_kwh,label='각 일자 origin에서 생성한24시간 예측',color='#e8782a',lw=1.4)
        axes[0].set_title(f'{stat["display_name"]} | April day-ahead:29 daily origins,696 hourly values')
        axes[0].set_ylabel('잠정 모듈합 kWh');axes[0].legend(loc='upper right',fontsize=9)
        axes[1].plot(day.id,day.predicted_kwh-day.actual_kwh,color='#5b5aa8',lw=.8);axes[1].axhline(0,color='#475569',lw=.7);axes[1].set_ylabel('예측-실측 kWh')
        endpoints=day[day.lead_hours==24]
        axes[2].plot(endpoints.id,endpoints.actual_kwh,'o-',ms=3,color='#0f6b78',label='24시간뒤 실측');axes[2].plot(endpoints.id,endpoints.predicted_kwh,'o-',ms=3,color='#e8782a',label='24시간뒤 예측')
        axes[2].set_ylabel('lead24 kWh');axes[2].legend(loc='upper right',fontsize=9)
        axes[2].set_title(f'24곡선 MAE {stat["test_metrics"]["mae_kwh"]:.3f} | lead24 endpoint MAE {stat["test_endpoint_metrics"]["mae_kwh"]:.3f}',fontsize=10)
        for ax in axes:ax.grid(alpha=.18)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        for extension in ['png','svg']:
            file=figure_dir/f'{model}_april_day.{extension}';fig.savefig(file,dpi=180);links[f'april_day_{extension}']=str(file)
        plt.close(fig)
        long=pd.read_csv(location('long745',model)/'future_path.csv');long['id']=pd.to_datetime(long['id']);may=pd.read_csv(location('long745',model)/'may_predictions.csv');may['id']=pd.to_datetime(may['id'])
        stat=combined['scenarios']['long745']['models'][model]
        fig,axes=plt.subplots(2,1,figsize=(12,6.5),sharex=True,gridspec_kw={'height_ratios':[2,1]},layout='constrained')
        axes[0].plot(long.id,long.predicted_kwh,label='Apr30단일 origin의745시간 예측',color='#e8782a',lw=1.5)
        axes[0].plot(may.id,may.actual_kwh,label='May10~31 제공실측505시간',color='#0f6b78',lw=1.)
        axes[0].axvspan(long.id.min(),may.id.min(),color='#aab6c4',alpha=.14,label='비점수bridge:관측값으로채우지않음')
        axes[0].set_title(f'{stat["display_name"]} | Separate long task:fixed P745,scored May P505')
        axes[0].set_ylabel('잠정 모듈합 kWh');axes[0].legend(loc='upper right',fontsize=9)
        axes[1].plot(may.id,may.predicted_kwh-may.actual_kwh,color='#5b5aa8',lw=.9);axes[1].axhline(0,color='#475569',lw=.7);axes[1].set_ylabel('예측-실측 kWh')
        axes[1].set_title(f'MAE {stat["may_metrics"]["mae_kwh"]:.3f} | RMSE {stat["may_metrics"]["rmse_kwh"]:.3f} | MAPE {stat["may_metrics"]["mape_percent"]:.4f}%',fontsize=10)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        for ax in axes:ax.grid(alpha=.18)
        for extension in ['png','svg']:
            file=figure_dir/f'{model}_may_long.{extension}';fig.savefig(file,dpi=180);links[f'may_long_{extension}']=str(file)
        plt.close(fig)
        combined['figures'][model]=links
    write_json(OUT/'comparison_results.json',combined)
    print('Saved24model figures:12PNG+12SVG')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['lock','score','figures'],required=True);args=p.parse_args()
    {'lock':freeze_selection,'score':score_may,'figures':draw_figures}[args.stage]()
