"""Freeze all CV choices, export dashboard contracts, then one-shot May score."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import argparse
import numpy as np,pandas as pd
from common import *
LABEL={'baseline':'과거 수준 기준선','arima':'정규화 ARIMA','ridge':'정규화 Ridge','xgboost':'정규화 XGBoost',
       'patchtst':'재학습 PatchTST','itransformer':'재학습 iTransformer','chronos2':'고정 Chronos-2'}

def error_shape(f):
    d=f[f.horizon_offset_hours>=240];a=d.actual_kwh.to_numpy();p=d.predicted_kwh.to_numpy()
    return {'actual_std_kwh':float(a.std(ddof=1)),'prediction_std_kwh':float(p.std(ddof=1)),
            'pearson_actual_prediction':float(np.corrcoef(a,p)[0,1]) if p.std()>1e-8 else None,
            'prediction_mean_kwh':float(p.mean()),'actual_mean_kwh':float(a.mean()),
            'interpretation':'Amplitude/direction are measured in native identical units; a small prediction amplitude is not itself proof of unit-conversion error.'}
def model_hashes(models):
    out={}
    for model,r in models.items():
        for split,fit in r['fits'].items():
            for key in ['saved_model','prediction_csv']:
                path=fit[key];out[path]=sha(path)
        out[r['selection_lock']]=sha(r['selection_lock'])
    return out
def dashboard(r,scope='internal_test'):
    models=[];series=None
    for name,result in r['models'].items():
        metric=result['may'] if scope=='may' else result['internal_test']['late505']
        models.append({'id':name,'label':LABEL[name],'name':LABEL[name],'metrics':metric,'mae':metric['mae_kwh'],
                      'rmse':metric['rmse_kwh'],'r2':metric['r2'],'validation_mae':result['validation']['mean_origin_late505_mae_kwh'],
                      'selected':name==r['selected_model'],'config':result['selected_config']})
        f=pd.read_csv(result['fits']['future_unlabelled' if scope=='may' else 'internal_test']['prediction_csv'],float_precision='round_trip')
        f=f[f.horizon_offset_hours>=240].copy()
        if scope=='may':a=pd.read_csv(OUT/'may_actual_505.csv',float_precision='round_trip');actual=a.energy_kwh.to_numpy()
        else:actual=f.actual_kwh.to_numpy()
        if series is None:series=pd.DataFrame({'timestamp':f.id.to_numpy(),'actual_kwh':actual})
        assert np.array_equal(series.timestamp.to_numpy(),f.id.to_numpy());series[name]=f.predicted_kwh.to_numpy()
    payload={'models':models,'selected_model':r['selected_model'],'series':series.to_dict('records'),
      'horizon_note':'4/30 고정 원점 · 745시간 경로 · 5/10~5/31 00시 정답505시간 평가(앞240시간 정답 미제공)' if scope=='may' else '3/29 고정 원점 · 마지막505시간 내부시험(4/8~4/29 00시)',
      'scale_audit':r['scale_audit'],'quality_note':r['quality_note'],'notes':r['interpretation_limits'],
      'evaluation_scope':scope,'selection_uses_May':False,'model_reselected_after_test_or_May':False}
    write(OUT/'dashboard_evaluation.json',payload);write(OUT/'evaluation_payload.json',payload)
    selected=next(m for m in models if m['selected'])
    write(OUT/'dashboard_summary.json',{'selected_model':r['selected_model'],'model_name':selected['label'],'metrics':selected['metrics'],
           'validation_mae_kwh':selected['validation_mae'],'quality_note':r['quality_note'],'horizon_note':payload['horizon_note'],
           'operational_forecast_model':'historical_prefix_mean_baseline','replay_uses_April_checkpoint':False})

def freeze():
    if (OUT/'results.json').exists() and read(OUT/'results.json').get('May_read'):
        raise RuntimeError('Already May-scored. Cannot reset selection or enable another scoring/tuning pass.')
    assert read(OUT/'verification.json')['status']=='pass'
    models=read(OUT/'classical_results.json')['models']
    for family in ['patchtst','itransformer','chronos2']:models[family]=read(OUT/family/'results.json')
    chosen=min(models,key=lambda k:models[k]['validation']['mean_origin_late505_mae_kwh'])
    baseline=models['baseline']['validation']['mean_origin_late505_mae_kwh'];winner=models[chosen]['validation']['mean_origin_late505_mae_kwh']
    hashes=model_hashes(models)
    lock={'status':'frozen_before_May','selected_model':chosen,'selection_objective':'Mean three CV origins Late505 MAE only',
          'validation_origins':VALID,'latest_validation_label':'2025-03-28 00:00:00','internal_test_origin':str(TEST),
          'candidate_count':196,'model_and_prediction_hashes':hashes,'May_labels_read':False,
          'unit':'Native provisional guide kWh, no divide288/1000 target correction','model_reselection_after_May_forbidden':True}
    write(OUT/'global_selection_lock.json',lock)
    allframes=[];rows=[]
    for name,result in models.items():
        fs=[pd.read_csv(result['validation_prediction_csv'],float_precision='round_trip')]
        fs.extend(pd.read_csv(result['fits'][split]['prediction_csv'],float_precision='round_trip') for split in ['internal_test','future_unlabelled'])
        for f in fs:f['model']=name
        allframes.extend(fs);test=fs[1];result['native_error_shape_internal_test']=error_shape(test)
        rows.append({'model':name,'display_name':LABEL[name],'selected':name==chosen,'CV505_MAE_kwh':result['validation']['mean_origin_late505_mae_kwh'],
                'internal505_MAE_kwh':result['internal_test']['late505']['mae_kwh'],'internal505_RMSE_kwh':result['internal_test']['late505']['rmse_kwh'],
                'internal505_R2':result['internal_test']['late505']['r2'],'internal505_bias_pred_minus_actual_kwh':result['internal_test']['late505']['bias_pred_minus_actual_kwh']})
    pd.concat(allframes,ignore_index=True).to_csv(OUT/'all_predictions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(rows).to_csv(OUT/'comparison_results.csv',index=False,encoding='utf-8-sig')
    audit=read(OUT/'normalization_audit.json')
    r={'status':'selection_frozen_pending_May_evaluation','models':models,'selected_model':chosen,'candidate_count':196,
      'validation_origins':VALID,'selection':'Three origin Late505 MAE only','May_read':False,
      'baseline_vs_selected_validation':{'baseline_MAE_kwh':baseline,'selected_MAE_kwh':winner,'improvement_kwh':baseline-winner,
               'improvement_percent':float((baseline-winner)/baseline*100),'interpretation':'Best among tested196 configurations; tiny advantage is not proof of reliable future superiority'},
      'scale_audit':audit,'quality_note':'검증상 차이는 매우 작으며 변동 예측력이 확인되었다고 단정할 수 없습니다. 잠정 단위와 계측 계층 불확실성을 유지합니다.',
      'global_selection_lock':str(OUT/'global_selection_lock.json'),'native_unit':'provisional guide kWh',
      'serving_API':{'file':str(OUT/'api.py'),'function':'serving_forecast(series,origin,points=24,module_id=None)',
                   'model':'independently fitted historical prefix mean baseline','quality':'Recent past replay metrics only, no calibrated future interval'},
      'interpretation_limits':['Original guide physical scaling and meter hierarchy remain unresolved; no label-unit correction performed',
       'R2 is invariant when truth and prediction are both converted to the same unit; lowMAPE reflects high mean relative to variance',
       'Three CV paths overlap (1515 scoring rows,745 unique target hours); only one later internal origin and one May origin exist',
       'April/May were exposed in earlier EDA; this is revised historical evaluation, not a pristine blind prospective test',
       'Neural training stops at LRfloor and validation plateau, saving actual best checkpoints; moreepochs do not imply more skill',
       'Final epoch/schedule and XGB iteration parameters are frozen fromCV; no April epoch/model selection',
       'Online March/December replay uses only its own historical prefix and never April-trained weights',
       'No future current/voltage/fault-cleaning statistics or realized daily/monthly totals enter forecasting features',
       'No validated future prediction interval is supplied; point forecasts and empirical replay errors remain separate'],
      'paths':{'comparison_csv':str(OUT/'comparison_results.csv'),'all_predictions_csv':str(OUT/'all_predictions.csv'),
               'normalization_audit':str(OUT/'normalization_audit.json'),'verification':str(OUT/'verification.json'),
               'dashboard_evaluation':str(OUT/'dashboard_evaluation.json'),'dashboard_summary':str(OUT/'dashboard_summary.json')}}
    write(OUT/'results.json',r);dashboard(r);print({'frozen':chosen,'CVMAE':winner,'baseline':baseline,'improvement':baseline-winner})
    return r

def conditional_block_ci(delta,index):
    # Conditional uncertainty for this one origin path; not independent months.
    groups=pd.Series(np.asarray(delta),index=pd.DatetimeIndex(index)).groupby(pd.DatetimeIndex(index).normalize())
    sums=np.array([d.sum() for _,d in groups]);counts=np.array([len(d) for _,d in groups]);rng=np.random.default_rng(SEED)
    draw=rng.integers(0,len(sums),size=(10000,len(sums)));b=sums[draw].sum(1)/counts[draw].sum(1)
    return {'baseline_minus_model_mean_abs_error_kwh':float(np.asarray(delta).mean()),'conditional95CI_kwh':np.quantile(b,[.025,.975]).tolist(),
       'bootstrap_replicates':10000,'seed':SEED,'daily_block_count':len(sums),'block_hours':counts.tolist(),
       'direction':'Positive means model improves MAE over baseline','limit':'Conditional on one path; not independent-origin/month generalization'}

def evaluate_may(actual,source_path=None):
    r=read(OUT/'results.json');lock=read(OUT/'global_selection_lock.json')
    if r.get('May_read'):raise RuntimeError('May has already been scored. No second scoring/tuning pass is permitted.')
    assert lock['status']=='frozen_before_May' and r['selected_model']==lock['selected_model']
    for path,expected in lock['model_and_prediction_hashes'].items():assert sha(path)==expected
    idx=pd.date_range('2025-05-10','2025-05-31',freq='h');actual=actual.sort_index()
    assert len(actual)==505 and actual.index.equals(idx) and np.isfinite(actual.to_numpy()).all()
    pd.DataFrame({'timestamp':actual.index,'energy_kwh':actual.to_numpy()}).to_csv(OUT/'may_actual_505.csv',index=False,encoding='utf-8-sig')
    predictions={};rows=[];baseline=None
    for name,result in r['models'].items():
        f=pd.read_csv(result['fits']['future_unlabelled']['prediction_csv'],float_precision='round_trip');d=f[f.horizon_offset_hours>=240].copy()
        assert pd.DatetimeIndex(pd.to_datetime(d.id)).equals(idx)
        p=d.predicted_kwh.to_numpy();predictions[name]=p;metric=metrics(actual.to_numpy(),p);result['may']=metric
        rows.append({'model':name,'display_name':LABEL[name],'selected_before_May':name==r['selected_model'],**metric})
        if name=='baseline':baseline=p
    observed_best=min(r['models'],key=lambda name:r['models'][name]['may']['mae_kwh'])
    ci={name:conditional_block_ci(abs(baseline-actual.to_numpy())-abs(p-actual.to_numpy()),idx) for name,p in predictions.items() if name!='baseline'}
    r.update({'status':'complete','May_read':True,'May_scoring_passes':1,'observed_May_best_model':observed_best,
               'May_best_is_posthoc_only':True,'selected_model_changed_after_May':False,'May_conditional_paired_error_CIs':ci,
               'May_source':{'path':str(source_path) if source_path else 'Caller supplied native-energy505 Series','sha256':sha(source_path) if source_path else None,
                      'first':str(idx[0]),'last':str(idx[-1]),'count':505,'unit_assumption':'May labels interpreted as guide-consistent kWh; provisional, no rescaling'}})
    pd.DataFrame(rows).to_csv(OUT/'may_comparison.csv',index=False,encoding='utf-8-sig')
    labelled=pd.DataFrame({'timestamp':idx,'actual_kwh':actual.to_numpy(),**predictions});labelled.to_csv(OUT/'may_model_predictions.csv',index=False,encoding='utf-8-sig')
    write(OUT/'results.json',r);dashboard(r,'may');plot_may(r,labelled)
    print({'fixed_selected':r['selected_model'],'selected_MayMAE':r['models'][r['selected_model']]['may']['mae_kwh'],
           'posthoc_best_May_only':observed_best,'no_reselection':True})
    return r

def plot_may(r,f):
    import os
    sys.path.insert(0,str(ROOT/'output/analysis/.plot_deps'));os.environ['MPLCONFIGDIR']=str(OUT/'tmp/mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt,matplotlib.dates as mdates
    from matplotlib.font_manager import fontManager
    from matplotlib.ticker import FuncFormatter
    fontManager.addfont('C:/Windows/Fonts/malgun.ttf');plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'svg.fonttype':'none'})
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True);plots={};dates=pd.to_datetime(f.timestamp)
    for name in r['models']:
        fig,axs=plt.subplots(2,1,figsize=(13,6.8),sharex=True,gridspec_kw={'height_ratios':[2,1]})
        axs[0].plot(dates,f.actual_kwh,color='#758596',lw=.8,label='관측');axs[0].plot(dates,f[name],color='#246b91',lw=1.4,label='검증 후 고정된 예측')
        axs[0].set_title(LABEL[name]+' | 4/30 원점 May505 평가 (축 확대)');axs[0].set_ylabel('잠정 kWh');axs[0].legend(ncol=2)
        axs[0].yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:,.0f}'))
        axs[1].plot(dates,f[name]-f.actual_kwh,color='#246b91',lw=.7);axs[1].axhline(0,color='gray',lw=.8);axs[1].set_ylabel('예측 - 관측(kWh)')
        axs[1].xaxis.set_major_locator(mdates.DayLocator(interval=3));axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        for ax in axs:ax.grid(alpha=.2);ax.set_xlim(dates.iloc[0],dates.iloc[-1])
        metric=r['models'][name]['may'];fig.text(.5,.015,f'MAE {metric["mae_kwh"]:.4f} · R² {metric["r2"]:.4f} · May 정답은 최종1회 평가에만 사용, 재선택·편향 보정 없음',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.05,1,1));paths={}
        for ext in ['png','svg']:
            path=figdir/f'may_{name}.{ext}';fig.savefig(path,dpi=180,bbox_inches='tight');paths[ext]=str(path)
        plt.close(fig);plots[name]=paths
    write(OUT/'plot_manifest.json',{'May_model_specific_plots':plots,'selected_model':r['selected_model'],'May_used_for_tuning':False})

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--may-csv');a=p.parse_args()
    if a.freeze:freeze()
    if a.may_csv:
        f=pd.read_csv(a.may_csv,float_precision='round_trip')
        if not {'timestamp','energy_kwh'}.issubset(f.columns):raise ValueError('CSV must contain timestamp,energy_kwh in native guide provisional units')
        s=pd.Series(f.energy_kwh.to_numpy(),index=pd.to_datetime(f.timestamp));evaluate_may(s,a.may_csv)
if __name__=='__main__':main()
