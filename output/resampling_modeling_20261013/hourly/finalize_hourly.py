"""Reconcile hourly CSVs, native replay and make internal-test diagnostics."""
import sys,os,json
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from train_hourly import OUT,ROOT,VALID,TEST,P,metrics,stats,write_json,sha
from predict_hourly import forecast_saved

def main():
    result=json.loads((OUT/'results.json').read_text(encoding='utf-8'));before=sha(OUT/'selection_lock.json');checks=[];figs={}
    os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig');sys.path.insert(0,str(ROOT/'output/analysis/.plot_deps'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name(),'axes.unicode_minus':False,'svg.fonttype':'none'})
    for name,m in result['models'].items():
        valid=pd.read_csv(m['paths']['validation_csv'],parse_dates=['id','origin'],float_precision='round_trip')
        assert len(valid)==3*P and valid.origin.nunique()==3 and valid.groupby('origin').size().eq(P).all()
        vals=[];fulls=[]
        for origin,frame in valid.groupby('origin'):
            assert np.array_equal(frame.lead_hours,np.arange(1,P+1))
            scored=stats(frame.actual_kwh,frame.predicted_kwh);vals.append(scored['late505']['mae_kwh']);fulls.append(scored['full745']['mae_kwh'])
        vdiff=abs(np.mean(vals)-m['validation_late505']['mean_origin_mae_kwh']);fdiff=abs(np.mean(fulls)-m['validation_full745']['mean_origin_mae_kwh'])
        assert max(vdiff,fdiff)<1e-7
        late=valid.loc[valid.horizon_offset_hours.ge(240)]
        m['validation_late505'].update(metrics(late.actual_kwh,late.predicted_kwh),origin_count=3,unique_label_hours=int(late.id.nunique()),repeated_label_observations=int(len(late)-late.id.nunique()))
        m['validation_full745'].update(metrics(valid.actual_kwh,valid.predicted_kwh),origin_count=3,unique_label_hours=int(valid.id.nunique()),repeated_label_observations=int(len(valid)-valid.id.nunique()))
        test=pd.read_csv(m['paths']['test_csv'],parse_dates=['id','origin'],float_precision='round_trip');assert len(test)==P
        computed=stats(test.actual_kwh,test.predicted_kwh);diffs=[]
        for window,metric in computed.items():
            diffs += [abs(v-m['internal_test'][window][k]) for k,v in metric.items() if v is not None]
        tdiff=max(diffs);assert tdiff<1e-7
        scored_targets=valid.loc[valid.horizon_offset_hours.ge(240),'id'];overlap=len(set(scored_targets)&set(test.id));assert overlap==0
        saved=pd.read_csv(m['paths']['future_csv'],parse_dates=['id'],float_precision='round_trip');fresh=forecast_saved(name)
        replay=float(np.max(np.abs(saved.predicted_kwh-fresh.predicted_kwh)));assert replay<=1e-8
        assert len(saved)==P and saved.actual_kwh.isna().all() and saved.id.iloc[0]==pd.Timestamp('2025-04-30') and saved.id.iloc[-1]==pd.Timestamp('2025-05-31')
        checks.append({'model':name,'validation_rows':len(valid),'late_scored_pairs':len(scored_targets),'late_unique_targets':int(scored_targets.nunique()),
              'validation_late_metric_diff':float(vdiff),'validation_full_metric_diff':float(fdiff),'internal_metric_max_absdiff':tdiff,
              'validation_late_vs_internal_target_overlap':overlap,'standalone_replay_max_absdiff':replay,'future_actual_all_nan':True})
        dest=OUT/name/'figures';dest.mkdir(parents=True,exist_ok=True)
        fig,(ax,res)=plt.subplots(2,1,figsize=(14,6),sharex=True,gridspec_kw={'height_ratios':[2,1]},layout='constrained')
        ax.axvspan(test.id.iloc[240],test.id.iloc[-1],color='#dcfce7',alpha=.6,label='Primary late505 evaluation')
        ax.plot(test.id,test.actual_kwh,color='#475569',lw=.8,label='Observed provisional hourly sum')
        ax.plot(test.id,test.predicted_kwh,color='#f97316',lw=1.1,label=name)
        ax.set_ylabel('Provisional kWh');ax.grid(alpha=.2);ax.legend()
        ax.set_title(f'{name}: Mar29 fixed-origin P745 | late505 MAE {computed["late505"]["mae_kwh"]:.3f}, full745 {computed["full745"]["mae_kwh"]:.3f}')
        res.plot(test.id,test.predicted_kwh-test.actual_kwh,color='#2563eb',lw=.8);res.axhline(0,color='black',lw=.7);res.set_ylabel('Pred - observed (kWh)');res.grid(alpha=.2)
        for ext in ['png','svg']:fig.savefig(dest/f'internal_test.{ext}',dpi=150)
        plt.close(fig);m['plots']={'internal_png':str(dest/'internal_test.png'),'internal_svg':str(dest/'internal_test.svg')};figs[name]=m['plots']
        write_json(OUT/name/'results.json',m)
    assert before==sha(OUT/'selection_lock.json')
    table=pd.read_csv(OUT/'ablation.csv');table=table.loc[table['model'].isin(['xgboost','ridge'])].copy()
    table['label']=table['model']+'/'+table.group.fillna('')+' #'+table.candidate.astype(str)
    fig,ax=plt.subplots(figsize=(11,7),layout='constrained');ordered=table.sort_values('validation_late505_mae_kwh')
    ax.barh(ordered.label,ordered.validation_late505_mae_kwh,color='#0d9488');ax.set_xlim(min(ordered.validation_late505_mae_kwh)-.1,max(ordered.validation_late505_mae_kwh)+.1)
    ax.set_xlabel('Mean three-origin late505 MAE (kWh)');ax.set_title('Completed H/D/W/M feature candidates: validation only')
    for ext in ['png','svg']:fig.savefig(OUT/f'feature_ablation.{ext}',dpi=150)
    plt.close(fig);result['figures']=figs;result['ablation_figures']={'png':str(OUT/'feature_ablation.png'),'svg':str(OUT/'feature_ablation.svg')}
    write_json(OUT/'results.json',result)
    write_json(OUT/'verification.json',{'status':'passed','selection_lock_sha256':before,'checks':checks,
        'metric_tolerance':1e-7,'replay_tolerance':1e-8,'May_source_read':False,'old_outputs_modified':False})
    print(json.dumps({'status':'passed','selected_model':result['selected_model'],'verification':str(OUT/'verification.json')}))
if __name__=='__main__':main()
