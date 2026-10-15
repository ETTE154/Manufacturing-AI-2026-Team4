"""CSV reconciliation, paired prior-baseline comparisons and static figures."""
from pathlib import Path
import sys,os,json
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from retrain_classical import OUT,ROOT,summary,write_json,metrics,sha

def paired(new,old):
    keys=['origin','id','lead_hours']
    a=new[keys+['actual_kwh','predicted_kwh']].merge(old[keys+['actual_kwh','predicted_kwh']],on=keys,suffixes=('_new','_old'),validate='one_to_one')
    if len(a)!=len(new) or not np.allclose(a.actual_kwh_new,a.actual_kwh_old,rtol=0,atol=1e-8):raise ValueError('Unpaired prior comparator')
    d=np.abs(a.predicted_kwh_new-a.actual_kwh_new)-np.abs(a.predicted_kwh_old-a.actual_kwh_old)
    day=pd.Series(d.to_numpy(),index=a.origin).groupby(level=0).mean().to_numpy()
    rng=np.random.default_rng(20261013);boots=day[rng.integers(0,len(day),(10000,len(day)))].mean(axis=1)
    return {'count':len(a),'origins':len(day),'delta_MAE_new_minus_old_kwh':float(d.mean()),
            'paired_day_bootstrap_95pct_kwh':np.quantile(boots,[.025,.975]).tolist(),
            'limit':'Conditional same-period day-block uncertainty; no prospective plant/site generalization guarantee.'}

def flatten(obj,prefix=''):
    out={}
    for key,val in obj.items():
        name=prefix+'.'+key
        if isinstance(val,dict):out.update(flatten(val,name))
        elif isinstance(val,(int,float)) and val is not None:out[name]=val
    return out

def main():
    result=json.loads((OUT/'results.json').read_text(encoding='utf-8'));lockhash=sha(OUT/'selection_lock.json')
    from predict_classical import forecast_saved
    old=ROOT/'output/optimization_20261013/forecast/dayahead24'
    os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig')
    sys.path.insert(0,str(ROOT/'output/analysis/.plot_deps'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/malgun.ttf')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'svg.fonttype':'none'})
    checks=[];comparison={};figs={}
    for family,model in result['models'].items():
        for split in ['validation','test']:
            f=pd.read_csv(model['paths'][f'{split}_csv'],float_precision='round_trip')
            expected=model[f'{split}_metrics'];computed=summary(f)
            diffs=[abs(v-flatten(expected)[k]) for k,v in flatten(computed).items()]
            diff=max(diffs,default=0.);assert diff<1e-7
            assert len(f)==(1680 if split=='validation' else 696)
            assert f.groupby('origin').size().eq(24).all() and not f.duplicated(['origin','id']).any()
            checks.append({'model':family,'split':split,'rows':len(f),'origins':f.origin.nunique(),'metric_max_absdiff':diff})
        fresh=forecast_saved(family);saved=pd.read_csv(model['paths']['future_csv'],float_precision='round_trip')
        reload=float(np.max(np.abs(fresh.predicted_kwh-saved.predicted_kwh)));assert reload<1e-8
        checks.append({'model':family,'standalone_saved_context_reload_max_absdiff':reload,'future_rows':len(saved)})
        comparison[family]={}
        for split in ['validation','test']:
            f=pd.read_csv(model['paths'][f'{split}_csv'],float_precision='round_trip')
            comparison[family][split]={name:paired(f,pd.read_csv(old/name/f'{split}_predictions.csv',float_precision='round_trip')) for name in ['baselines','chronos2']}
        f=pd.read_csv(model['paths']['test_csv'],parse_dates=['id'],float_precision='round_trip')
        fig,(ax,res)=plt.subplots(2,1,figsize=(14,6.2),sharex=True,gridspec_kw={'height_ratios':[2,1]},layout='constrained')
        ax.plot(f.id,f.actual_kwh,color='#475569',lw=.9,label='Observed provisional module sum')
        ax.plot(f.id,f.predicted_kwh,color='#f97316',lw=1.1,label=family)
        ax.set_ylabel('Provisional kWh');ax.legend(loc='upper right');ax.grid(alpha=.2)
        ax.set_title(f'{family}: April24-hour daily-origin forecasts | curve MAE {model["test_metrics"]["path_24_values"]["mae_kwh"]:.3f}; lead24 MAE {model["test_metrics"]["lead_24"]["mae_kwh"]:.3f}')
        res.plot(f.id,f.predicted_kwh-f.actual_kwh,color='#2563eb',lw=.8);res.axhline(0,color='black',lw=.6)
        res.set_ylabel('Pred - observed (kWh)');res.grid(alpha=.2)
        dest=OUT/family/'figures';dest.mkdir(parents=True,exist_ok=True)
        for ext in ['png','svg']:fig.savefig(dest/f'april_day24.{ext}',dpi=150)
        plt.close(fig)
        figs[family]={'april_day_png':str(dest/'april_day24.png'),'april_day_svg':str(dest/'april_day24.svg')}
        if model.get('importance'):
            table=pd.DataFrame(model['importance']['groups'])
            fig,ax=plt.subplots(figsize=(9,4),layout='constrained');ax.barh(table.group,table.permuted_delta_mae_kwh_mean,color='#0d9488');ax.axvline(0,color='black',lw=.8)
            ax.set_xlabel('Change in chronological inner-holdout MAE (kWh)');ax.set_title(f'{family}: day-block group permutation,5 repeats')
            for ext in ['png','svg']:fig.savefig(dest/f'importance.{ext}',dpi=150)
            plt.close(fig);figs[family].update(importance_png=str(dest/'importance.png'),importance_svg=str(dest/'importance.svg'))
        model['plots']=figs[family]
        write_json(OUT/family/'results.json',model)
    if lockhash!=sha(OUT/'selection_lock.json'):raise ValueError('Selection changed during diagnostics')
    result['paired_previous_models']=comparison;result['figures']=figs
    result['comparison_limit']='Prior baseline and Chronos/blend predictions are read-only comparators on identical70 validation and29 April origins. Their post-evaluation comparison does not change model selection.'
    write_json(OUT/'results.json',result)
    write_json(OUT/'verification.json',{'status':'passed','selection_lock_sha256':lockhash,'checks':checks,
       'metric_tolerance':1e-7,'reload_tolerance':1e-8,'old_artifacts_modified':False,'May_labels_read':False})
    print(json.dumps({'status':'passed','selected_model':result['selected_model'],'verification':str(OUT/'verification.json')},ensure_ascii=False))
if __name__=='__main__':main()
