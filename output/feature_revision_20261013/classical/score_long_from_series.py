"""Score already-frozen long paths with a Series supplied by the parent scorer.

This helper never opens the May source file. The parent can read it once for all
model families, then pass exactly the same505 values after the global lock.
"""
from pathlib import Path
import sys,os,json
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from retrain_classical import OUT,ROOT,metrics,write_json,sha

def score_long_from_series(may_target,global_lock_path):
    global_lock_path=Path(global_lock_path)
    if not global_lock_path.is_file():raise ValueError('Parent global selection lock is required')
    text=global_lock_path.read_text(encoding='utf-8')
    for local in [OUT/'selection_lock.json',OUT/'long745/selection_lock.json']:
        if sha(local) not in text:raise ValueError('Global lock must contain current classical day and long selection-lock hashes')
    expected=pd.date_range('2025-05-10','2025-05-31',freq='h')
    may=pd.Series(np.asarray(may_target,dtype=np.float64),index=pd.DatetimeIndex(may_target.index))
    if not may.index.equals(expected) or not np.isfinite(may).all():raise ValueError('May scorer must supply identical505 hourly labels')
    base=OUT/'long745';result=json.loads((base/'results.json').read_text(encoding='utf-8'))
    if result.get('May_score_status')=='scored_after_parent_global_lock':raise ValueError('Secondary May paths were already scored; do not tune/rescore')
    os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig');sys.path.insert(0,str(ROOT/'output/analysis/.plot_deps'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name(),'axes.unicode_minus':False,'svg.fonttype':'none'})
    for name,m in result['models'].items():
        f=pd.read_csv(m['paths']['future_csv'],parse_dates=['id'],float_precision='round_trip')
        assert len(f)==745 and f.actual_kwh.isna().all()
        f['actual_kwh']=may.reindex(f.id).to_numpy(dtype=np.float64)
        f['absolute_error_kwh']=np.abs(f.predicted_kwh-f.actual_kwh).astype(np.float64)
        scored=f.iloc[240:].copy();assert len(scored)==505 and scored.actual_kwh.notna().all()
        m['may_metrics']=metrics(scored.actual_kwh,scored.predicted_kwh)
        dest=base/name
        f.to_csv(dest/'full_forecast_with_may_actuals.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
        scored.to_csv(dest/'may_predictions.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
        m['paths']['may_csv']=str(dest/'may_predictions.csv')
        fig,(ax,res)=plt.subplots(2,1,figsize=(14,6),sharex=True,gridspec_kw={'height_ratios':[2,1]},layout='constrained')
        ax.axvspan(f.id.iloc[0],f.id.iloc[239],color='#e2e8f0',alpha=.6,label='No public actual labels')
        ax.plot(f.id,f.actual_kwh,color='#475569',lw=.9,label='Observed May provisional sum')
        ax.plot(f.id,f.predicted_kwh,color='#f97316',lw=1.,label='Frozen Apr30-origin prediction')
        ax.set_ylabel('Provisional kWh');ax.set_title(f'{name}: P745 context-only forecast, May505 MAE {m["may_metrics"]["mae_kwh"]:.3f}kWh')
        ax.legend();ax.grid(alpha=.2)
        res.plot(f.id,f.predicted_kwh-f.actual_kwh,color='#2563eb',lw=.8);res.axhline(0,color='black',lw=.6)
        res.set_ylabel('Pred - observed (kWh)');res.grid(alpha=.2)
        figs=dest/'figures';figs.mkdir(parents=True,exist_ok=True)
        for ext in ['png','svg']:fig.savefig(figs/f'may_long.{ext}',dpi=150)
        plt.close(fig)
        m['plots']={'may_long_png':str(figs/'may_long.png'),'may_long_svg':str(figs/'may_long.svg')}
        write_json(dest/'results.json',m)
    result.update(May_score_status='scored_after_parent_global_lock',global_selection_lock=str(global_lock_path),
                  global_selection_lock_sha256=sha(global_lock_path),May_labels_read=True,
                  May_labels_received_from_parent=True,May_source_file_opened_by_helper=False,
                  May_selection_changed=False,May_input_contract='Parent supplied505 float64 values from one shared read; all future paths were saved before this call.')
    write_json(base/'results.json',result)
    day=json.loads((OUT/'results.json').read_text(encoding='utf-8'));day['secondary_long745']=result;write_json(OUT/'results.json',day)
    return result
