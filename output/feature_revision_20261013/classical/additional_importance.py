"""Train-ranked feature diagnostics and final coefficient exports, not HPO."""
import sys,os,json
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import joblib
from retrain_classical import (OUT,ROOT,load_hourly,prefix,select_sensor_features,fit_model,predict,features_at,
                              idx_future,metrics,write_json,TARGET,SEED,sha)

def main():
    result=json.loads((OUT/'results.json').read_text(encoding='utf-8'));f=load_hourly();before=sha(OUT/'selection_lock.json')
    os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig');sys.path.insert(0,str(ROOT/'output/analysis/.plot_deps'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name(),'axes.unicode_minus':False,'svg.fonttype':'none'})
    for family,m in result['models'].items():
        dest=OUT/family;bundle=joblib.load(dest/'saved_model/model.joblib')
        if family in ['ridge','elasticnet']:
            schema=bundle['feature_schema'];coef=bundle['object'][-1].coef_;scale=bundle['object'][0].scale_
            table=pd.DataFrame({'group':[g for g,n in schema],'feature':[n for g,n in schema],
                                'coefficient_per_training_std_kwh':coef,'coefficient_per_raw_unit_kwh':coef/scale})
            table.to_csv(dest/'coefficients_final.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
            m['final_coefficients_csv']=str(dest/'coefficients_final.csv')
        elif family=='xgboost':
            schema=bundle['feature_schema'];table=pd.DataFrame({'group':[g for g,n in schema],'feature':[n for g,n in schema],
                                                              'normalized_training_gain':bundle['object'].feature_importances_})
            table.to_csv(dest/'gain_final.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
            m['training_gain_csv']=str(dest/'gain_final.csv')
        elif family=='arimax':
            names=bundle['object'].model.param_names;params=np.asarray(bundle['object'].params,dtype=float)
            table=pd.DataFrame({'parameter':names,'coefficient':params})
            table.to_csv(dest/'coefficients_final.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
            m['final_coefficients_csv']=str(dest/'coefficients_final.csv')
        if family in ['xgboost','ridge','elasticnet']:
            origin=pd.Timestamp('2025-03-25');h=prefix(f,origin);selected,audit=select_sensor_features(h,max_features=12)
            diagnostic=fit_model(h,24,m['selected_config'],selected);xs=[];ys=[]
            for o in pd.date_range(origin,periods=7,freq='D'):
                past=prefix(f,o);xs.append(features_at(past,len(past),np.arange(24),m['selected_config']['group'],selected))
                ys.append(f.loc[idx_future(past,24),TARGET].to_numpy(dtype=float))
            X=np.concatenate(xs);y=np.concatenate(ys);base=metrics(y,predict(diagnostic,h,len(y),override_x=X))['mae_kwh']
            score=diagnostic['object'].feature_importances_ if family=='xgboost' else np.abs(diagnostic['object'][-1].coef_)
            candidates=np.argsort(score)[::-1][:20];schema=diagnostic['feature_schema'];rng=np.random.default_rng(SEED);rows=[]
            for c in candidates:
                diffs=[]
                for repeat in range(5):
                    altered=X.copy();altered[:,c]=X[:,c].reshape(7,24)[rng.permutation(7)].reshape(-1)
                    diffs.append(metrics(y,predict(diagnostic,h,len(y),override_x=altered))['mae_kwh']-base)
                rows.append({'feature':schema[c][1],'group':schema[c][0],'training_only_ranking_score':float(score[c]),
                             'baseline_inner_mae_kwh':base,'permuted_delta_mae_kwh_mean':float(np.mean(diffs)),
                             'permuted_delta_mae_kwh_std':float(np.std(diffs,ddof=1)),'repeats':5})
            table=pd.DataFrame(rows);table.to_csv(dest/'chronological_feature_permutation.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
            fig,ax=plt.subplots(figsize=(11,7),layout='constrained');ordered=table.sort_values('permuted_delta_mae_kwh_mean')
            ax.barh(ordered.feature,ordered.permuted_delta_mae_kwh_mean,color='#0d9488');ax.axvline(0,color='black',lw=.8)
            ax.set_xlabel('March chronological inner-holdout MAE change (kWh)');ax.set_title(f'{family}: training-ranked top20,5 day-block permutations')
            for ext in ['png','svg']:fig.savefig(dest/'figures'/f'feature_permutation.{ext}',dpi=150)
            plt.close(fig)
            m['chronological_feature_permutation_csv']=str(dest/'chronological_feature_permutation.csv')
            m['plots'].update(feature_permutation_png=str(dest/'figures/feature_permutation.png'),feature_permutation_svg=str(dest/'figures/feature_permutation.svg'))
            result['figures'][family].update(m['plots'])
            m['importance_note']='Top20 chosen only by training gain/absolute standardized coefficient; seven later March days diagnose permutation reliance. No HPO/config changes. Weak/negative deltas do not establish causal sensor effects.'
        write_json(dest/'results.json',m)
    assert before==sha(OUT/'selection_lock.json')
    write_json(OUT/'results.json',result)
    print('Final coefficients and chronological feature-level diagnostics exported; selection and predictions unchanged.')
if __name__=='__main__':main()
