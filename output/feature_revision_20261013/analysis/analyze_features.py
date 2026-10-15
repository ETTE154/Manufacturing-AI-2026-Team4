"""Pre-April signal diagnostics; never selects using April/May labels."""
from pathlib import Path
import sys,os,json,time
sys.dont_write_bytecode=True
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'output/feature_revision_20261013';A=OUT/'analysis'
sys.path.insert(0,str(OUT))
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error,r2_score
from threadpoolctl import threadpool_limits
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from feature_common import load_hourly,load_catalog,feature_columns,correlation,bh_fdr,select_sensor_features,SEED,TARGET

def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,(Path,pd.Timestamp)):return str(v)
    if isinstance(v,np.bool_):return bool(v)
    return v

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def savecsv(frame,name):frame.to_csv(A/name,index=False,float_format='%.17g')

def met(y,p):return {'count':len(y),'mae_kwh':float(mean_absolute_error(y,p)),'r2':float(r2_score(y,p))}

def correlations(f):
    cols=[c for c in f if c!=TARGET];lags=[0,1,6,24,48,168];records=[]
    for method in ['pearson','spearman']:
        for lag in lags:
            for c in cols:
                x=f[c].shift(lag); y=f[TARGET]
                r,p,n=correlation(x,y,method)
                iidp=float(2*stats.t.sf(abs(r)*np.sqrt((n-2)/max(1-r*r,1e-15)),n-2)) if np.isfinite(r) else np.nan
                records.append({'feature':c,'method':method,'lag_hours':lag,'r':r,'p_iid':iidp,'p_AR1_approx':p,'n':n,
                                'forecast_available':lag>=1,'BH_family':method+('/diagnostic_same_hour' if lag==0 else '/past_lags')})
    result=pd.DataFrame(records)
    result['q_BH_AR1']=result.groupby('BH_family')['p_AR1_approx'].transform(lambda s:bh_fdr(s.to_numpy()))
    savecsv(result,'lag_correlations.csv')
    early=f[f.index<'2025-03-01'];late=f[f.index>='2025-03-01'];stable=[]
    for c in cols:
        for method in ['pearson','spearman']:
            re,_,ne=correlation(early[c].shift(1),early[TARGET],method)
            rl,_,nl=correlation(late[c].shift(1),late[TARGET],method)
            stable.append({'feature':c,'method':method,'lag_hours':1,'r_DecFeb':re,'r_March':rl,
                           'same_sign':np.sign(re)==np.sign(rl),'abs_change':abs(re-rl),
                           'minimum_abs_r':min(abs(re),abs(rl)),'n_DecFeb':ne,'n_March':nl})
    stability=pd.DataFrame(stable);savecsv(stability,'correlation_stability.csv')
    return result,stability

def prefix_selections(full):
    selections=[]
    for label,origin in [('fold1','2025-01-10'),('fold2','2025-01-24'),('fold3','2025-02-07'),
                         ('fold4','2025-02-21'),('fold5','2025-03-07'),('April_schema_lock','2025-04-01')]:
        selected,audit=select_sensor_features(full[full.index<pd.Timestamp(origin)])
        write(A/'selection_audits'/f'{label}.json',audit)
        selections.append({'fold':label,'origin':origin,'training_rows':audit['training_rows'],
                           'selected_count':len(selected),'selected_columns':selected,'selection_uses_April_or_May':False})
    write(A/'prefix_selection_results.json',selections)
    return selections

def shuffle_null(f,cols,repeats=200):
    # Align to full days after the largest lag. Target-day exchange is within
    # each calendar month; preserve each day's 24-value order and month levels.
    lags=[1,6,24,48,168];start=168;n=len(f)-start
    assert n%24==0
    x=np.concatenate([f[cols].iloc[start-lag:len(f)-lag].to_numpy() for lag in lags],axis=1)
    names=[(c,lag) for lag in lags for c in cols]
    z=(x-x.mean(axis=0))/x.std(axis=0,ddof=1)
    y=f[TARGET].iloc[start:].to_numpy();yz=(y-y.mean())/y.std(ddof=1)
    observed=z.T@yz/(n-1);best=int(np.argmax(np.abs(observed)))
    day_values=yz.reshape(-1,24);day_month=f.index[start::24].month.to_numpy()
    rng=np.random.default_rng(SEED);ym=np.empty((n,repeats))
    for b in range(repeats):
        order=np.arange(len(day_values))
        for month in np.unique(day_month):
            pos=np.flatnonzero(day_month==month);order[pos]=rng.permutation(pos)
        ym[:,b]=day_values[order].reshape(-1)
    null=np.max(np.abs(z.T@ym/(n-1)),axis=0)
    summary={'repeats':repeats,'feature_count':len(cols),'lag_hours':lags,'tested_pairs':len(names),
             'hours':n,'observed_max_abs_r':float(abs(observed[best])),
             'observed_max_feature':names[best][0],'observed_max_lag':names[best][1],
             'null_q05':float(np.quantile(null,.05)),'null_median':float(np.median(null)),
             'null_q95':float(np.quantile(null,.95)),'max_stat_empirical_p':float((1+(null>=abs(observed[best])).sum())/(repeats+1)),
             'method':'200 seeded target day-block permutations within calendar month, preserving24-hour order and the target marginal distribution; max over all past-feature/lag pairs controls search opportunity under this reference null',
             'limits':'Reference diagnostic, not proof of independence or exact familywise validity under arbitrary nonstationarity. April/May excluded; same-hour features excluded.',
             'null_max_abs_r':null.tolist()}
    write(A/'block_shuffle_null.json',summary)
    return summary

def time_structure(f):
    y=f[TARGET].to_numpy();av=acf(y,nlags=168,fft=True);pv=pacf(y,nlags=168,method='ywm')
    savecsv(pd.DataFrame({'lag_hours':np.arange(169),'acf':av,'pacf':pv}),'target_acf_pacf.csv')
    lj=acorr_ljungbox(y,lags=[1,24,168],return_df=True).reset_index(names='lag')
    calendar=[]
    for kind,keys in [('hour_of_day',f.index.hour),('weekday',f.index.dayofweek)]:
        for key in sorted(set(keys)):
            values=y[np.asarray(keys)==key]
            calendar.append({'group':kind,'value':int(key),'count':len(values),'mean_kwh':values.mean(),
                             'std_kwh':values.std(ddof=1),'deviation_from_global_kwh':values.mean()-y.mean()})
    cal=pd.DataFrame(calendar);savecsv(cal,'calendar_effects.csv')
    # Predeclared one-step chronological explanatory holdout, not HPO choice.
    design=pd.DataFrame({'hour_sin':np.sin(2*np.pi*f.index.hour/24),'hour_cos':np.cos(2*np.pi*f.index.hour/24),
                         'week_sin':np.sin(2*np.pi*f.index.dayofweek/7),'week_cos':np.cos(2*np.pi*f.index.dayofweek/7)},index=f.index)
    train=f.index<'2025-03-01';model=Ridge(alpha=100).fit(design[train],y[train])
    calendar_holdout={'train':'Dec-Feb','holdout':'March','model':'Fixed calendar sine/cosine Ridge alpha100; no search',
                      'calendar_model':met(y[~train],model.predict(design[~train])),
                      'training_mean_baseline':met(y[~train],np.full((~train).sum(),y[train].mean()))}
    summary={'acf_selected':{str(lag):float(av[lag]) for lag in [1,6,24,48,168]},
             'acf_max_abs_lag1_168':float(np.max(abs(av[1:]))),'pacf_max_abs_lag1_168':float(np.max(abs(pv[1:]))),
             'naive_white_noise_95_bound':float(1.96/np.sqrt(len(y))),
             'ljung_box':lj.to_dict('records'),'calendar_holdout':calendar_holdout,
             'mean_kwh':float(y.mean()),'std_kwh':float(y.std(ddof=1)),'cv_percent':float(y.std(ddof=1)/y.mean()*100),
             'limits':'Failure to reject or small ACF does not prove independence; hour/weekday effects are descriptive and must earn held-out improvement.'}
    write(A/'time_structure.json',summary)
    return summary,cal

def redundancy(f,cols):
    x=f[cols].to_numpy();scaler=StandardScaler().fit(x);z=scaler.transform(x)
    model=PCA(svd_solver='full').fit(z);ratios=model.explained_variance_ratio_;eig=model.explained_variance_
    cumulative=np.cumsum(ratios)
    savecsv(pd.DataFrame({'component':np.arange(1,len(cols)+1),'eigenvalue':eig,'explained_ratio':ratios,'cumulative_ratio':cumulative}),'pca_spectrum.csv')
    corr=f[cols].corr().to_numpy();pairs=[]
    for i,j in zip(*np.where(np.triu(abs(corr)>.95,k=1))):
        pairs.append({'feature1':cols[i],'feature2':cols[j],'r':float(corr[i,j])})
    savecsv(pd.DataFrame(pairs,columns=['feature1','feature2','r']),'redundancy_pairs.csv')
    summary={'feature_count':len(cols),'standardizer_fit_period':'Dec-March explanatory only; fold models fit their own scaler',
             'first_component_variance':float(ratios[0]),'components_for_90_percent':int(np.searchsorted(cumulative,.9)+1),
             'components_for_95_percent':int(np.searchsorted(cumulative,.95)+1),'components_for_99_percent':int(np.searchsorted(cumulative,.99)+1),
             'participation_effective_rank':float(eig.sum()**2/np.sum(eig**2)),'abs_r_over_095_pairs':len(pairs),
             'eigenvalues_below_1e_8':int((eig<1e-8).sum()),
             'excluded_exact_duplicates':'activePower mean = hourly energy over1h; activePower std duplicates canonical std_power_kw',
             'limits':'PCA redundancy describes variance and does not establish predictive information; global PCA is not passed to fold models'}
    write(A/'redundancy_summary.json',summary)
    return summary,cumulative

def importance(f,cols,catalog):
    family={r['name']:r['family'] for r in catalog['features']}
    X=f[cols].shift(1).copy()
    for lag in [1,24,168]:X[f'target_lag{lag}']=f[TARGET].shift(lag);family[f'target_lag{lag}']='past_target'
    for period,key in [(24,'hour'),(7,'weekday')]:
        value=f.index.hour if key=='hour' else f.index.dayofweek
        for trig in ['sin','cos']:
            name=f'{key}_{trig}';X[name]=(np.sin if trig=='sin' else np.cos)(2*np.pi*value/period);family[name]='known_calendar'
    good=X.notna().all(axis=1);X=X[good];y=f.loc[good,TARGET]
    tr=X.index<'2025-03-01';xtr=X[tr].to_numpy();xva=X[~tr].to_numpy();yt=y[tr].to_numpy();yv=y[~tr].to_numpy()
    assert len(yv)==744
    scaler=StandardScaler().fit(xtr);ztr=scaler.transform(xtr);zva=scaler.transform(xva)
    ridge=Ridge(alpha=100).fit(ztr,yt)
    tree=ExtraTreesRegressor(n_estimators=100,max_depth=6,min_samples_leaf=24,random_state=SEED,n_jobs=4).fit(xtr,yt)
    models={'ridge':(ridge,ztr,zva),'extra_trees':(tree,xtr,xva)}
    base=met(yv,np.full(len(yv),yt.mean()));summaries={'training_mean_baseline':base};rows=[]
    rng=np.random.default_rng(SEED);all_names=X.columns.tolist();groups={}
    for i,c in enumerate(all_names):groups.setdefault(family[c],[]).append(i)
    orders=[(rng.permutation(31)[:,None]*24+np.arange(24)).reshape(-1) for _ in range(20)]
    for modelname,(model,xt,xv) in models.items():
        pred=model.predict(xv);score=met(yv,pred);summaries[modelname]=score
        for name,indices in groups.items():
            differences=[];r2drops=[]
            for order in orders:
                mutated=xv.copy();mutated[:,indices]=xv[order][:,indices]
                mp=model.predict(mutated);differences.append(mean_absolute_error(yv,mp)-score['mae_kwh']);r2drops.append(score['r2']-r2_score(yv,mp))
            ablation=None
            if modelname=='ridge':
                keep=[i for i in range(xv.shape[1]) if i not in indices]
                abmodel=Ridge(alpha=100).fit(xt[:,keep],yt)
                ablation=float(mean_absolute_error(yv,abmodel.predict(xv[:,keep]))-score['mae_kwh'])
            rows.append({'model':modelname,'group':name,'columns':len(indices),'base_MAE':score['mae_kwh'],'base_R2':score['r2'],
                         'permutation_MAE_increase_mean':np.mean(differences),'permutation_MAE_increase_std':np.std(differences,ddof=1),
                         'permutation_q05':np.quantile(differences,.05),'permutation_q95':np.quantile(differences,.95),
                         'permutation_R2_drop_mean':np.mean(r2drops),'remove_group_refit_MAE_increase':ablation})
    groupframe=pd.DataFrame(rows);savecsv(groupframe,'importance_grouped.csv')
    # Exact fitted-linear perturbations avoid expensive hundreds of refits.
    pred=ridge.predict(zva);single=[]
    for j,c in enumerate(all_names):
        values=[mean_absolute_error(yv,pred+ridge.coef_[j]*(zva[order,j]-zva[:,j]))-summaries['ridge']['mae_kwh'] for order in orders[:10]]
        single.append({'feature':c,'group':family[c],'model':'fixed_Ridge_alpha100','coefficient_standardized':ridge.coef_[j],
                       'permutation_MAE_increase_mean':np.mean(values),'permutation_MAE_increase_std':np.std(values,ddof=1),
                       'repeats':10,'interpretation':'Model-specific chronological held-out reliance; correlated features split or hide importance'})
    savecsv(pd.DataFrame(single),'importance_individual_ridge.csv')
    summary={'protocol':'Fit Dec-Feb, evaluate March chronological INNER holdout inside the pre-April analysis prefix. One-step forecast uses only row t-1 and known calendar at t.',
             'train_rows':len(yt),'holdout_rows':len(yv),'models':summaries,
             'grouped_permutation':'20 seeded day-block shuffles; all columns in each family moved together. q05/q95 are permutation distribution, not confidence intervals',
             'individual_table':'10 block shuffles of every fitted Ridge column; correlated ties and low-R2 caveat apply',
             'top_groups':groupframe.sort_values('permutation_MAE_increase_mean',ascending=False).head(8).to_dict('records'),
             'limits':'Low held-out R2 makes low importance inconclusive about intrinsic feature value. Correlation substitution can dilute individual scores. Group removal refits are descriptive fixed-Ridge ablations, not outer-CV selection.',
             'source':'https://scikit-learn.org/stable/modules/permutation_importance.html'}
    write(A/'importance_summary.json',summary)
    return summary,groupframe

def counter_and_noise(f):
    audit=pd.read_csv(OUT/'data/derived_audit_only.csv',parse_dates=['hour'],float_precision='round_trip')
    audit=audit[audit.hour<'2025-04-01'];rows=[];noise=[]
    for m,q in audit.groupby('module_id',sort=True):
        q=q.sort_values('hour');y=f[TARGET].reindex(q.hour).to_numpy();t=np.arange(len(q));level=q.counter_last.to_numpy()
        residual=level-np.polyval(np.polyfit(t,level,1),t)
        rows.append({'module_id':int(m),'level_vs_time_r':np.corrcoef(level,t)[0,1],
                     'level_vs_aggregate_target_r':np.corrcoef(level,y)[0,1],
                     'detrended_level_vs_target_r':np.corrcoef(residual,y)[0,1],
                     'within_hour_delta_vs_same_hour_module_energy_r':q.counter_last.sub(q.counter_first).corr(q.energy_kwh),
                     'operation_rate_unique_values':int(q.operation_rate.nunique()),
                     'operation_rate_min':q.operation_rate.min(),'operation_rate_max':q.operation_rate.max()})
        expected=float(np.mean(q.std_power_kw.to_numpy()**2)/720);observed=float(q.energy_kwh.var(ddof=1))
        noise.append({'module_id':int(m),'IID_expected_hourly_variance_kwh2':expected,
                      'observed_hourly_variance_kwh2':observed,'observed_over_IID_expected':observed/expected})
    cf=pd.DataFrame(rows);savecsv(cf,'counter_trend_audit.csv');nf=pd.DataFrame(noise);savecsv(nf,'iid_module_variance.csv')
    expected=float(nf.IID_expected_hourly_variance_kwh2.sum());observed=float(f[TARGET].var(ddof=1))
    covariance=f[feature_columns('module_energy')].cov().to_numpy()
    summary={'assumptions':'Conditionally assume stationary five-second power with independent samples within hours and independent module innovations, with no extra predictable hourly mean changes. This is a diagnostic, not a proven data-generating process.',
             'formula':'sum_m mean_hour(std_power_kw[m,h]^2)/720; one-hour energy numeric equals hourly mean kW under guide units',
             'hours':len(f),'IID_expected_aggregate_hourly_variance_kwh2':expected,
             'actual_aggregate_hourly_variance_kwh2':observed,'actual_over_IID_expected':observed/expected,
             'IID_expected_std_kwh':float(np.sqrt(expected)),'actual_std_kwh':float(np.sqrt(observed)),
             'sum_actual_module_variance_kwh2':float(np.trace(covariance)),
             'net_between_module_covariance_kwh2':float(covariance.sum()-np.trace(covariance)),
             'observed_covariance_sum_vs_target_var_absdiff':float(abs(covariance.sum()-observed)),
             'limits':'Similar variance does not prove Gaussianity, independence, synthetic generation or impossibility of forecasting. It quantifies how much apparent hourly variation could arise from averaging fast fluctuations.'}
    write(A/'iid_noise_diagnostic.json',summary)
    prior=json.loads((ROOT/'output/optimization_20261013/research/forecastability_audit.json').read_text(encoding='utf-8'))['phase_power_diagnostic']
    sample=pd.read_csv(ROOT/'output/analysis/sampled_sensor_rows.csv')
    sample=sample[pd.to_datetime(sample.dt)<'2025-04-01'].copy()
    phase=sum(sample[f'voltage{p}']*sample[f'current{p}']*sample[f'powerFactor{p}']/100 for p in 'RST')/1000
    prior['reportedPower_vs_phase_formula_sample_Pearson_r']=float(sample.activePower.corr(phase))
    sample['formula_phase_kw']=phase
    prior['correlation_sample_rows']=len(sample)
    prior['correlation_analysis_period']='Dec-March only; April excluded'
    prior['prior_ratio_sample_scope']='Previously recorded ratio quantiles use the full16849-point Dec-Apr systematic sample; carried as a unit diagnostic only, not feature selection'
    prior['within_module_correlations']=[{'module':str(m),'sample_rows':len(q),'Pearson_r':float(q.activePower.corr(q.formula_phase_kw))} for m,q in sample.groupby('module(equipment)')]
    prior['pooled_correlation_limit']='Pooled correlation includes module level differences. Same-observation relationship does not give access to future voltage/current/PF, and high correlation does not certify compatible physical units.'
    prior['source_path']='output/optimization_20261013/research/forecastability_audit.json + preexisting systematic sensor sample'
    write(A/'phase_power_diagnostic.json',prior)
    return summary,nf,cf,prior

def make_plots(corr,structure,cal,shuffle,pca,cumulative,importance_frame,noise_frame,noise):
    figures=A/'figures';figures.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(figures/'.mpl_cache')
    sys.path.insert(0,str(ROOT/'output/analysis/.plot_deps'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
    plt.rcParams.update({'font.family':['Malgun Gothic','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none','mathtext.fontset':'dejavusans'})
    records=[]
    def save(fig,name,title):
        fig.savefig(figures/f'{name}.png',dpi=150,bbox_inches='tight');fig.savefig(figures/f'{name}.svg',bbox_inches='tight');plt.close(fig)
        records.append({'name':name,'title':title,'png':str(figures/f'{name}.png'),'svg':str(figures/f'{name}.svg')})
    pear=corr[corr.method.eq('pearson')]
    same=pear[pear.lag_hours.eq(0)].assign(magnitude=lambda d:d.r.abs()).nlargest(12,'magnitude')
    fig,ax=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    ax[0].barh(same.feature[::-1],same.r[::-1],color='#678db2');ax[0].set(xlim=(-1,1),title='동시 상관: 현재 시간은 예측 입력 금지',xlabel='Pearson r')
    candidates=set(feature_columns('physical_sensors'));past=pear[pear.lag_hours.eq(1)&pear.feature.isin(candidates)].assign(magnitude=lambda d:d.r.abs()).nlargest(12,'magnitude')
    ax[1].barh(past.feature[::-1],past.r[::-1],color='#45a188');ax[1].set(xlim=(-.15,.15),title='과거 완료 시간(t-1) 센서와 y(t)',xlabel='Pearson r')
    save(fig,'contemporaneous_vs_past','동시 구성관계와 과거 예측 상관')
    physical=pear[pear.feature.isin(candidates)];rank=physical[physical.lag_hours.gt(0)].groupby('feature').r.apply(lambda s:s.abs().max()).nlargest(24).index
    matrix=physical[physical.feature.isin(rank)].pivot(index='feature',columns='lag_hours',values='r').reindex(rank)
    fig,ax=plt.subplots(figsize=(9,9),layout='constrained');im=ax.imshow(matrix.to_numpy(),aspect='auto',cmap='RdBu_r',vmin=-.15,vmax=.15)
    ax.set_xticks(np.arange(len(matrix.columns)),[f'{l}h'+(' 진단' if l==0 else '') for l in matrix.columns]);ax.set_yticks(np.arange(len(rank)),rank)
    ax.set(title='센서별 시간 간격 상관: Dec–March',xlabel='센서가 정답보다 앞선 시간');fig.colorbar(im,ax=ax,label='Pearson r')
    save(fig,'sensor_lag_heatmap','과거 센서 상관 히트맵')
    ts=pd.read_csv(A/'target_acf_pacf.csv');fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ax,c in zip(axs[0],['acf','pacf']):
        ax.vlines(ts.lag_hours.iloc[1:],0,ts[c].iloc[1:],color='#446582');ax.axhspan(-structure['naive_white_noise_95_bound'],structure['naive_white_noise_95_bound'],color='grey',alpha=.15);ax.set(title=c.upper()+': 1–168h',xlabel='lag (h)',ylim=(-.1,.1))
    for ax,group,title in zip(axs[1],['hour_of_day','weekday'],['시간대별 평균 편차','요일별 평균 편차']):
        q=cal[cal.group.eq(group)];ax.bar(q.value,q.deviation_from_global_kwh,color='#699e8c');ax.set(title=title,xlabel='시간대' if group=='hour_of_day' else '요일: 월0…일6',ylabel='전체 평균 대비 kWh')
    save(fig,'time_structure','자기상관과 시간대·요일 효과')
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained');ax.hist(shuffle['null_max_abs_r'],bins=22,color='#8ca9bd',edgecolor='white')
    ax.axvline(shuffle['observed_max_abs_r'],color='#ca5f40',lw=2,label=f"실제 max |r| {shuffle['observed_max_abs_r']:.4f}");ax.axvline(shuffle['null_q95'],ls='--',color='#333',label='블록 순열 95% 분위')
    ax.set(title=f"{shuffle['tested_pairs']:,}개 피처·lag 탐색의 우연 상관 대조",xlabel='max |Pearson r|',ylabel='순열 횟수');ax.legend()
    save(fig,'block_shuffle_null','200회 날짜 블록 순열 대조')
    fig,axs=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    for ax,model in zip(axs,['ridge','extra_trees']):
        q=importance_frame[importance_frame.model.eq(model)].sort_values('permutation_MAE_increase_mean');ax.barh(q.group,q.permutation_MAE_increase_mean,xerr=q.permutation_MAE_increase_std,color='#779daa');ax.axvline(0,color='grey');ax.set(title=f'{model}: March holdout R²={q.base_R2.iloc[0]:.4f}',xlabel='피처 그룹 순열 후 MAE 증가 (kWh)')
    save(fig,'grouped_importance','시간 순서 holdout의 그룹 중요도')
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained');axs[0].plot(np.arange(1,len(cumulative)+1),cumulative,color='#437691');axs[0].axhline(.95,ls='--',color='grey');axs[0].set(title='표준화 피처 PCA 누적 분산',xlabel='성분 수',ylabel='누적 설명비율')
    spectrum=pd.read_csv(A/'pca_spectrum.csv');axs[1].semilogy(spectrum.component,np.maximum(spectrum.eigenvalue,1e-16),color='#679d82');axs[1].set(title='고윳값 스펙트럼',xlabel='성분',ylabel='고윳값 (로그)')
    from matplotlib.ticker import FuncFormatter
    axs[1].yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:.0e}'))
    save(fig,'pca_redundancy','피처 중복과 PCA 분산 구조')
    fig,axs=plt.subplots(1,2,figsize=(11,5),layout='constrained');positions=np.arange(len(noise_frame));axs[0].bar(positions-.18,noise_frame.IID_expected_hourly_variance_kwh2,.36,label='조건부 IID 기대');axs[0].bar(positions+.18,noise_frame.observed_hourly_variance_kwh2,.36,label='실제 시간 분산');axs[0].set_xticks(positions,noise_frame.module_id);axs[0].set(title='모듈별 시간 전력 분산',xlabel='모듈 ID',ylabel='분산 (kWh²)');axs[0].legend()
    axs[1].bar(['조건부 IID 기대','실제 합계'],[noise['IID_expected_std_kwh'],noise['actual_std_kwh']],color=['#8dabbd','#c88c70']);axs[1].set(title='합계 표준편차: 가정 진단',ylabel='kWh');save(fig,'iid_variance_diagnostic','고빈도 변동의 평균화 진단')
    return records

def main():
    A.mkdir(parents=True,exist_ok=True);started=time.perf_counter();full=load_hourly();f=full[full.index<'2025-04-01'];catalog=load_catalog()
    print('stage correlations',flush=True);corr,stability=correlations(f)
    print('stage prefix selection',flush=True);selections=prefix_selections(full)
    physical=feature_columns('physical_sensors');energy=feature_columns('module_energy');cols=physical+energy
    print('stage shuffle and time',flush=True);null=shuffle_null(f,cols);structure,cal=time_structure(f)
    print('stage PCA and importance',flush=True);pc,cu=redundancy(f,cols);imp,ig=importance(f,cols,catalog)
    print('stage counter/noise and plots',flush=True);noise,nf,cf,phase=counter_and_noise(f)
    plots=make_plots(corr,structure,cal,null,pc,cu,ig,nf,noise)
    p=corr[corr.method.eq('pearson')]
    top_same=p[p.lag_hours.eq(0)].assign(abs_r=lambda d:d.r.abs()).nlargest(15,'abs_r')
    top_past=p[p.lag_hours.gt(0)].assign(abs_r=lambda d:d.r.abs()).nlargest(20,'abs_r')
    stats_frame=f.drop(columns=TARGET).describe().T.reset_index(names='feature');stats_frame['cv_abs_mean']=stats_frame['std']/stats_frame['mean'].abs();savecsv(stats_frame,'feature_statistics.csv')
    result={'status':'complete','analysis_period':'Dec1–March31 only,2904hours; explanatory, not global feature selection',
            'hourly_data':'data/hourly_features.csv','feature_catalog':'data/feature_catalog.json',
            'data_contract':json.loads((OUT/'data/extraction_verification.json').read_text(encoding='utf-8')),
            'target':'target_energy_kwh; provisional13-module sum with unknown meter hierarchy/units',
            'plots':plots,'top_contemporaneous':top_same.to_dict('records'),'top_lagged':top_past.to_dict('records'),
            'top_lagged_physical':p[p.lag_hours.gt(0)&p.feature.isin(physical)].assign(abs_r=lambda d:d.r.abs()).nlargest(20,'abs_r').to_dict('records'),
            'selectorcounts':selections,'importance':imp,'PCA':pc,'ACF_calendar':structure,'shuffle':null,
            'IIDnoise':noise,'countertrend':cf.to_dict('records'),'phase_power':phase,
            'recommendations':['Never use same-hour energy/activePower as input to its aggregate target; shift into completed past or use supervised sequence windows.',
                               'Freeze training-prefix selected schemas before April; final3600h weight/scaler refit may reuse schema but cannot reselect with April labels.',
                               'Compare selected-zero/target-only and multivariate all-sensor candidates in outer CV. Small stable association must still earn predictive gain.',
                               'Report R²/change-prediction skill and baseline-relative error beside low percentage errors on the39k level.',
                               'Provider confirmation is needed for meter hierarchy, unit/circuit scope and actual failure labels. No unit correction applied.'],
            'statistical_limits':['Same-hour large module-energy correlations are algebraic constituent relationships, not evidence of forecasting ability.',
                                  'AR1-adjusted correlation p and BH are approximate under serial dependence/trend; block-max null offers a separate search-opportunity diagnostic.',
                                  'No intrinsic lack of information is proved by small correlation or low importance from low-R² models.',
                                  'April/May were visible in preceding project EDA; this controlled revision is not a pristine prospective experiment.'],
            'references':[{'url':'https://scikit-learn.org/stable/modules/permutation_importance.html','claim':'Model-specific importance and weak-model/correlation limitations'},
                          {'url':'https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html','claim':'Individual correlated importance can be diluted'},
                          {'url':'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html','claim':'Pearson coefficient and standard null-distribution assumptions'}],
            'elapsed_seconds':time.perf_counter()-started,'seed':SEED,'CPU_threads':4}
    write(A/'analysis_results.json',result)
    print(json.dumps(clean({'summary':str(A/'analysis_results.json'),'selected_counts':[v['selected_count'] for v in selections],
                           'max_past_r':null['observed_max_abs_r'],'shuffle_q95':null['null_q95'],'shuffle_p':null['max_stat_empirical_p'],
                           'IIDvariance_ratio':noise['actual_over_IID_expected'],'importance_models':imp['models'],
                           'elapsed':time.perf_counter()-started})),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
