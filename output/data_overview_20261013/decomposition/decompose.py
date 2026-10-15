"""Descriptive robust STL/MSTL and raw-signal periodicity diagnostics.

All outputs are confined to this new folder. The audited target is read-only.
Full-history decomposition is intentionally retrospective, never a forecast input.
"""
from pathlib import Path
import sys,os,json,hashlib,time,itertools
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from scipy.stats import norm
from statsmodels.tsa.seasonal import STL,MSTL
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
import statsmodels,scipy
from threadpoolctl import threadpool_limits
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
SOURCE=ROOT/'output/analysis/baseline_audit/complete_hour_module_sum.csv'
EXPECTED_SHA='729b149a3e3165dfb339f0386249bfbda0326657f9e722d86da8138db36117de'
SEED=20261013
PERMUTATIONS=1999
NULL_DECOMPOSITIONS=19
threadpool_limits(limits=4)
os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig')
sys.path.insert(0,str(ROOT/'output/analysis/.plot_deps'))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name(),
                     'axes.unicode_minus':False,'svg.fonttype':'none','axes.formatter.useoffset':False})

REFERENCES=[
 {'title':'statsmodels0.15 STL API','url':'https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html',
  'use':'Fixed robust LOESS settings, smoother windows and trend/low-pass defaults.'},
 {'title':'statsmodels0.15 MSTL API','url':'https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.MSTL.html',
  'use':'Multiple periods24/168, explicit default-equivalent windows11/15, two iterations and robust STL kwargs; no Box-Cox transformation.'},
 {'title':'Forecasting: Principles and Practice3, STL features','url':'https://otexts.com/fpp3/features.html#stl-features',
  'use':'Descriptive trend/seasonal strength definitions based on remainder variance; no out-of-sample forecast R2 interpretation.'},
 {'title':'statsmodels ACF API','url':'https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html',
  'use':'Raw-series ACF and Bartlett pointwise95% confidence intervals.'},
 {'title':'statsmodels Ljung-Box API','url':'https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html',
  'use':'Joint raw autocorrelation diagnostics, not a test specific to one seasonal period.'},
 {'title':'SciPy periodogram API','url':'https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html',
  'use':'One-sided boxcar spectrum with mean removal, fs1sample/time-step and spectrum scaling.'}]

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(Path,pd.Timestamp)):return str(x)
    if isinstance(x,(bool,np.bool_)):return bool(x)
    if isinstance(x,(int,np.integer)):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x
def save_json(path,x):Path(path).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def variance(a):return float(np.var(np.asarray(a,dtype=np.float64),ddof=1))
def strength(component,resid):
    denom=variance(np.asarray(component)+np.asarray(resid))
    return max(0.,min(1.,1-variance(resid)/denom)) if denom>0 else 0.

def components(result,y):
    seasonal=result.seasonal
    if isinstance(seasonal,pd.DataFrame):
        seasonal_parts={str(c):seasonal[c].to_numpy(dtype=np.float64) for c in seasonal.columns}
        total=seasonal.to_numpy(dtype=np.float64).sum(axis=1)
    else:
        total=np.asarray(seasonal,dtype=np.float64);seasonal_parts={}
    frame=pd.DataFrame({'observed':np.asarray(y,dtype=np.float64),'trend':np.asarray(result.trend,dtype=np.float64),
                        'seasonal':total,'resid':np.asarray(result.resid,dtype=np.float64)},index=y.index)
    for name,values in seasonal_parts.items():frame[name]=values
    frame['robust_weight']=np.asarray(result.weights,dtype=np.float64)
    return frame

def decomposition_statistics(frame):
    parts=['trend']+[c for c in frame if c.startswith('seasonal_')]+['resid']
    if len(parts)==2:parts=['trend','seasonal','resid']
    cov=frame[parts].cov().to_numpy(dtype=np.float64)
    var_y=variance(frame.observed);sum_var=float(np.trace(cov));cross=float(cov.sum()-sum_var)
    error=float(np.max(np.abs(frame.observed-(frame.trend+frame.seasonal+frame.resid))))
    out={'input_mean':float(frame.observed.mean()),'input_std':float(frame.observed.std(ddof=1)),
         'input_cv_percent':float(frame.observed.std(ddof=1)/abs(frame.observed.mean())*100),
         'residual_std':float(frame.resid.std(ddof=1)),'trend_std':float(frame.trend.std(ddof=1)),
         'seasonal_std':float(frame.seasonal.std(ddof=1)),
         'seasonal_strength':strength(frame.seasonal,frame.resid),'trend_strength':strength(frame.trend,frame.resid),
         'component_scale_as_percent_of_observed_mean':{c:float(frame[c].std(ddof=1)/abs(frame.observed.mean())*100) for c in ['trend','seasonal','resid']},
         'component_variances':{c:variance(frame[c]) for c in parts},
         'component_variance_to_input_variance':{c:variance(frame[c])/var_y for c in parts},
         'covariance_matrix':{'components':parts,'matrix':cov.tolist()},
         'variance_identity':{'observed_variance':var_y,'sum_component_variances':sum_var,'twice_sum_cross_covariances':cross,
                              'reconciled_variance':sum_var+cross,'absolute_error':abs(var_y-sum_var-cross)},
         'reconstruction_max_absdiff':error,
         'interpretation':'Strength is an in-sample decomposition statistic, not significance, forecast accuracy or predictive R2. Component variance shares need not sum to1 because components covary; the covariance identity is provided.'}
    if len(parts)>3:
        out['individual_seasonal_strength']={c:strength(frame[c],frame.resid) for c in parts if c.startswith('seasonal_')}
        out['individual_seasonal_std']={c:float(frame[c].std(ddof=1)) for c in parts if c.startswith('seasonal_')}
    return out

def four_panel(frame,title,unit,stem):
    fig,axes=plt.subplots(4,1,figsize=(14,9.4),sharex=True,layout='constrained')
    labels=[('observed','관측값','#475569'),('trend','추세','#2563eb'),('seasonal','지정 주기 계절성분','#0d9488'),('resid','잔차','#9333ea')]
    for ax,(col,label,color) in zip(axes,labels):
        ax.plot(frame.index,frame[col],color=color,lw=.8 if len(frame)>1000 else 1.1)
        ax.set_ylabel(label+'\n'+unit);ax.grid(alpha=.2)
        if col in ['seasonal','resid']:ax.axhline(0,color='black',lw=.5)
    fig.suptitle(title+'\n전체 과거 자료를 이용한 기술적 분해: 미래 예측 피처로 사용하지 않음',fontsize=13)
    for ext in ['png','svg']:fig.savefig(OUT/f'{stem}.{ext}',dpi=150)
    plt.close(fig)
    return {'png':str(OUT/f'{stem}.png'),'svg':str(OUT/f'{stem}.svg')}

def phase_fraction(a,period):
    a=np.asarray(a,dtype=np.float64);phase=np.arange(len(a))%period;counts=np.bincount(phase,minlength=period)
    sums=np.bincount(phase,weights=a,minlength=period);means=sums/counts
    return float(np.sum(counts*(means-a.mean())**2)/np.sum((a-a.mean())**2))
def spectrum_fraction(a):
    n=len(a);fft=np.fft.rfft(np.asarray(a,dtype=np.float64)-np.mean(a));power=np.abs(fft)**2
    power[1:-1 if n%2==0 else None]*=2
    power[0]=0;return power/power.sum()

def raw_diagnostics(y,periods_to_check,label,max_lag):
    a=y.to_numpy(dtype=np.float64);n=len(a)
    acfs,ci=acf(a,nlags=max_lag,alpha=.05,fft=True,bartlett_confint=True)
    acf_table=pd.DataFrame({'lag':np.arange(len(acfs)),'acf':acfs,'ci95_lower':ci[:,0],'ci95_upper':ci[:,1]})
    acf_table.to_csv(OUT/f'{label}_raw_acf.csv',index=False,float_format='%.17g')
    lags=[24,168,336] if label=='H' else [7,14,28]
    lb=acorr_ljungbox(a,lags=lags,model_df=0,return_df=True)
    lb.reset_index(names='lag').to_csv(OUT/f'{label}_raw_ljungbox.csv',index=False,float_format='%.17g')
    freq,power=periodogram(a,fs=1.,window='boxcar',detrend='constant',scaling='spectrum')
    fraction=power/power.sum()
    table=pd.DataFrame({'frequency_cycles_per_step':freq,'period_steps':np.divide(1,freq,out=np.full_like(freq,np.nan),where=freq>0),
                        'power_spectrum_unit_squared':power,'fraction_total_spectral_power':fraction})
    table.to_csv(OUT/f'{label}_raw_periodogram.csv',index=False,float_format='%.17g')
    rng=np.random.default_rng(SEED+(0 if label=='H' else 1));phase_null={p:[] for p in periods_to_check};bin_null={p:[] for p in periods_to_check};max_null=[]
    bin_indices={p:int(np.argmin(abs(freq-1/p))) for p in periods_to_check}
    actual_phase={p:phase_fraction(a,p) for p in periods_to_check}
    for repeat in range(PERMUTATIONS):
        shuffled=rng.permutation(a);spectral=spectrum_fraction(shuffled);max_null.append(float(spectral[1:].max()))
        for p in periods_to_check:
            phase_null[p].append(phase_fraction(shuffled,p));bin_null[p].append(float(spectral[bin_indices[p]]))
    diagnostics=[]
    for p in periods_to_check:
        bin_index=bin_indices[p];pn=np.asarray(phase_null[p]);bn=np.asarray(bin_null[p])
        row={'period_steps':p,'cycles_observed':n/p,'raw_acf_at_period':float(acfs[p]),'raw_acf_ci95_at_period':ci[p].tolist(),
             'zero_in_pointwise_ci95':bool(ci[p,0]<=0<=ci[p,1]),
             'phase_mean_between_variance_fraction':actual_phase[p],
             'phase_permutation_p':float((1+np.sum(pn>=actual_phase[p]))/(PERMUTATIONS+1)),
             'phase_null_95pct_reference_range':np.quantile(pn,[.025,.975]).tolist(),
             'spectral_bin_frequency':float(freq[bin_index]),'spectral_bin_period':float(1/freq[bin_index]),
             'spectral_nearest_bin_power_fraction':float(fraction[bin_index]),
             'spectral_bin_permutation_p':float((1+np.sum(bn>=fraction[bin_index]))/(PERMUTATIONS+1)),
             'spectral_bin_note':'Nearest finite-window Fourier bin; it equals the exact desired frequency only if sample length is divisible by the period.'}
        diagnostics.append(row)
    peak=int(np.argmax(fraction[1:])+1)
    spectrum_global={'peak_frequency':float(freq[peak]),'peak_period_steps':float(1/freq[peak]),'peak_power_fraction':float(fraction[peak]),
                     'max_peak_any_frequency_permutation_p':float((1+np.sum(np.asarray(max_null)>=fraction[peak]))/(PERMUTATIONS+1))}
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(14,7),layout='constrained')
    ax1.vlines(np.arange(1,len(acfs)),0,acfs[1:],color='#2563eb',lw=.7)
    ax1.fill_between(np.arange(len(acfs)),ci[:,0]-acfs,ci[:,1]-acfs,color='#94a3b8',alpha=.25,label='Bartlett95% pointwise band around0')
    for p in periods_to_check:ax1.axvline(p,color='#f97316',lw=1,ls='--')
    ax1.axhline(0,color='black',lw=.5);ax1.set_xlabel('Lag (hours)' if label=='H' else 'Lag (days)');ax1.set_ylabel('Raw ACF');ax1.legend();ax1.grid(alpha=.2)
    keep=freq>0;ax2.plot(1/freq[keep],fraction[keep],color='#0d9488',lw=.8);ax2.set_xscale('log')
    for p in periods_to_check:ax2.axvline(p,color='#f97316',lw=1,ls='--')
    ax2.set_xlabel('Period (hours)' if label=='H' else 'Period (days)');ax2.set_ylabel('Fraction spectral power');ax2.grid(alpha=.2)
    fig.suptitle(f'{label}: 원자료 주기성 진단 - STL 계절성분과 독립적으로 검사')
    for ext in ['png','svg']:fig.savefig(OUT/f'{label}_raw_periodicity.{ext}',dpi=150)
    plt.close(fig)
    return {'sample_count':n,'sample_step':'hour' if label=='H' else 'day','period_tests':diagnostics,
            'ljung_box':[{'lag':int(lag),'Q':float(row.lb_stat),'p':float(row.lb_pvalue)} for lag,row in lb.iterrows()],
            'spectrum_global':spectrum_global,'permutations':PERMUTATIONS,
            'null':'IID exchangeable ordering: shuffling preserves the observed marginal values and removes time alignment. Approximate if serial dependence exists; nonrejection does not prove independence.',
            'acf_limit':'Pointwise95% intervals are not simultaneous confidence bands; Ljung-Box tests joint correlations up to a lag, not a specific24/168/7period.',
            'figure':{'png':str(OUT/f'{label}_raw_periodicity.png'),'svg':str(OUT/f'{label}_raw_periodicity.svg')},
            'csv':{'acf':str(OUT/f'{label}_raw_acf.csv'),'ljungbox':str(OUT/f'{label}_raw_ljungbox.csv'),'periodogram':str(OUT/f'{label}_raw_periodogram.csv')}}

def main():
    OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic();before=sha(SOURCE)
    if before!=EXPECTED_SHA:raise ValueError('Audited hourly source SHA differs')
    source=pd.read_csv(SOURCE,float_precision='round_trip');y=pd.Series(source.provisional_sum_energy_kwh.to_numpy(dtype=np.float64),index=pd.to_datetime(source.hour),name='energy_kwh')
    expected=pd.date_range('2024-12-01','2025-04-29 23:00',freq='h');assert y.index.equals(expected) and np.isfinite(y).all()
    sys.path.insert(0,str(ROOT/'output/resampling_modeling_20261013'))
    from resample_common import periods
    tables={label:periods(y,freq,complete_only=True) for label,freq in [('D','D'),('W','W-SUN'),('M','MS')]}
    assert len(tables['D'])==150 and len(tables['W'])==21 and len(tables['M'])==4
    for label,t in tables.items():t.to_csv(OUT/f'{label}_complete_periods.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    d=pd.Series(tables['D'].rate_kwh_per_hour.to_numpy(dtype=np.float64),index=tables['D'].index,name='daily_average_load')
    plan={'source':str(SOURCE),'source_sha256':before,'seed':SEED,'full_history_descriptive_only':True,'settings_search':False,
          'STL':{'seasonal':13,'robust':True,'trend':'Documented library default','low_pass':'Documented library default','fit_iterations':'Library defaults'},
          'MSTL':{'periods':[24,168],'windows':[11,15],'iterate':2,'lmbda':None,'stl_kwargs':{'robust':True}},
          'raw_permutation_tests':PERMUTATIONS,'same_setting_STL_null_calibrations':NULL_DECOMPOSITIONS,
          'null_calibration_limit':'19 shuffled-reference decompositions are an illustrative forced-pattern comparison, not a precise tail probability or optimized smoother.'}
    save_json(OUT/'analysis_plan.json',plan)
    cases={};case_specs=[('hourly_stl24',y,24,'시간자료 STL:24시간 지정','시간 전력량(kWh,잠정)'),
                        ('hourly_stl168',y,168,'시간자료 STL:168시간 지정','시간 전력량(kWh,잠정)'),
                        ('daily_stl7',d,7,'일자료 STL:7일 지정','기간 평균 부하(kWh/h)')]
    fitted_frames={}
    for cid,series,p,label,unit in case_specs:
        model=STL(series,period=p,seasonal=13,robust=True);fit=model.fit();frame=components(fit,series)
        if cid=='daily_stl7':frame['observed_daily_energy_kwh']=tables['D'].energy_kwh.to_numpy(dtype=np.float64)
        frame.to_csv(OUT/f'{cid}_components.csv',index_label='time',encoding='utf-8-sig',float_format='%.17g')
        stat=decomposition_statistics(frame)
        cases[cid]={'id':cid,'label':label,'period':p,'frequency':'H' if cid.startswith('hourly') else 'D','unit':unit,
                    'samples':len(series),'settings':model.config,'components_csv':str(OUT/f'{cid}_components.csv'),
                    'figure':four_panel(frame,label,unit,cid),**stat}
        fitted_frames[cid]=frame
        print(f'{cid} fit: seasonalstrength={stat["seasonal_strength"]:.4f}; trendstrength={stat["trend_strength"]:.4f}; elapsed{time.monotonic()-started:.1f}s',flush=True)
    mstl=MSTL(y,periods=(24,168),windows=(11,15),lmbda=None,iterate=2,stl_kwargs={'robust':True});multi=components(mstl.fit(),y)
    multi.to_csv(OUT/'hourly_mstl24_168_components.csv',index_label='time',encoding='utf-8-sig',float_format='%.17g')
    cases['hourly_mstl24_168']={'id':'hourly_mstl24_168','label':'시간자료 MSTL:24+168시간 지정','period':[24,168],
        'frequency':'H','unit':'시간 전력량(kWh,잠정)','samples':len(y),'settings':plan['MSTL'],
        'components_csv':str(OUT/'hourly_mstl24_168_components.csv'),
        'figure':four_panel(multi,'시간자료 MSTL:24+168시간 지정','시간 전력량(kWh,잠정)','hourly_mstl24_168'),**decomposition_statistics(multi)}
    fig,axes=plt.subplots(2,1,figsize=(14,6),sharex=True,layout='constrained')
    for ax,col in zip(axes,['seasonal_24','seasonal_168']):
        ax.plot(multi.index,multi[col],color='#0d9488',lw=.8);ax.axhline(0,color='black',lw=.6);ax.set_ylabel(col+'\nkWh,잠정');ax.grid(alpha=.2)
    fig.suptitle('MSTL의 두 계절성분: 지정 주기별 추출값이며 주기성 검정 결과와 구분')
    for ext in ['png','svg']:fig.savefig(OUT/f'hourly_mstl_individual_seasons.{ext}',dpi=150)
    plt.close(fig)
    cases['hourly_mstl24_168']['individual_seasons_figure']={'png':str(OUT/'hourly_mstl_individual_seasons.png'),'svg':str(OUT/'hourly_mstl_individual_seasons.svg')}
    print(f'MSTL supported and fitted; elapsed{time.monotonic()-started:.1f}s',flush=True)
    raw={'H':raw_diagnostics(y,[24,168],'H',336),'D':raw_diagnostics(d,[7],'D',28)}
    all_period_tests=[t for diag in raw.values() for t in diag['period_tests']]
    adjusted=multipletests([t['phase_permutation_p'] for t in all_period_tests],alpha=.05,method='holm')[1]
    for row,padj in zip(all_period_tests,adjusted):row['phase_permutation_Holm_adjusted_p']=float(padj)
    print(f'Raw ACF/LjungBox/spectrum and1999shuffle tests finished; elapsed{time.monotonic()-started:.1f}s',flush=True)
    rng=np.random.default_rng(SEED+10);null_rows=[]
    for cid,series,p,label,unit in case_specs:
        observed=cases[cid]['seasonal_strength'];strengths=[];trend_strengths=[]
        for repeat in range(NULL_DECOMPOSITIONS):
            shuffled=rng.permutation(series.to_numpy(dtype=np.float64));fit=STL(shuffled,period=p,seasonal=13,robust=True).fit()
            ss=strength(fit.seasonal,fit.resid);ts=strength(fit.trend,fit.resid);strengths.append(ss);trend_strengths.append(ts)
            null_rows.append({'case':cid,'permutation':repeat+1,'seasonal_strength':ss,'trend_strength':ts})
            if (repeat+1)%5==0:print(f'{cid} null-reference{repeat+1}/{NULL_DECOMPOSITIONS}; elapsed{time.monotonic()-started:.1f}s',flush=True)
        cases[cid]['same_setting_shuffled_STL_reference']={'replicates':NULL_DECOMPOSITIONS,'seasonal_strength_mean':float(np.mean(strengths)),
             'seasonal_strength_5_95pct_reference_range':np.quantile(strengths,[.05,.95]).tolist(),
             'coarse_one_sided_permutation_p':float((1+np.sum(np.asarray(strengths)>=observed))/(NULL_DECOMPOSITIONS+1)),
             'trend_strength_mean':float(np.mean(trend_strengths)),
             'trend_strength_5_95pct_reference_range':np.quantile(trend_strengths,[.05,.95]).tolist(),
             'coarse_one_sided_trend_permutation_p':float((1+np.sum(np.asarray(trend_strengths)>=cases[cid]['trend_strength']))/(NULL_DECOMPOSITIONS+1)),
             'minimum_p_resolution':1/(NULL_DECOMPOSITIONS+1),
             'interpretation':'The same STL settings extract a seasonal curve even after time order is shuffled. This19replicate reference is illustrative; raw prespecified phase tests use1999 permutations.'}
    pd.DataFrame(null_rows).to_csv(OUT/'shuffled_STL_strength_reference.csv',index=False,float_format='%.17g')
    strengths=pd.DataFrame([{'case':cid,'seasonal_strength':c['seasonal_strength'],'trend_strength':c['trend_strength'],
                   'input_std':c['input_std'],'seasonal_std':c['seasonal_std'],'trend_std':c['trend_std'],'residual_std':c['residual_std'],
                   'unit':c['unit'],'seasonal_std_as_percent_observed_mean':c['component_scale_as_percent_of_observed_mean']['seasonal']} for cid,c in cases.items()])
    strengths.to_csv(OUT/'strength_and_scale.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    names=strengths.case.to_list();pos=np.arange(len(names));axes[0].barh(pos-.15,strengths.seasonal_strength,height=.3,color='#0d9488',label='Descriptive seasonal strength');axes[0].barh(pos+.15,strengths.trend_strength,height=.3,color='#2563eb',label='Descriptive trend strength')
    axes[0].set_yticks(pos,names);axes[0].set_xlabel('In-sample strength (not significance or prediction score)');axes[0].legend()
    for i,cid in enumerate(names):
        ref=cases[cid].get('same_setting_shuffled_STL_reference')
        if ref:axes[0].plot(ref['seasonal_strength_5_95pct_reference_range'],[i-.15,i-.15],color='#f97316',lw=3)
    axes[1].barh(pos,strengths.seasonal_std_as_percent_observed_mean,color='#9333ea');axes[1].set_yticks(pos,names);axes[1].set_xlabel('Seasonal SD / observed mean (%)');axes[1].grid(axis='x',alpha=.2)
    fig.suptitle('분해 강도와 실제 스케일 | 주황선: 시간순서를 섞어도 생성되는 STL강도 참고범위')
    for ext in ['png','svg']:fig.savefig(OUT/f'strength_and_scale.{ext}',dpi=150)
    plt.close(fig)
    verification=[]
    for cid,c in cases.items():
        frame=pd.read_csv(c['components_csv'],float_precision='round_trip')
        recon=float(np.max(np.abs(frame.observed-frame.trend-frame.seasonal-frame.resid)));assert recon<1e-8
        csv_strength=strength(frame.seasonal,frame.resid);diff=abs(csv_strength-c['seasonal_strength']);assert diff<1e-10
        assert len(frame)==c['samples'] and np.isfinite(frame.select_dtypes('number').to_numpy()).all()
        verification.append({'case':cid,'rows':len(frame),'reconstruction_max_absdiff':recon,'strength_csv_recompute_diff':diff})
    assert sha(SOURCE)==before
    result={'status':'complete','source':{'path':str(SOURCE),'sha256':before,'source_unchanged':True,'hours':len(y),'start':str(y.index[0]),'end':str(y.index[-1]),
         'target':'13-module complete-hour provisional guidekWh sum; unknown meter hierarchy and scale remain unresolved.'},
         'cases':cases,'strength_table_csv':str(OUT/'strength_and_scale.csv'),'strength_figure':{'png':str(OUT/'strength_and_scale.png'),'svg':str(OUT/'strength_and_scale.svg')},
         'raw_periodicity':raw,'short_series_limits':{'weekly':{'complete_periods':21,'hypothetical_annual_period':52,'decomposition_performed':False,'reason':'21complete weeks do not cover one full annual cycle; no arbitrary52week seasonality fitted.'},
             'monthly':{'complete_periods':4,'hypothetical_annual_period':12,'decomposition_performed':False,'reason':'4complete months cannot estimate annual12month seasonal effects. April is partial and excluded.'}},
         'interpretation':{'full_history_leakage':'STL/MSTL uses smoothing over the completeDec-Apr history, including values later than each timestamp. These descriptive components are not admissible historical-origin forecast features. Forecast applications must recompute within the available training prefix and validate chronologically.',
             'forced_pattern':'A seasonal curve necessarily follows a requested period; positive extracted strength alone is not evidence of genuine periodicity. Same-setting shuffled references and tests on raw time order distinguish the two.',
             'variance_not_R2':'No forecast R2 is calculated. The additive reconstruction is a definition and covariance-adjusted variance accounting is descriptive, not proof of predictive fit.',
             'uncertainty':'Permutation tests assume exchangeable ordering under the null. Pointwise ACF intervals and short weekly/monthly coverage cannot establish independence, causality, future predictability or factory generalization.',
             'units':'Hourly components: provisional energy in each1hour bucket(kWh). Daily components: full-day energy/24observed hours(kWh/h), period average load without rated-capacity normalization. Component percentages are relative to observed mean, not equipment load factors.'},
         'settings_search_performed':False,'May_data_used':False,'libraries':{'statsmodels':statsmodels.__version__,'scipy':scipy.__version__,'numpy':np.__version__,'pandas':pd.__version__},
         'references':REFERENCES,'references_accessed_date':'2026-10-13',
         'verification':{'status':'passed','cases':verification,'original_target_sha_unchanged':True},'runtime_seconds':time.monotonic()-started}
    save_json(OUT/'results.json',result);save_json(OUT/'verification.json',result['verification'])
    print(json.dumps({'status':'complete','cases':list(cases),'raw_phase_Holm_p':[r['phase_permutation_Holm_adjusted_p'] for r in all_period_tests],
                      'output':str(OUT/'results.json'),'runtime_seconds':result['runtime_seconds']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
