"""Inspect historical predictability and conditional physical-unit consistency.

Diagnostic analysis only; no May information or model selection is used here.
"""
from pathlib import Path
import os,sys,json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'output/modeling/forecast'))
from common import load_training,write_json

def main():
    s=load_training()
    # Diagnostic-only history, ending before the new day-ahead April test.
    development=s.loc[s.index<'2025-04-01']
    coeff=acf(development.to_numpy(),nlags=168,fft=True)
    lj=acorr_ljungbox(development.to_numpy(),lags=[1,24,168],return_df=True)
    sensor=pd.read_csv(ROOT/'output/analysis/sampled_sensor_rows.csv')
    required=['voltageR','voltageS','voltageT','currentR','currentS','currentT','powerFactorR','powerFactorS','powerFactorT','activePower']
    if any(c not in sensor for c in required):raise ValueError('Sensor diagnostic sample lacks phase features')
    # Conditional on voltages being line-to-neutral RMS, current A, PF in percent,
    # and all channels measuring the same circuit simultaneously.
    phase_w=sum(sensor['voltage'+p]*sensor['current'+p]*sensor['powerFactor'+p]/100 for p in 'RST')
    valid=phase_w>0
    ratio_guide=sensor.loc[valid,'activePower']/(phase_w[valid]/1000)
    ratio_if_w=sensor.loc[valid,'activePower']/phase_w[valid]
    result={'mode':'diagnostic_not_hyperparameter_selection','history_start':str(development.index.min()),
       'history_end':str(development.index.max()),'hours':len(development),
       'mean_kwh':float(development.mean()),'std_kwh':float(development.std()),
       'cv_percent':float(development.std()/development.mean()*100),
       'acf_selected':{str(k):float(coeff[k]) for k in [1,2,6,12,24,48,72,168]},
       'acf_max_abs_lag1_168':float(np.abs(coeff[1:]).max()),
       'ljung_box':lj.reset_index(names='lag').to_dict('records'),
       'signal_interpretation':'Failure to reject a white-noise null is not proof of independence. Small effect sizes and held-out baseline comparisons assess available predictive structure.',
       'phase_power_diagnostic':{'sample_rows':len(sensor),'sample_type':'Previous systematic descriptive sensor sample; not a full raw-row audit',
          'assumptions':'Same circuit/time; phase voltage is line-to-neutral V, phase current A, PF percent',
          'formula':'sum(V_phase * I_phase * PF_phase / 100) / 1000 => kW',
          'reported_activePower_if_guide_kw_ratio_quantiles':dict(zip(['q01','median','q99'],map(float,np.quantile(ratio_guide,[.01,.5,.99])))),
          'reported_activePower_if_w_ratio_quantiles':dict(zip(['q01','median','q99'],map(float,np.quantile(ratio_if_w,[.01,.5,.99])))),
          'interpretation':'Recorded scale and conditional electrical identity need provider confirmation. Changing the guide unit from kW to W alone may not reconcile phase channels. No unit correction is applied.'}}
    write_json(OUT/'forecastability_audit.json',result)
    sys.path.insert(0,str(ROOT/'output/analysis/.plot_deps'))
    os.environ['MPLCONFIGDIR']=str(OUT/'tmp/mpl')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False})
    fig,ax=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    ax[0].plot(development.index,development,color='#176b82',lw=.5)
    ax[0].set(title='12–3월 시간 전력량 · 확대 축',ylabel='잠정 합계 kWh')
    ax[1].stem(np.arange(1,169),coeff[1:],markerfmt=' ',basefmt=' ')
    bound=1.96/np.sqrt(len(development));ax[1].axhline(bound,color='#bf5e28',ls='--');ax[1].axhline(-bound,color='#bf5e28',ls='--')
    ax[1].set(title='자기상관 · 점별 근사95%선',xlabel='지연 시간',ylabel='ACF')
    ax[2].hist(ratio_guide,bins=50,color='#176b82',alpha=.85)
    ax[2].set(title='위상 전력 공식 대비 기록값 비율',xlabel='activePower(kW가정) / 위상공식(kW)',ylabel='표본 수')
    fig.suptitle('최적화 전 데이터 신호·단위 진단 · 물리식은 계측 조건 가정')
    for ext in ['png','svg']:fig.savefig(OUT/f'forecastability_audit.{ext}',dpi=170)
    plt.close(fig)
    print(json.dumps({k:result[k] for k in ['hours','cv_percent','acf_selected','ljung_box','phase_power_diagnostic']},ensure_ascii=True,indent=2))

if __name__=='__main__':main()
