"""Freeze validation-only selection, then evaluate May and render local report.

May is opened only after the immutable global selection manifest is written.
Native model fitting is performed by separate, already locked scripts.
"""
from pathlib import Path
import sys,json,hashlib,html,datetime,os,uuid
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path.insert(0,str(OUT));from resample_common import load_hourly_target,metrics

def write(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def frame(path):
    f=pd.read_csv(path,float_precision='round_trip');f['id']=pd.to_datetime(f['id']);return f
def rel(path):return Path(path).relative_to(OUT).as_posix()

def model_cards():
    cards=[]
    co=read(OUT/'coarse/results.json');ho=read(OUT/'hourly/results.json');ne=read(OUT/'neural/results.json')
    assert co['status']=='complete' and ne['status']=='complete'
    for key,v in co['models'].items():
        cards.append(dict(key=key,label=key.replace('_',' ').title(),grain=key.split('_')[0],
          val=v['validation']['late505'],test=v['test']['late505'],config=v['selected_config'],
          paths=v['prediction_csv'],lock=co['selection_lock'],source=rel(OUT/'coarse'/key/'results.json'),
          trials=None,reload=v['reload_max_abs_diff'],scope='완료된 과거 기간의 부하율 + 과거 시간 배분 패턴'))
    for key,v in ho['models'].items():
        cards.append(dict(key='hourly_'+key,label='Hourly '+key.title(),grain='hourly',
          val=v['validation_late505'],test=v['internal_test']['late505'],config=v['selected_config'],
          paths=dict(validation=v['paths']['validation_csv'],test=v['paths']['test_csv'],future=v['paths']['future_csv']),
          lock=ho['selection_lock'],source=rel(OUT/'hourly'/key/'results.json'),trials=v['trials_count'],
          reload=v['saved_model']['reload_max_absdiff_kwh'],scope='과거 시간 부하 + 선택된 H/D/W/M 요약 + 미래 달력'))
    for grain,s in ne['scenarios'].items():
        assert len(s['models'])==3
        for key,v in s['models'].items():
            candidates=read(v['candidate_results_json'])['candidate_results']
            cards.append(dict(key=grain+'_'+key,label=grain.title()+' '+{'patchtst':'PatchTST','itransformer':'iTransformer','chronos2':'Chronos-2'}[key],grain=grain,
              val=v['validation']['late505'],test=v['internal_test']['late505'],config=v['selected_config'],
              paths=dict(validation=v['validation_prediction_csv'],test=v['fits']['internal_test']['prediction_csv'],future=v['fits']['future_unlabelled']['prediction_csv']),
              lock=v['selection_lock'],source=rel(OUT/'neural'/grain/key/'results.json'),trials=len(candidates),
              reload=max(x.get('max_abs_native_replay_error',x.get('cached_raw_plus_locked_blend_replay_max_error',0)) for x in v['replay_verification']),
              scope=v['input_scope']))
    assert len(cards)==17
    for c in cards:
        # Different native result schemas share the same forecast data contract.
        for k,p in c['paths'].items():
            f=frame(p);expected=2235 if k=='validation' else 745
            assert len(f)==expected and np.isfinite(f.predicted_kwh).all()
            if k=='future':assert f.actual_kwh.isna().all()
            else:
                score=f[f.horizon_offset_hours.between(240,744)]
                recomputed=metrics(score.actual_kwh,score.predicted_kwh)
                assert abs(recomputed['mae_kwh']-c['val' if k=='validation' else 'test']['mae_kwh'])<1e-7
    return cards

def freeze(cards):
    dest=OUT/'global_selection_lock.json'
    selected=min(cards,key=lambda c:(c['val']['mae_kwh'],c['key']))
    hashes={rel(c['lock']):sha(c['lock']) for c in cards}
    future_hashes={c['key']:sha(c['paths']['future']) for c in cards}
    content=dict(created_utc=now(),selected_model=selected['key'],criterion='Mean3-origin Late505 MAE; no test/May selection',
      validation_origins=['2025-02-15','2025-02-22','2025-02-25'],selected_validation_mae=selected['val']['mae_kwh'],
      rankings=[dict(model=c['key'],validation_mae_kwh=c['val']['mae_kwh'])for c in sorted(cards,key=lambda c:c['val']['mae_kwh'])],
      native_selection_lock_sha256=hashes,future_path_sha256=future_hashes,May_read_at_lock=False,
      native_candidate_count=76+40+sum(c['trials']for c in cards if 'neural/' in c['source']),
      model_families_after_selection=17,optimality_limit='Best among finite137 candidates under this historical protocol; not a mathematical/global optimum',
      test_selection_prohibited=True,earlier_EDA_acknowledged=True)
    if dest.exists():
        old=read(dest)
        for k in ['selected_model','native_selection_lock_sha256','future_path_sha256','rankings']:
            assert old[k]==content[k],f'Frozen manifest changed: {k}'
        content=old
    else:write(dest,content)
    return content

def evaluate(cards,lock):
    evaluation_started=now();assert (OUT/'global_selection_lock.json').exists()
    source=ROOT/'may-dataset/rtu_ground_truth_may.csv'
    may=pd.read_csv(source,float_precision='round_trip');may['id']=pd.to_datetime(may['id'])
    assert len(may)==505 and may.id.equals(pd.Series(pd.date_range('2025-05-10','2025-05-31 00:00',freq='h')))
    assert sha(source)=='a0e44dc6481fba7d78a2c5b60f0f5e0c2bcc69a8f732c9229bbbdaa38dee04bf'
    dest=OUT/'evaluation';dest.mkdir(exist_ok=True)
    rows=[];allpred=[];native=[]
    for c in cards:
        path=frame(c['paths']['future']);f=may[['id','hourly_pow','may_bill','may_carbon']].merge(path[['id','predicted_kwh','horizon_offset_hours','lead_hours']],on='id',validate='one_to_one')
        assert len(f)==505 and list(f.horizon_offset_hours)==list(range(240,745))
        f['actual_kwh']=f.hourly_pow;f['model']=c['key'];f['predicted_bill_won']=f.predicted_kwh*180;f['predicted_carbon_kg']=f.predicted_kwh*.424
        f['absolute_error_kwh']=abs(f.predicted_kwh-f.actual_kwh);f['error_pred_minus_actual_kwh']=f.predicted_kwh-f.actual_kwh
        c['may']=metrics(f.actual_kwh,f.predicted_kwh);c['may_csv']=rel(dest/(c['key']+'_may_predictions.csv'))
        f.to_csv(OUT/c['may_csv'],index=False,encoding='utf-8-sig',float_format='%.17g');allpred.append(f)
        rows.append(dict(model=c['key'],grain=c['grain'],validation_mae_kwh=c['val']['mae_kwh'],internal_mae_kwh=c['test']['mae_kwh'],
          may_mae_kwh=c['may']['mae_kwh'],may_rmse_kwh=c['may']['rmse_kwh'],may_r2=c['may']['r2'],may_mape_percent=c['may']['mape_percent'],
          may_bias_kwh=c['may']['bias_pred_minus_actual_kwh'],selected_by_validation=c['key']==lock['selected_model']))
        # Native aggregates are scored only where every hour of the period is supplied.
        for freq,grain in [('D','daily'),('W-SUN','weekly'),('MS','monthly')]:
            g=f.set_index('id').resample(freq).agg(actual_kwh=('actual_kwh','sum'),predicted_kwh=('predicted_kwh','sum'),observed_hours=('actual_kwh','size'))
            starts=g.index-pd.Timedelta(days=6) if grain=='weekly' else g.index
            ends=starts+pd.Timedelta(days=1 if grain=='daily' else 7) if grain!='monthly' else starts+pd.offsets.MonthBegin(1)
            good=g.observed_hours.to_numpy()==((ends-starts)/pd.Timedelta(hours=1)).to_numpy(int)
            g=g[good]
            native.append(dict(model=c['key'],evaluation_grain=grain,complete_periods=len(g),**(metrics(g.actual_kwh,g.predicted_kwh) if len(g) else {})))
    pd.DataFrame(rows).to_csv(dest/'model_comparison.csv',index=False,encoding='utf-8-sig')
    pd.concat(allpred).to_csv(dest/'all_model_may_predictions.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    pd.DataFrame(native).to_csv(dest/'native_aggregate_evaluation.csv',index=False,encoding='utf-8-sig')
    c=next(c for c in cards if c['key']==lock['selected_model'])
    pred=frame(c['paths']['future']);pred['predicted_bill_won']=pred.predicted_kwh*180;pred['predicted_carbon_kg']=pred.predicted_kwh*.424
    pred.to_csv(dest/'selected_full745_forecast.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    f=next(f for f in allpred if f.model.iloc[0]==lock['selected_model'])
    submission=pd.DataFrame(dict(id=f.id,hourly_pow=f.predicted_kwh,may_bill=f.predicted_bill_won,may_carbon=f.predicted_carbon_kg,
      agg_pow=np.repeat(f.predicted_kwh.sum(),505)))
    submission.to_csv(dest/'selected_guide505_forecast.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    # Paired circular24h-block bootstrap for uncertainty, without selection changes.
    # Overlapping validation labels are not treated as1515 independent observations.
    base=next(f for f in allpred if f.model.iloc[0]=='hourly_baseline');rng=np.random.default_rng(20261013)
    uncertainty=[]
    for key,split,ff,bb in [('daily_chronos2','may',f,base)]:
        delta=np.abs(ff.predicted_kwh.to_numpy()-ff.actual_kwh.to_numpy())-np.abs(bb.predicted_kwh.to_numpy()-bb.actual_kwh.to_numpy())
        samples=[];n=len(delta)
        for _ in range(2000):
            ix=((rng.integers(0,n,size=int(np.ceil(n/24)))[:,None]+np.arange(24))%n).ravel()[:n];samples.append(delta[ix].mean())
        uncertainty.append(dict(model=key,baseline='hourly_baseline',split=split,mae_difference_kwh=float(delta.mean()),
          circular_block_hours=24,bootstrap_resamples=2000,ci95_kwh=np.quantile(samples,[.025,.975]).tolist(),
          interpretation='Descriptive uncertainty of locked predictions only; not proof of prospective superiority or a tuning criterion'))
    result=dict(created_utc=now(),evaluation_started_utc=evaluation_started,global_selection_lock_sha256=sha(OUT/'global_selection_lock.json'),
      may_source_sha256=sha(source),selected_model=lock['selected_model'],observed_best_may_model=min(cards,key=lambda c:c['may']['mae_kwh'])['key'],
      selection_changed_after_may=False,may_points=505,forecast_points=745,may_leads='241..745h from last completed observation',
      native_evaluation_coverage=dict(daily_complete=21,weekly_complete=2,monthly_complete=0),
      mean_actual_may_kwh=float(may.hourly_pow.mean()),mean_train_kwh=float(load_hourly_target().mean()),
      models=cards,paired_uncertainty=uncertainty,
      target_limit='Guide provisional kWh; unresolved meter hierarchy/units; fixed origin without May target/sensor updates')
    write(OUT/'results.json',result);return result,allpred

def plots(result,allpred):
    os.environ['MPLCONFIGDIR']=str(OUT/'tmp/matplotlib');sys.path.insert(0,str(ROOT/'output/analysis/.plot_deps'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    fp=FontProperties(fname='C:/Windows/Fonts/malgun.ttf');plt.rcParams['font.family']=fp.get_name();plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['axes.formatter.useoffset']=False
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    for c in result['models']:
        c['figures']={}
        for split in ['validation','test','may']:
            f=next(f for f in allpred if f.model.iloc[0]==c['key']) if split=='may' else frame(c['paths'][split])
            if split=='validation':f=f[pd.to_datetime(f.origin)==pd.Timestamp('2025-02-25')]
            f=f[f.horizon_offset_hours.between(240,744)]
            fig,axs=plt.subplots(3,1,figsize=(11.5,7),gridspec_kw={'height_ratios':[2,1.6,1]},layout='constrained')
            for a,z,title in [(axs[0],f,'평가 505시간 전체'),(axs[1],f.iloc[:72],'평가 시작 후 72시간 확대')]:
                a.plot(z.id,z.actual_kwh,label='실제',color='#334155',lw=.9);a.plot(z.id,z.predicted_kwh,label='예측',color='#dc2626',lw=1.4)
                a.set_title(title,fontsize=10);a.set_ylabel('시간 전력량 (kWh)');a.grid(alpha=.25);a.legend(loc='upper right',fontsize=8)
            axs[2].plot(f.id,f.predicted_kwh-f.actual_kwh,color='#7c3aed',lw=.9);axs[2].axhline(0,color='black',lw=.7);axs[2].set_ylabel('예측 - 실제');axs[2].grid(alpha=.25)
            met=metrics(f.actual_kwh,f.predicted_kwh)
            fig.suptitle(f"{c['label']} | {split} | MAE {met['mae_kwh']:.4f} kWh | R² {met['r2']:.5f}",fontsize=12)
            for a in axs:a.tick_params(axis='x',labelsize=8)
            target=dest/f"{c['key']}_{split}.png";fig.savefig(target,dpi=120);plt.close(fig);c['figures'][split]=rel(target)
    ordered=sorted(result['models'],key=lambda c:c['val']['mae_kwh']);fig,axs=plt.subplots(1,2,figsize=(12,6),layout='constrained')
    for a,split,title in [(axs[0],'val','검증: 모델 선택에 사용'),(axs[1],'may','5월: 선택 이후 평가')]:
        a.barh([c['label']for c in ordered],[c[split]['mae_kwh']for c in ordered],color=['#dc2626'if c['key']==result['selected_model'] else '#2563eb' for c in ordered]);a.invert_yaxis();a.set_xlabel('MAE (kWh / 시간)');a.set_title(title);a.grid(axis='x',alpha=.2)
    fig.savefig(dest/'comparison.png',dpi=130);plt.close(fig)
    y=load_hourly_target();fig,axs=plt.subplots(4,1,figsize=(11,8),layout='constrained')
    sys.path.insert(0,str(OUT));from resample_common import periods
    for a,freq,title in zip(axs,['H','D','W-SUN','MS'],['시간 3,600개','일 150개','완전한 주 21개','완전한 월 4개']):
        g=periods(y,freq);a.plot(g.index,g.rate_kwh_per_hour,'o-',ms=2,lw=.8);a.set_title(title,fontsize=10);a.set_ylabel('평균 부하율');a.grid(alpha=.2)
    fig.suptitle('기간 길이를 보정한 부하율: 모든 축은 kWh/시간');fig.savefig(dest/'resampled_history.png',dpi=120);plt.close(fig)
    write(OUT/'results.json',result)

def report(result,lock):
    cards=sorted(result['models'],key=lambda c:c['val']['mae_kwh']);selected=next(c for c in cards if c['key']==result['selected_model']);best=next(c for c in cards if c['key']==result['observed_best_may_model']);base=next(c for c in cards if c['key']=='hourly_baseline')
    esc=html.escape
    rows=''.join(f"<tr class={'selected'if c['key']==selected['key']else ''}><td>{esc(c['label'])}{' <b>검증 선택</b>'if c['key']==selected['key']else ''}</td><td>{c['val']['mae_kwh']:.5f}</td><td>{c['test']['mae_kwh']:.5f}</td><td>{c['may']['mae_kwh']:.5f}</td><td>{c['may']['r2']:.6f}</td><td>{c['may']['bias_pred_minus_actual_kwh']:.3f}</td><td><a href='{c['may_csv']}'>CSV</a> · <a href='{c['source']}'>설정</a></td></tr>"for c in cards)
    options=''.join(f"<option value='{c['key']}'>{esc(c['label'])}</option>"for c in cards)
    data=json.dumps({c['key']:dict(figures=c['figures'],config=c['config'],scope=c['scope'])for c in cards},ensure_ascii=False).replace('</','<\\/')
    u=result['paired_uncertainty'][0];gain=base['val']['mae_kwh']-selected['val']['mae_kwh']
    document=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>시간·일·주·월 리샘플링 예측 실험</title>
<style>body{{font:16px/1.65 "Malgun Gothic",sans-serif;color:#172033;background:#f8fafc;margin:0}}main{{max-width:1180px;margin:auto;padding:28px}}h1{{font-size:29px}}h2{{font-size:22px;margin-top:36px}}.box{{background:white;border:1px solid #cbd5e1;border-radius:10px;padding:20px;margin:18px 0}}.notice{{border-left:5px solid #d97706}}.table{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px;border-bottom:1px solid #e2e8f0;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}.selected{{background:#fff1f2}}a{{color:#1d4ed8}}img{{max-width:100%;height:auto}}select{{font:inherit;max-width:100%;padding:8px;margin:6px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f9;padding:12px;font-size:13px}}.small{{font-size:14px;color:#475569}}</style>
<main><h1>시간·일·주·월 리샘플링 기반 미래 전력량 예측</h1><p>프로젝트 가이드의 미래 전력량·요금·탄소 예측을 유지하여 {lock['native_candidate_count']}개 설정을 검증하고, 선택된 17개 모델을 같은 5월 505시간에서 평가했습니다.</p>
<div class="box"><b>검증 기준 선택: {esc(selected['label'])}</b><p>일별 Chronos-2 예측과 과거 평균의 혼합(Chronos 25%, 평균 75%)입니다. 3개 원점 검증 MAE {selected['val']['mae_kwh']:.5f} kWh, 시간 평균 기준선 대비 개선 {gain:.5f} kWh에 그칩니다. 검증상 최저 오차이며, 확실한 성능 우위를 입증한 결과는 아닙니다.</p><p>선택 모델의 5월 MAE <b>{selected['may']['mae_kwh']:.4f} kWh</b>, R² <b>{selected['may']['r2']:.5f}</b>. 5월에서 관측된 최저 MAE는 {esc(best['label'])}의 {best['may']['mae_kwh']:.4f} kWh였지만 이 결과로 모델을 다시 선택하지 않았습니다.</p></div>
<div class="box notice"><b>결론: 리샘플링으로 유의미한 시간별 변동 예측 개선을 확인하지 못했습니다.</b><p>시간별 부하를 평균 수준으로 추정하는 능력과 실제 시간별 변동을 예측하는 능력은 다릅니다. 작은 MAPE를 정확도 99%라는 식으로 해석할 수 없습니다. 현재 자료에서는 단순 기준선을 함께 유지하는 것이 합리적이며, 높은 용량 모델 또는 월별 학습 모델의 신뢰성을 주장할 근거가 부족합니다.</p></div>
<h2>1. 데이터와 전처리</h2><div class="table"><table><tr><th>단위</th><th>전체 버킷</th><th>완료된 학습 버킷</th><th>첫 검증 원점의 학습 버킷</th><th>산출 데이터</th></tr><tr><td>시간 H</td><td>3,600</td><td>3,600</td><td>1,824</td><td><a href="data/hourly_dataset.csv">시간 CSV</a></td></tr><tr><td>일 D</td><td>150</td><td>150</td><td>76</td><td><a href="data/daily_dataset.csv">일 CSV</a></td></tr><tr><td>주 W-SUN</td><td>23</td><td>21</td><td>10</td><td><a href="data/weekly_dataset.csv">주 CSV</a></td></tr><tr><td>월 MS</td><td>5</td><td>4</td><td>2</td><td><a href="data/monthly_dataset.csv">월 CSV</a></td></tr></table></div>
<p>원본 5초 간격 전력(kW)을 전력량 = Σ P × 5/3600으로 적분하고 13개 모듈을 합쳤습니다. 완전히 관측한 시간만 사용하며, 학습 범위는 2024-12-01 00:00부터 2025-04-29 23:00까지입니다. 4월 30일 00:00의 단일 원본 시점은 완료된 1시간으로 취급하지 않습니다.</p>
<p>일·주·월의 전력량 합계와 함께 관측 시간 수, 기대 시간 수, coverage, complete, available_at을 저장했습니다. 학습 목표는 기간 평균 부하율(kWh/시간)로 만들어 월 길이 차이를 보정했습니다. 완전하지 않은 주 2개와 4월 29일까지만 있는 4월 월간 버킷은 주·월 모델 학습에서 제외했습니다. 주는 월요일 00:00에, 월은 다음 달 1일 00:00에 이용할 수 있습니다.</p>
<p>시간 단위 XGBoost·Ridge 입력은 과거 부하의 lag1/24/168, 이동 평균·표준편차, 예측 거리와 달력입니다. 완료된 과거 일·주·월 요약을 추가하는 H→HD→HDW→HDWM 비교도 수행했습니다. 검증에서 XGBoost는 과거 시간 이력만 사용하는 설정이, Ridge는 HD가 선택되었습니다. 미래 센서 실제값 또는 미래 집계 총량은 입력하지 않았습니다. Ridge와 신경망의 scaler, 과거 시간 배분 패턴은 각 학습 구간 안에서 계산했습니다. <a href="hourly/ablation.csv">전체 피처 조합 비교표</a></p>
<img src="hourly/feature_ablation.png" alt="시간 모델에 과거 일·주·월 요약 피처를 추가한 검증 비교">
<img src="figures/resampled_history.png" alt="시간·일·주·월의 부하율 추세"><p class="small">집계에 따라 흔들림이 작아져도 예측 정보가 새로 생기는 것은 아닙니다. 월 데이터 4개는 고용량 모델 학습에 부족합니다. 원본 단위와 13개 계측기의 포함 관계가 미확인되어 가이드 정의에 따른 잠정 전력량을 사용했습니다.</p>
<h2>2. 예측 범위와 공정한 평가</h2><p>최종 입력은 2025-04-29 23:00의 완료된 시간까지입니다. 4월 30일 00:00부터 5월 31일 00:00까지 <b>745시간을 고정 원점에서 예측</b>했습니다. 가이드의 정답 데이터는 5월 10일 00:00부터 시작하므로, 마지막 관측 시간 기준 <b>241~745시간 뒤의 505개 값</b>을 평가했습니다. 해당 시각은 1시간 버킷의 시작 시각입니다. 예측 중 5월의 실제 전력량이나 센서로 갱신하지 않았습니다.</p>
<p>모델 선택은 2월 15·22·25일의 3개 원점에서 각각 745시간을 예측하고 마지막 505시간의 MAE를 평균한 기준입니다. 선택 이후 별도로 3월 29일 원점의 마지막 505시간(4월 8일~29일 00:00)을 평가했습니다. 검증의 최종 라벨은 3월 28일 00:00이며 내부 평가와 겹치지 않습니다. 단, 검증 원점끼리 중복된 라벨이 있어 1,515개 평가 쌍의 고유 시각은 745개입니다. 3개의 독립 실험이라는 주장은 하지 않습니다.</p>
<p>일별 모델은 32일 부하율을 예측하고 과거 배분 패턴으로 시간값으로 변환했습니다. 주·월 모델도 각 달력 기간의 부하율을 시간값으로 배분합니다. 검증 결과 배분 패턴은 균등 배분이 선택되었습니다. 각 모델은 자기 예측 기간 평균을 사용하며 실제 미래 총합에 맞추지 않았습니다. H/D/W/M 간 계층 합계 일치 제약을 공동 학습한 모델은 아닙니다.</p>
<div class="box notice">이번 실험의 모델 선택과 학습에서는 5월 라벨을 사용하지 않았습니다. 다만 이전 분석에서 4·5월 자료를 살펴본 이력이 있어 완전히 새로운 맹검 실험으로 해석할 수 없습니다. 주·월 모델은 표본 부족에 맞춘 ARIMA 또는 수준 기준선입니다.</div>
<h2>3. 동일한 505시간에서의 모델별 결과</h2><p>MAE의 단위는 모두 시간 전력량 kWh이며 기간이 다른 오차를 직접 섞지 않았습니다. 내부·5월 결과는 설정 선택에 사용하지 않았습니다.</p><div class="table"><table><tr><th>모델</th><th>검증 MAE</th><th>내부 MAE</th><th>5월 MAE</th><th>5월 R²</th><th>5월 편향</th><th>자료</th></tr>{rows}</table></div><img src="figures/comparison.png" alt="검증과 5월 MAE 비교">
<p>5월 평균 실제값은 {result['mean_actual_may_kwh']:.3f} kWh이며 전체 학습 평균 {result['mean_train_kwh']:.3f} kWh보다 낮습니다. 현재 고정 원점 모델은 이 수준 차이를 잘 반영하지 못했습니다. R²가 음수인 결과는 5월 실제 평균을 사용하는 사후 상수 예측보다 제곱오차가 컸다는 뜻입니다. 해당 실제 평균은 예측 당시 사용할 수 없으므로 서비스 기준선으로 입력하지 않았습니다.</p>
<p>잠금된 선택 모델과 시간 기준선의 5월 MAE 차이(모델 - 기준선)는 {u['mae_difference_kwh']:.4f} kWh입니다. 24시간 블록을 이용한 기술적 bootstrap 95% 구간은 [{u['ci95_kwh'][0]:.4f}, {u['ci95_kwh'][1]:.4f}] kWh입니다. 이는 이 평가 구간의 불확실성 참고치이며 미래 데이터의 우위나 전역 최적성을 입증하지 않습니다.</p>
<p>원래 기간 단위의 합계 평가도 <a href="evaluation/native_aggregate_evaluation.csv">별도 CSV</a>에 저장했습니다. 5월 정답으로 완전한 일은 21개, 주는 2개이며 완전한 월은 없습니다. 따라서 월 총량의 정확도 또는 월별 모델 신뢰도를 이번 5월 자료로 검증했다고 주장할 수 없습니다.</p>
<h2>4. 모델별 실제값·예측값과 오차</h2><p>아래 메뉴로 17개 모델의 실제값, 예측값, 시작 72시간 확대, 잔차를 확인할 수 있습니다. 검증 그림은 마지막 원점(2월 25일)이며 표의 검증 점수는 3개 원점 전체입니다.</p><label>모델 <select id="model">{options}</select></label><label>평가 <select id="split"><option value="may">5월 505시간</option><option value="test">내부 평가 505시간</option><option value="validation">2월 25일 검증 505시간</option></select></label><img id="chart" alt="선택한 모델의 실제값·예측값·잔차"><pre id="config"></pre>
<h2>5. 연구 모델·학습 한계</h2><p><a href="https://arxiv.org/abs/2211.14730">PatchTST</a>와 <a href="https://arxiv.org/abs/2310.06625">iTransformer</a>의 공개 모델 코어를 사용해 H/D 모델을 학습했습니다. 유사한 산업 계측 공개 데이터인 <a href="https://www.energystatusdata.kit.edu/hipe.php">HIPE</a>를 참조했지만 동일 데이터셋의 절대 SOTA를 재현한 것은 아닙니다. 일별 신경망의 첫 내부 학습은 44일뿐이며 context7+예측32로 만들 수 있는 중복 학습 창은 6개입니다. 일부 iTransformer 후보는 epoch24 상한에서 최저 손실이 나와 미수렴 가능성이 있습니다.</p><p><a href="https://github.com/amazon-science/chronos-forecasting">Chronos-2 공식 고정 체크포인트</a>는 가중치 미세조정 없이 문맥 길이·평균 혼합 비율만 검증으로 선택했습니다. 시간 Chronos는 혼합 alpha0으로 단순 평균과 동일한 결과입니다. 일별 Chronos alpha0.25도 확실한 우위를 보이지 않았습니다. ARIMA와 XGBoost를 함께 학습했으며 제한된 후보 범위의 최저 점수만 보고했습니다.</p>
<h2>6. 결과와 재현 파일</h2><p><a href="evaluation/selected_guide505_forecast.csv">선택 모델: 가이드 505시간 전력량·요금·탄소 CSV</a> · <a href="evaluation/selected_full745_forecast.csv">선택 모델 전체 745시간 예측</a> · <a href="evaluation/model_comparison.csv">모델 비교표</a> · <a href="evaluation/all_model_may_predictions.csv">전체 모델 실제값·예측값</a> · <a href="global_selection_lock.json">5월 평가 전 선택 잠금</a> · <a href="verification.json">통합 검증</a> · <a href="independent_audit.json">독립 감사</a> · <a href="README.txt">재현 명령</a></p>
<p class="small">요금은 예측 전력량×180원/kWh, 탄소는 ×0.424kgCO₂/kWh인 가이드 가정입니다. 별도 예측 모델의 검증 점수로 계산하지 않았습니다. 제출 CSV agg_pow는 제공 정답 형식에 맞춘 505시간 합계이며 완전한 5월 총량을 의미하지 않습니다. 원본과 기존 분석 결과는 보존했습니다.</p></main>
<script>const data={data};const m=document.getElementById('model'),s=document.getElementById('split');m.value={json.dumps(selected['key'])};function update(){{document.getElementById('chart').src=data[m.value].figures[s.value];document.getElementById('config').textContent='선택 설정: '+JSON.stringify(data[m.value].config,null,2)+'\\n입력: '+JSON.stringify(data[m.value].scope);}}m.addEventListener('change',update);s.addEventListener('change',update);update();</script></html>'''
    (OUT/'resampling_report.html').write_text(document,encoding='utf-8')

def main():
    cards=model_cards();lock=freeze(cards);print('FROZEN BEFORE MAY',lock['selected_model'],lock['native_candidate_count'],flush=True)
    result,preds=evaluate(cards,lock);plots(result,preds);report(result,lock)
    print(json.dumps(dict(selected=result['selected_model'],observed_best=result['observed_best_may_model'],models=len(cards),paired=result['paired_uncertainty']),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
