실제 시계열 분해와 원자료의 주기성 근거

검증된 complete_hour_module_sum.csv의3600개 시간값을 읽기만 했다.
원본 SHA는 실행 전후 동일하며, 13개 모듈 합계의 단위/계측계층은 잠정 가이드 가정을 유지한다.
May자료를 읽거나 원래 합계파일을 다시 작성하지 않았다.

고정 설정으로 수행한 실제 분해:
- hourly_stl24: H3600개, period24, seasonal13,robustTrue,라이브러리 기본trend41/lowpass25.
- hourly_stl168: H3600개, period168,seasonal13,robustTrue,기본trend285/lowpass169.
- daily_stl7: 완료일150개 기간 평균 부하=일전력량/24관측시간(kWh/h),period7,seasonal13,robustTrue.
- hourly_mstl24_168: 공식statsmodels0.15 MSTL,periods24/168,windows11/15,iterate2,robustTrue,변환없음.
성능을 보며 smoother설정/주기를 고르는 탐색은 하지 않았다.
각 components.csv에는 observed/trend/seasonal/resid 및 robust_weight를 저장했고
MSTL에는 seasonal_24/seasonal_168, D에는 observed_daily_energy_kwh도 포함했다.
각 case의4panel PNG/SVG와 MSTL개별계절그림, H/D원자료진단 및 강도/스케일그림은 standalone파일이다.

H y축은1시간전력량(kWh,잠정), D y축은기간평균부하(kWh/h)다.
정격용량 대비 부하율(%)을 계산하지 않았다. 계절SD/평균(%)는 관측평균에 대한 변동규모다.
분해 계절강도는H24=.146654,H168=.143848,D7=.192973,MSTLcombined=.232672다.
이 수치는 분해 후 남은 분산의 기술적 비율이며 유의성이나 예측R2/정확도가 아니다.
componentvariance는 상관될 수 있으므로 단순 shares합을100%로 강요하지 않고 covariance identity를 기록했다.

실제 하루/한주 반복 근거는 약하다:
원자료 phase-mean between-variance 검정1999순열 p는 H24=.9845,H168=.5885,D7=.739이고
세 사전 지정검정의 Holm보정p는 모두1.0이다. ACF95%구간은 모두0을 포함한다.
H LjungBox24/168p=.3971/.7376, D7p=.9030이다. 비기각은 독립성의 증명이 아니다.
periodogram최대peak는H5.405시간이지만 전체frequency최대값 순열p=.0655이며 D최대peakp=.71이다.
168시간과7일 주기의 가까운 Fourierbin은 실제로171.429시간/7.143일이므로 exact주기라고 부르지 않는다.

똑같은 STL설정을 시간순서만 섞은19개 자료에도 적용했다.
H24/168의계절강도90%참고범위는[.1231,.1544]/[.1351,.1724]이며 실제강도가그안에있다.
D7의[.0395,.1956]에도 실제강도.1930이포함된다. 이19개 reference는정밀한꼬리확률/신뢰구간이아니다.
STL은주기를지정하면무작위변동에서도계절곡선을만들수있으므로 시각적모양만으로실제계절성을주장하지않는다.
순열검정은귀무가설의exchangeability를가정하며 강한시간의존이있으면단순shuffle이부적절할수있다.

주간완료관측21개,월간완료관측4개뿐이므로 각각52주/12월 annual분해를임의로수행하지않았다.
Dec1부분주,Apr28~29부분주 및April부분월은 완료period자료에서제외했다.

전체Dec-Apr 시계열의STL/MSTL은 각시점뒤의값을 사용하는 retrospective smoothing이다.
따라서 추출된 trend/seasonal/resid를 과거원점 예측모델 피처로 그대로 넣으면 미래정보가누출된다.
향후 예측용 분해를할때에는 각훈련prefix에서만 다시분해하고 시간순검증을 해야한다.
본분해는 EDA/PDF설명만을위한기술분석이다.

results.json: cases.{hourly_stl24,hourly_stl168,daily_stl7,hourly_mstl24_168}, 그림/CSV,
strength/std/variance/covariance,raw_periodicity,short_series_limits,해석과공식references.
verification.json: 모든componentCSV의재구성오차<1e-8,강도재계산차이0,원본SHA유지pass.
공식statsmodels STL/MSTL/ACF/LjungBox, SciPy periodogram 및 FPP3 strength 출처는results.references에있다.
실행: E:\AI work\AWS_프로젝트\output\modeling\.venv\Scripts\python.exe decompose.py
고정seed20261013,CPUthreads4,statsmodels0.15.0. 모든새산출물은이폴더에만있다.
