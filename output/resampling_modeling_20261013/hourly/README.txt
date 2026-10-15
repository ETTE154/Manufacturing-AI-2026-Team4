시간별 목표값 + 완료 H/D/W/M 문맥을 이용한745시간 고정원점 예측

새 파일은 이 hourly 폴더에만 저장했다. 원본과 기존 analysis/modeling/optimization/feature_revision 산출물은 보존했다.
3600개 complete-hour 목표값은 Dec1 00~Apr29 23이며 13개 모듈의 잠정 에너지 합이다.
5초 전력의1시간적분 단위는 가이드 기준kWh지만, 모듈 계측 계층은 불확실하다.

예측 경로: 원점부터 offset0..744의745개 시간값. 마지막 완료 시간key에 대한 lead1..745다.
주선정 지표: offsets240..744의 후반505시간 MAE를 세 원점별로 계산한 뒤 동일 가중 평균한다.
전745시간 MAE는 보조값이다. 검증 원점은Feb15/Feb22/Feb25(1824/1992/2064시간history).
처음 제안된Feb28은 성능평가 전에 거부했다. 마지막 검증labelMarch28 00 < 내부원점March29이며 중첩0이다.
세 검증의505시간 구간끼리는 겹친다:1515평가쌍이지만 unique label시간은745개뿐이다.
내부 시험은Mar29 00~Apr29 00(P745), 후반505는Apr8 00~Apr29 00다. 시험으로 추가선택하지 않는다.
최종 forecast는Apr30 00~May31 00(P745)이며 actual은NaN이다. May자료는 이 담당자가 읽지 않는다.

사전선언 후보40개: XGB13, Ridge4, ARIMA13, baseline10. 이전 검증선택 XGB history-only설정을 포함했다.
동일 설정으로H/HD/HDW/HDWM feature-group ablation을 비교했다.
H: target lag1/24/168, mean24/168/720, std24/168, 원점hour, knownfuturecalendar/lead.
D: 완료일 rate의7일/28일평균, 마지막완료일 minus7일평균, 지원수/누락flag.
W: 마지막완료주 rate 및최근4주평균, 지원수/누락flag. 주간은Monday00~nextMonday00이며Sunday표시label이다.
M: 마지막완료월rate/energy, 최근3완료월rate평균, 완료수/누락/3월미만지원flag.
known_partial_month_observed_hours는 예측원점이 속한 달에서 이미관측한시간수다.
이 값은 원점에 고정되어745future hours동안갱신하지 않고 미래달의 측정값을 뜻하지 않는다.
H_previous_history_only는 이전 검증선택의11개 과거피처+lead만사용하며 targetcalendar와D/W/M이없다.

각 학습origin context는 해당prefix만 resample하며 period.available_at=endexclusive<=origin인 완료구간만쓴다.
시작부분weeklyfragment, 현재부분일/주/월 및 마지막불완전구간은 완료period피처에서제외한다.
초기학습에완료month/week가없으면 그 과거prefix평균으로대체하고명시적인flag를남긴다.
세 검증원점의완료월은각2개,최종원점은4개다. M은제한된과거문맥후보이며 학습된월별모델이아니다.
원자료센서403개와미래sensor/periodactual은입력하지않는다. hourly직접모델에는profile배분weight가없다.

XGB/Ridge는stride6의과거원점과offset0..24,48..744(by24)을stack한direct잔차모델이다.
target index<=fittingprefix end로필터하며 terminalorigin의부분horizon은허용한다.
Ridge StandardScaler는해당prefix의direct training examples에만fit한다. XGB/ARIMA에는global scaler가없다.
residualcenter는반복된direct training labels의평균이므로 원시history평균과조금다를수있다.
future입력은 원점context에knowncalendar/lead를붙여 생성한다. 미래actual을rolling입력으로재사용하지않는다.

검증선택: 최근1440시간평균 baseline76.23463796, XGB76.23593561, ARIMA76.23714199, Ridge76.29280679kWh.
XGB는이전H-only configuration, Ridge는HD가선택됐다. 월/주피처의뚜렷한개선근거는없다.
선택된baseline와XGB의차이는0.00130kWh이고여러겹치는검증구간에의존하므로 의미있는우위로해석하지않는다.
internalLate505 MAE는baseline77.949361,ARIMA77.997246,XGB78.006886,Ridge78.056070kWh.
선택을내부시험이나May결과로변경하지않는다. 이전작업이April/May를본post-EDA실험이다.

results.json: family별selectedconfig, validation_late505/full745(전체 및원점별지표),internaltest,paths,models,figures.
ablation.csv /feature_ablation.PNG,SVG: 모든feature-group/설정의검증결과.
temporal_audit.json /split_audit.json /verification.json: 미래오염,완료period,분할중첩,CSV지표,재로드,CLI 및동결hash.
saved_model/model.joblib: nativeestimator와원점의완료resamplingcontext를저장하여데이터파일없이정확재현한다.
XGB는model_native.json,ARIMA는model_statsmodels.pkl도저장했다. native/전체reload최대차0이다.
동결후모델/선정파일은추가변경하지않는다. 부모가모든familyglobal lock후공통May505평가를실행한다.

환경: E:\AI work\AWS_프로젝트\output\modeling\.venv\Scripts\python.exe; seed20261013; CPUthreads4.
재현학습: train_hourly.py --stage search, 이후 --stage evaluate (기존파일위에반복실행하지않는다).
단독추론: predict_hourly.py --model selected|xgboost|ridge|arima|baseline --output <이hourly폴더내CSV절대경로>
추론CLI는저장된Apr30원점만재현한다. 현장이력자동갱신API는구현하지않았다.
