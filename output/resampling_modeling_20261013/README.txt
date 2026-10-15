시간·일·주·월 리샘플링 기반 프로젝트 가이드 예측 실험

보고서: resampling_report.html
공통 결과: results.json / evaluation/model_comparison.csv
검증으로 선택: daily_chronos2 (Chronos25% + 과거 평균75%)
5월에서 관측된 최저 모델은 별도로 보고하며 재선택하지 않습니다.

Python 실행 환경 (원래 패키지 환경 보존):
E:\AI work\AWS_프로젝트\output\modeling\.venv\Scripts\python.exe

동결된 저장 모델의 예측 재현 (May 원본 파일을 읽지 않음):
& 'E:\AI work\AWS_프로젝트\output\modeling\.venv\Scripts\python.exe' 'E:\AI work\AWS_프로젝트\output\resampling_modeling_20261013\predict_saved.py' --model selected --output 'E:\AI work\AWS_프로젝트\output\resampling_modeling_20261013\replay\selected745.csv'

--model은 global_selection_lock.json의 rankings에 있는 17개 모델 이름을 받습니다.
Chronos는 기본적으로 공식 로컬 체크포인트 forward를 다시 수행합니다.
--cached-chronos를 지정하면 해시가 확인된 기존 raw 예측과 동결 혼합 설정을 사용합니다.
모든 예측 CLI는 저장한 4월30일 원점 재실행용입니다. 실시간 신규 센서 입력/재학습 서비스는 아닙니다.
neural/neural_resampling.py는 신경망에 한해 --history CSV(시간/전력량) 입력도 제공합니다.

프로젝트 가이드 형식:
evaluation/selected_guide505_forecast.csv: id,hourly_pow,may_bill,may_carbon,agg_pow
agg_pow는 제공 정답 형식과 일치하는505시간 합계입니다. 완전한5월 합계가 아닙니다.
요금180원/kWh와 탄소0.424kgCO2/kWh는 가이드의 상수 가정입니다.
evaluation/selected_full745_forecast.csv는 4월30일00시~5월31일00시 전체 경로입니다.

개발/재평가 소스 (읽고 검토한 자체 작성 파일):
resample_common.py: 원점별 prefix·집계·기간 완료/이용가능 시각 계약
coarse/coarse_models.py: 일·주·월76개 후보
hourly/train_hourly.py: 시간 이력/완료된D/W/M 피처40개 후보
hourly/finalize_hourly.py: 동결 설정의 내부 평가/최종 학습
neural/neural_resampling.py: H/D PatchTST·iTransformer9개 후보
neural/chronos_resampling.py: H/D 고정 Chronos 각6개 후보
integrate_report.py: 선택 lock 작성 후 May 비교/시각화/보고서

현재 결과를 보존하려면 학습 스크립트를 다시 실행하지 마십시오.
재실험은 별도 출력 폴더·새 protocol로 실행해야 기존 동결 파일이 유지됩니다.
integrate_report.py는 기존 global lock의모델·설정·미래경로 해시와 달라지면 중단합니다.

검증 기간: Feb15/Feb22/Feb25 원점; 각각 마지막505시간 MAE 평균.
선택 라벨은Mar28 00시까지이며 Mar29 원점의 내부 평가와 겹치지 않습니다.
검증 원점들끼리는 라벨 중첩이 있습니다.1515쌍의 고유 시각은745개입니다.
미래 실제 센서/전력량/일·주·월 총합을 입력하지 않았습니다.
이전 April/May EDA가 있었으므로 완전히 새로운 맹검 평가로 주장하지 않습니다.
월 모델은 완료된4개월(검증 때2개월)로 만든 투명한 수준 기준선입니다.
고용량 신경망의 일별 첫 내부학습 창6개와 epoch cap을 신뢰성 한계로 보고했습니다.
원본 센서/전력량 단위와13개 모듈의 포함 관계는 아직 확인되지 않았습니다.
원본 데이터, 이전 결과, 프로젝트 레지스트리는 수정하지 않았습니다.
