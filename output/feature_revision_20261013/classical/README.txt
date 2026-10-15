새 피처 재검증 및 전통 모델 실험

이 폴더만 새로 작성했으며 원본, 기존 analysis/modeling/optimization 산출물은 보존했다.
시간 축은 검증된 recorded localtime이며 5초 원자료를 1시간으로 집계한 3600개 완전 관측 시간이다.
타깃은 13개 모듈의 잠정 에너지 합(kWh)이다. 계측 계층이 불명확하여 실제 공장 소비로 확정하지 않는다.

previous_feature_audit.json: 이전 훈련행-추론행의 독립 비교 차이0, 미래변조 영향0, 시간 정렬 버그 없음.
이전의 원점 고정 lag24는 곡선 전체에서 같은 값인 설계였다. 새 day 모델에는 각 target hour에 대응하는
전날 및 전주 같은 시간의 target/module/sensor history를 포함했다. 이 값은24개 target 모두 원점 이전이다.
long745에서는 이를 미래 lag로 쓰지 않는다. 마지막 관측 하루/한 주 안의 같은 phase를 반복 참조한다.

주 실험 후보24개: XGB10, Ridge6, ElasticNet2, ARIMAX3, baseline3.
동일한5개의14일 expanding-fold(70 daily origins)에서24-value path MAE로 선택했다.
센서 선택은 각 fitting prefix 내에서 BH-FDR, 초기/후기 안정성 및 중복 제거를 수행했다.
선택 센서 수는0/2/0/1/0이며 March31까지 최종 schema는0개다. 선택0은 정상 결과다.
all-sensor 후보는403개 센서의806개 과거 lag feature를 쓰는 사전 선언 ablation이며,
통계 screening 통과 센서만 쓰는 selected-sensor 후보와 구분한다.
Ridge/ElasticNet/ARIMAX scaler는 fitting prefix에만 fit한다. XGB는 스케일러 없이 잔차 target을 학습한다.
Origin weights/schema는 각14일 검증 블록 및29일 April test 동안 고정이고 과거 context만 매일 갱신한다.
ARIMAX는 새로 완료된 관측을 filtering state에 반영하지만 parameter를 재학습하지 않는다.

results.json: 선택 설정, validation/test24path 및 lead1/6/24 지표, 선정 schema, 예측CSV, 저장모델, 그림.
feature_group_ablation.csv: 동일 외부검증 구간에서 사전선언 feature group/규제 후보 비교.
모델별 chronological_group_permutation.csv와 chronological_feature_permutation.csv:
March25~31의 시간순 inner holdout. March24까지 fit 및 sensor selection을 했고,
학습 중요도로만 top20을 정한 뒤24시간 block permutation을5번 수행했다. 이는 진단이며 선택에 사용하지 않는다.
모델별 coefficients_final.csv 또는 gain_final.csv는 최종 저장모델의 계수/훈련 gain이다.
상관과 중요도는 인과성, 현장 일반화 또는 확실한 개선의 근거로 해석하지 않는다.

과거 검증의 전통 모델 최저는 EMA720, MAE74.67748973이다.
4월 XGB MAE79.29292207은 기존 Chronos+mean 혼합79.34035982보다 약0.04744kWh 낮지만,
paired-day bootstrap95%구간[-0.28964,+0.18969]은0을 포함한다. 이 관측으로 선택을 바꾸지 않는다.
기존 Chronos 혼합의 과거 검증74.61737784는 새 전통 모델보다 좋다.
이전 작업에서 April/May를 이미 봤으므로 신규 실험은 post-EDA이며 prospective blind validation이 아니다.

long745/: 보조 XGB/Ridge4개 후보만 고정 원점 장기검증에서 선택했다.
저장 모델의745행 미래 경로는 Apr30 00부터 May31 00까지이며 actual은 모두NaN이다.
May 원본 파일은 이 재학습 담당자가 다시 읽지 않았다. 부모가 모든 모델 글로벌 lock 후 공통505label 평가를 완료했다.
결과는 상위 comparison_results.json 및 may_predictions/에 있으며, 새 XGB MAE88.26876366, Ridge88.61314989다.
설정 선택은 바꾸지 않았고 기존 Chronos를 유지한다. 담당 helper는 중복 실행하지 않았다.
선택 기준은 해당4개 후보 안의 최저이며 이전 Chronos/mean의 배포 선택을 대체하지 않는다.
May에는 미래 물리 센서 및 Apr30-May9 context가 없어 rolling24h 예측을 실증할 수 없다.

검증: verification.json 및 long745/verification.json. CSV 지표 허용차1e-7,
저장모델/저장된 과거 문맥 단독 재현 허용차1e-8; 실제 재로드 최대차0.0.
기본 inference는 Apr29 23까지 학습한 저장모델과 저장 문맥을 재사용하며 May label을 읽지 않는다.
전체 이력 EMA 상태는 frozen_ema_value로 저장하여 tail720만으로 재계산하는 오류를 피한다.

실행 환경: E:\AI work\AWS_프로젝트\output\modeling\.venv\Scripts\python.exe
학습: retrain_classical.py --stage search, 이후 --stage test
추론: predict_classical.py --scenario dayahead24 --model selected --output <이폴더내CSV절대경로>
장기: predict_classical.py --scenario long745 --model selected --output <이폴더내CSV절대경로>
장기 모델 옵션은 xgboost/ridge, 일일 옵션은 xgboost/ridge/elasticnet/arimax/baseline이다.
standalone inference는 저장 시점의 고정 원점만 지원한다. 임의 현장 갱신 API는 구현하지 않았다.
