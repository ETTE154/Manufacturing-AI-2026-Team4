피처 재검증과 예측 모델 재학습 (2026-10-13)

시작: feature_revision_report.html
  기존 피처 감사, 전체 피처 상관/시간 관계, 피처 중요도, 재선정 기준과 이름,
  기존 대비 재학습 결과, 모델별 실측/예측 그래프를 포함합니다.

주요 자료
  experiment_protocol.json : 24시간 업무와 745시간 과제의 시간 분리·가용시점·선정 원칙
  data/hourly_features.csv : 3,600시간 × 목표1 + 설비별 시간 통계442 (별도 hour 열)
  data/feature_catalog.json : 피처별 출처·단위·집계·가용시점·중복·배제 사유
  data/extraction_verification.json : 33,696,013행 원본 재집계 및 목표 일치 검증
  analysis/lag_correlations.csv : 피처별 Pearson/Spearman·lag0/1/6/24/48/168 및 BH 보정
  analysis/correlation_stability.csv : Dec–Feb와 March의 상관 안정성
  analysis/importance_individual_ridge.csv : 설명용 Ridge의 개별 피처 검증 중요도
  analysis/importance_grouped.csv : 그룹 검증 중요도·제외 실험
  analysis/analysis_results.json : ACF/PACF, 주기/달력, PCA, 날짜 순열 대조, 조건부 분산 진단
  analysis/selection_audits/ : fold별 training-prefix-only 센서 선택 기록
  comparison_results.json : 새 7개 24시간 모델/기준선과 기존 Chronos 비교, 장기 모델 별도 평가
  comparison_selection_lock.json : 설정 고정과 검증 MAE 기준 선택 기록
  may_predictions/ : 고정 후 공통으로 평가한 May505 예측·실측
  verification_results.json : 데이터·모델·지표·시간정렬·보고서 최종 검증

클래식 재학습
  classical/results.json : XGBoost/Ridge/ElasticNet/ARIMAX/EMA720 24시간 모델 결과
  classical/feature_group_ablation.csv : 피처 그룹별 후보 검증 성능
  classical/{모델}/gain_final.csv 또는 coefficients_final.csv : 전체 학습 중요도/계수
  classical/{모델}/chronological_feature_permutation.csv : 학습 중요도 상위20의 내부 시간 검증
  classical/long745/results.json : XGBoost/Ridge의 745시간 보조 모델
  각 모델 폴더에 저장 모델, 고정 입력 schema, 검증/평가/미래 예측 CSV가 있습니다.

신경망 재학습
  neural/results.json : iTransformer/PatchTST × 24시간/745시간, 4개 모델의 종합 결과
  iTransformer: 설비13 + 합계 목표 + prefix-selected 센서 입력, 합계 채널만 감독
  PatchTST: 채널별 표준화 후 공유 파라미터 학습, 13설비 역변환 예측만 합산
  PatchTST 센서는 auxiliary 학습 효과이며 추론 중 cross-channel attention이 아닙니다.
  neural/{scenario}/{family}/ : 선정 설정·정규화·입력 이름·checkpoints·예측 CSV·그림

PowerShell 저장 모델 재추론 (프로젝트 폴더에서 실행)
  & 'output/modeling/.venv/Scripts/python.exe' 'output/feature_revision_20261013/classical/predict_classical.py' --scenario dayahead24 --model selected --output 'output/feature_revision_20261013/new_day_prediction.csv'
  & 'output/modeling/.venv/Scripts/python.exe' 'output/feature_revision_20261013/neural/neural_revision.py' --predict --checkpoint 'output/feature_revision_20261013/neural/dayahead24/itransformer/saved_model.pt' --output 'output/feature_revision_20261013/new_neural_prediction.csv'
  실제 인자와 지원 범위는 각 CLI의 --help로 확인하십시오. 클래식은 저장된 현재 원점의
  문맥 재현용입니다. 신경망의 추가 history 입력은 연속한 완료 시간과 저장된 피처 이름이
  모두 필요합니다. 모델/피처 선택을 다시 수행하지 않습니다.

검증과 한계
  기존 원본·output/analysis·output/modeling·output/optimization_20261013은 보존했습니다.
  단위는 가이드 기준 잠정 kWh이고, 계측기 계층/단위 확인 문제는 남아 있습니다.
  같은 시간 센서·설비 전력량의 높은 상관은 미래 입력 가용성을 의미하지 않습니다.
  동시 V/I/PF 물리식의 상대 상관은 높지만 단위 배율 진단과 미래 지연 상관은 별개입니다.
  4월29개 일일 원점은 24시간 업무, May505는 Apr30단일 원점의 lead241~745 과제입니다.
  피처/스케일러는 각 학습 prefix에서 추정하며 최종 입력 schema는 미사용 평가 이전에 고정합니다.
  4월·5월을 기존 EDA에서 본 이력이 있으므로 pristine prospective 검증으로 주장하지 않습니다.
  후보 탐색 예산 안의 결과이며 더 많은 피처·대형 모델이 개선이나 전역 최적해를 보장하지 않습니다.
  작은 MAPE를 예측 정확도(100−MAPE)로 바꾸지 말고, 기준선·R²·일일 오차 차이를 함께 확인합니다.

실행 환경: 기존 modeling 전용 .venv, CPU thread4, seed20261013. 그림은 기존 analysis/.plot_deps와
Malgun Gothic을 사용했습니다. 원본 집계 script 및 모델 script는 각각의 하위 폴더에 있습니다.
