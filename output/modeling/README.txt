전력 예측·전기적 이상 후보 진단 모델 개발
========================================

통합 결과: project_modeling_report.html
연구 근거: research/research_review.html, research/literature_review.json
예측 지표: forecast/comparison_results.json
전체 예측: forecast/all_predictions.csv
고정 선택: forecast/deployment_selection_lock.json
진단 지표: diagnosis/diagnosis_model_metrics.json
검증 결과: verification_results.json, research/model_audit.json

범위와 해석
-----------
ARIMA, XGBoost, PatchTST, iTransformer를 학습하고 공식 Chronos-2 120M 사전학습
가중치를 zero-shot으로 비교했다. PatchTST/iTransformer는 공식 구조를 축소한
현 데이터 재학습이며 공개 벤치마크 수치의 재현이나 2026년 절대 SOTA 주장이 아니다.
HIPE는 산업 계측 구조상 가장 가까운 공개 데이터이며, 확인한 HIPE 연구는 단기
예측이다. 장기 horizon 설계에는 시간 단위 ECL 전력 벤치마크를 함께 참고했다.

완전한 시간 3,600개로 학습한다. 2025-04-30 00부터 745개를 출력해 관측 공백을
연결하고, 정답이 제공된 05-10 00~05-31 00의 505개만 평가한다. 첫 미래값은
origin offset=0이며 마지막 학습 관측 기준 lead=1이다. 모델 선택은 Feb25 origin
검증 P745 MAE로 고정했고 March29 내부 테스트·May 정답으로 다시 선택하지 않았다.
May 정답은 앞선 기술통계 단계에서 이미 열람했으므로 진정한 사전 미관측 검증은 아니다.

타깃은 13개 계측기의 동시 시간 전력량 합계다. 계측기 상하위 관계가 확인되지 않아
시설 전체 소비량으로 확정할 수 없다. May hourly_pow의 단위·계측 범위가 이 파생
kWh와 일치한다는 가정 하의 수치다. 비용 180원/kWh·탄소0.424kgCO2/kWh는 가이드
가정이며 실제 청구요금·현재 공식 배출계수를 확인한 결과가 아니다.

진단 데이터에는 실제 고장 정답이 없다. 전압·역률 등 규칙을 약한 라벨로 만들었다.
규칙 모사 분류의 높은 수치는 실제 고장 증거가 아니다. AE는 현재 5초 관측의 재구성
이상 점수를 산출한다. 미래 고장 발생 시점이나 고장 유형의 예측 성능은 아직 검증할
수 없다. 표본 경보 수는 전체 원본의 사건 수가 아니다.

로컬 실행 (PowerShell, 프로젝트 폴더에서)
----------------------------------------
$modelPython = Join-Path (Get-Location) 'output\modeling\.venv\Scripts\python.exe'

# 저장된 검증 선택 모델로 추론. 파생 시간 집계만 읽고 May 정답은 읽지 않는다.
& $modelPython .\output\modeling\forecast\predict_forecast.py --model selected

# ARIMA/XGB는 자체 저장된 과거 요약만으로 추론 가능
& $modelPython .\output\modeling\forecast\infer_saved.py --model xgboost --points 745

# 신규 시간 데이터: timestamp,energy_kwh 열을 가진 연속 960시간 이상 CSV
& $modelPython .\output\modeling\forecast\predict_forecast.py --model xgboost --history '.\new_hourly.csv' --steps 168 --output '.\new_predictions.csv'

# 신규 5초 센서 관측. 입력은 원본의 14개 센서 열을 포함한다.
& $modelPython .\output\modeling\diagnosis\predict_diagnosis.py --input '.\new_sensor_rows.csv' --output '.\new_anomaly_scores.csv'

# 아래는 이번 실험의 수행 순서다. 기존 선택 lock이 있는 현재 폴더에서
# 처음부터 반복하면 lock 검증이 중단될 수 있다. 새 학습 실험은 산출물을
# 별도로 보존한 작업 복사본에서 수행한다. 현재 모델 검증은 마지막 명령만 실행한다.
& $modelPython .\output\modeling\forecast\run_forecasts.py --internal-only --n-jobs 4
& $modelPython .\output\modeling\forecast\patchtst\train_patchtst.py --stage internal
& $modelPython .\output\modeling\forecast\itransformer_experiment.py
& $modelPython .\output\modeling\forecast\chronos2_experiment.py
& $modelPython .\output\modeling\forecast\compare_models.py --stage lock
& $modelPython .\output\modeling\forecast\patchtst\train_patchtst.py --stage final-refit
& $modelPython .\output\modeling\forecast\itransformer_experiment.py --final
& $modelPython .\output\modeling\forecast\chronos2_experiment.py --final
& $modelPython .\output\modeling\forecast\evaluate_may_from_lock.py --n-jobs 4
& $modelPython .\output\modeling\forecast\patchtst\train_patchtst.py --stage may-score
& $modelPython .\output\modeling\forecast\compare_models.py --stage score
& $modelPython .\output\modeling\build_model_report.py
& $modelPython .\output\modeling\verify_modeling.py

기존 배포 선택 lock은 다른 선택이나 config로 덮어쓰지 않는다. 새 실험은 별도 출력
폴더를 사용한다. 공개 모델 가중치·공식 모듈의 다운로드는 네트워크를 사용하지만
데이터 추론은 로컬이다. Chronos 캐시가 준비된 predict_forecast.py는 로컬 파일만
사용한다. 원본 PDF/CSV는 보존했으며 새 산출물은 output/modeling 하위에 저장했다.

환경
----
Python 3.12, NumPy/Pandas는 번들 런타임을 system-site-packages로 사용한다.
추가 패키지는 output/modeling/.venv에 설치했다. 정확한 버전은
environment_lock.json/requirements-modeling.txt를 참고한다. CPU PyTorch와
각 모델 4개 스레드로 실행했다. matplotlib는 기존 output/analysis/.plot_deps를
참조하며 Windows Malgun Gothic을 사용한다. 인터넷 연결 없는 새 환경에서 학습을
재현하려면 설치 패키지·공식 vendor·Chronos 모델 캐시도 함께 준비해야 한다.
