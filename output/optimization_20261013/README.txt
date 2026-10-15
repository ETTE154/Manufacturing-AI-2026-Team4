공개 원본 확인·예측 최적화·이상 탐지 추가 실험 (2026-10-13)

시작 파일
  optimization_report.html : 통합 보고서, 연구 근거, 전처리, 최적화 결과, 모델별 예측 그림
  source_audit/audit_report.html : 공식 다운로드와 제공 파일의 SHA-256 비교 및 데이터 해석 문제
  anomaly/anomaly_report.html : 8개 탐지 모델 × 후보/비교 구간 갤러리, 합성 주입 평가
  research/presentation_notes.txt : 연구 출처를 포함한 발표 문장
  verification_results.json : 독립 검증과 각 실험 검증의 최종 결과
  report_verification.json : 오프라인 HTML·이미지·선택 메뉴·브라우저 검증

예측 실험
  forecast/protocol.json : 시간 집계, train/validation/test 원점, 선택 기준, 한계
  forecast/comparison_results.json : 6개 모델 × 2개 목적의 설정과 지표
  forecast/comparison_selection_lock.json : 5월 평가 전 모델 선택과 설정 해시
  forecast/figures/ : 모델별 4월 24시간·5월 장기 예측 PNG/SVG 각 12개
  forecast/dayahead24/ : ARIMA13/XGB31/PatchTST7/iTransformer7/Chronos27/baseline17 후보
  forecast/long745/ : 동일 후보 예산, 745시간 단일 원점 과제
  각 모델 폴더에는 results.json, selection_lock.json, 예측 CSV, 최종 저장 모델이 있습니다.

PowerShell 재추론 예시 (프로젝트 폴더에서 실행)
  & 'output/modeling/.venv/Scripts/python.exe' 'output/optimization_20261013/forecast/predict_optimized.py' --scenario dayahead24 --model selected
  & 'output/modeling/.venv/Scripts/python.exe' 'output/optimization_20261013/forecast/predict_optimized.py' --scenario long745 --model xgboost
  --model : selected/arima/xgboost/patchtst/itransformer/chronos2/baseline
  이 CLI는 새 최적화 모델만 읽고 5월 정답을 읽지 않습니다. 저장된 4월29일 종료 이력으로
  현재 실험의 미래 경로를 재현합니다. 새 운영 기간의 일반적인 업데이트 API는 아닙니다.
  기존 output/modeling의 CLI는 기존 모델을 읽으므로 구분해서 사용합니다.

이상 탐지
  anomaly/anomaly_metrics.json : 원본 약한 규칙 일치도와 합성 정답 성능을 분리한 지표
  anomaly/threshold_locks.json : 3월 원본 정상 대리 구간에서 고정한 임계값
  anomaly/augmented_sample_manifest.json : 원본 관측 키, split, seed, 주입 유형·범위·크기
  anomaly/synthetic_observations.csv.gz : 합성 주입 데이터·정답 마스크
  anomaly/models/ : 8개 저장 탐지 모델
  anomaly/figures/ : 모델별 연속 구간 48쌍을 포함한 PNG/SVG 54쌍
  anomaly/predict_anomaly.py --help : 입력 CSV에 대한 모델별 점수·고정 임계값 경보 CLI
  원본 입력 파일·기존 출력 경로를 덮어쓰지 않도록 CLI가 검사합니다.

재현 조건
  기존 modeling의 전용 .venv 사용: CPU PyTorch, XGBoost, statsmodels, sklearn,
  chronos-forecasting. Chronos-2의 공식 고정 리비전 캐시 및 검토한 공식 신경망 모듈을
  읽어 사용합니다. Matplotlib은 기존 analysis/.plot_deps를 읽습니다.
  seed=20261013, CPU thread=4. 학습 스크립트는 protocol과 source 승인 기록을 검사합니다.
  build_report.py는 완성된 JSON·그림을 읽어 통합 HTML을 생성합니다.

해석상 주의점
  직접 내려받은 학습 CSV(4,909,245,995 bytes)와 5월 CSV(46,000 bytes)는 제공 파일과
  완전히 같습니다. 원본 및 output/analysis·output/modeling은 보존했습니다.
  localtime과 Unix 시간값, 전력/에너지 단위, 상하위 계측기 합계, 실제 삽입 정답은
  공급자 확인이 필요합니다. 전력량은 현재 가이드의 단위를 따른 잠정 kWh입니다.
  24시간 모델은 4월 29개 일일 원점에서 평가합니다. 5월은 원시 입력 없이 정답505개만
  제공되므로 4월30일 단일 원점에서 lead241~745를 평가하는 별도 업무입니다.
  검증 예산 내 최적 설정이며 전역 최적해·현재 절대 SOTA·현장 절감 효과를 보장하지 않습니다.
  기존 EDA에서 4월/5월을 본 이력이 있어 prospective blind test가 아닙니다.
  원본 센서 규칙과 합성 주입 마스크는 실제 고장 정답을 대체하지 않습니다.
  신경망 iTransformer는 공통 단변량 입력으로 다변량 장점을 평가할 수 없습니다.

데이터 다운로드 출처
  http://sahmyook-project.s3-website.ap-northeast-2.amazonaws.com/
  실제 객체 URL, 수신 메타데이터, SHA-256, 복사 목록은 source_audit/에 보존했습니다.
