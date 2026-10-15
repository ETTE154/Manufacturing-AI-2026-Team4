AWS 프로젝트 기본 데이터 분석
분석일: 2026-10-11

먼저 project_analysis.html을 브라우저에서 열어 보십시오.
그림과 핵심 표가 파일에 포함되어 있어 네트워크 연결 없이 볼 수 있습니다.
figures/overview.png는 주요 차트의 요약 이미지입니다. 그림별 PNG와 SVG도 보관했습니다.

원본
- ../../프로젝트가이드.pdf
- ../../traing-dataset/rtu_data_full.csv (33,696,013행, 19열, 13모듈)
- ../../may-dataset/rtu_ground_truth_may.csv (505행, 5열)
원본은 수정하지 않았습니다. 가이드 인증정보는 보고서에 복사하지 않았습니다.

계산과 해석
- 가이드 activePower의 단위 kW를 따릅니다. 표본당 에너지=activePower*5/3600 kWh.
- localtime을 시간축으로 사용합니다. timestamp는 한국시간 설명과 일치하지 않으며 3월9일 전환에서 중복 키가 발생합니다.
- 1모듈·1시간당 720개 고유 localtime 표본인 시간만 기준선 학습에 사용합니다.
- 학습 자료 마지막 4/30 00시는 1표본뿐이므로 완전 시간에서 제외합니다.
- 모듈 계층과 5월 목표 단위·범위가 확인되지 않아 합계는 잠정 모듈 합계입니다.
- 비용 180원/kWh와 탄소 0.424kgCO2/kWh는 가이드 가정값입니다.
- 5월은 5/10~5/31 00시 제공 구간이며 달력상 5월 전체가 아닙니다.
- 이상 후보의 전압/역률 기준은 탐색 기준입니다. 라벨과 현장 정상범위를 검증하지 않아 고장 확정으로 볼 수 없습니다.

주요 데이터
- analysis_metrics.json: 리포트의 수치·가정·검증 조건
- hourly_by_equipment.csv: 설비별 시간 집계, 표본 수 및 완전시간 표시
- hourly_module_sum.csv, daily_summary.csv, monthly_summary.csv: 완전 시간 기준 합계
- equipment_summary.csv: 설비별 전수 통계와 관측 에너지
- anomaly_candidate_rows.csv: 전수 탐색에서 찾은 후보 레코드
- anomaly_by_equipment.csv: 설비별 후보 수와 비율
- anomaly_events.csv: 연속 5초 후보 구간. 대부분 단일 표본이며 실제 장애 이벤트 수가 아닙니다.
- training_profile.json, time_anomaly_audit.json: 결측, 시간 키 및 누적 카운터 감사
- baseline_audit/: 4월 내부검증과 5월 고정 예측, MAE 및 코드
- baseline_audit/may_forecast_energy_cost_carbon.csv: 같은 고정 에너지 예측을 비용·탄소로 환산한 505개 시간별 결과
- may_audit/: 검증 데이터 품질과 분포 진단. 이 폴더의 5월 내부 분할은 데이터 진단용이며 프로젝트 공식 학습 성능이 아닙니다.
- sampled_sensor_rows.csv, sample_correlations.csv: 약16,849행 체계적 표본 기반 센서 관계

재현 순서 (프로젝트 루트에서 Python 실행)
1. python output/analysis/scripts/profile_training.py
2. python output/analysis/scripts/audit_times_anomalies.py
3. python output/analysis/baseline_audit/baseline_audit.py
4. python output/analysis/scripts/summarize_analysis.py
5. python output/analysis/scripts/build_visual_report.py

필요 분석 라이브러리: pandas, numpy. 그림: matplotlib, Pillow, fonttools.
현재 실행에는 Codex 번들 Python을 사용했고 부족한 그래프 의존성은 이 폴더의 .plot_deps에만 설치했습니다.
스크립트는 원본을 읽고 이 분석 결과 폴더에만 결과를 기록합니다.
