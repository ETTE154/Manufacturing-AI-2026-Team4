제공 데이터 기본 분석 보고서

최종 자료:
- pdf/basic_data_report.pdf: 18쪽 공유/인쇄용 보고서
- data_report.html: HTML 보고서
- figures_and_tables.zip: 새로 작성한 그래프 PNG/SVG와 분석 CSV 묶음

대상:
원본 Full 33,696,013행/19피처/13모듈/5초, May505행/5피처/1시간.
기간: 기록된localtime 기준 Full2024-12-01 00:00~2025-04-30 00:00.
완전 시간집계:2024-12-01 00:00~2025-04-29 23:00,3600시간.
실제지역시간과계측기상하위관계는미확정이므로 가이드 kWh는잠정값.
전수 센서통계는 검증된기존전체스캔 summary에서재계산.
센서분포/상관은13모듈×3600시간×4개의 기존층화표본187200행.
원본5초확대그림은 첫PM3하루17280행을이번에직접읽은기술예시.

시계열분해:
HrobustSTL24, HrobustSTL168, DrobustSTL7, HrobustMSTL24/168.
주21/月4는연간계절분해를수행하지않음.
전체과거자료분해는사후EDA이며 예측시점의입력피처로바로사용할수없음.
분해component가존재한다고 실제주기성입증으로해석하지않음.
입력단위: H시간전력량kWh(잠정), D완료일전력량/24h(kWh/h,기간평균부하).
정격용량정보는없으며 평균부하는정격대비부하율(%)이아님.

소스:
build_raw_views.py: bounded-prefix원자료그래프
evidence/build_evidence.py: 원본스키마/수집범위/단위/품질
visuals/build_data_overview_visuals.py: 해상도별집계그림
decomposition/decompose.py: 분해와주기진단
build_report.py: PDF/HTML+보고서수치+4개새집계CSV
render_and_pack.py: PDF렌더링/표·그림ZIP
verify_report.py: 통합검증/파일색인

Python runtime:
E:\AI work\AWS_프로젝트\output\modeling\.venv\Scripts\python.exe
PDF reportlab/PDF자료처리는 bundled runtime, 통계는기존.venv, 그림은기존.plot_deps활용.
분해·집계소스를재실행하면이폴더의생성파일이갱신되므로 기존완성자료보존후실행.
PDF marker 계약은이번최초생성전1회성공했고 다시실행하지않음.
원본·이전모델·이전보고서는미수정. 원본사본생성0/원본이동0/원본삭제0.
