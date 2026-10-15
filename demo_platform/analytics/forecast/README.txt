예측 실험 및 데모 서빙

results.json:196개 설정을 동일한3개 과거 원점의 마지막505시간 MAE로 비교.
dashboard_evaluation.json: root endpoint용7모델 및 고정 May505 실제/예측 시계열.
dashboard_summary.json: 선택모델·성능·한계의 요약.
comparison_results.csv:검증 및 내부 시험. may_comparison.csv: 최종 May1회 평가.
global_selection_lock.json: May 평가 전 선택과 모델/예측 파일SHA를 고정.
normalization_audit.json/verification.json:339개 prefix/scaler/inner-check,
14개 native 저장 경로 재현과 공식 코어/가중치 SHA 확인.
plot_manifest.json:7모델 각각의 May 실제·예측·잔차 PNG/SVG.

모델선택은Feb15/Feb22/Feb25의 각 P745 중offset240..744의 MAE 평균만
사용했습니다. 세 원점 정답은 겹치므로1515개 점수는745개 고유 시간입니다.
최신 검증 정답March28 00은 내부 시험March29보다 과거이며 정답 중첩0.
선택iTransformer의CV 개선은0.0295kWh(약0.039%)뿐입니다. 최저 검증
설정이라는 뜻이며 안정적인 미래 우위 또는 신호 학습 성공의 증명이 아닙니다.
실제로 선택모델 MayMAE89.1146은 기준선89.0083보다 높으며 모든 모델의
MayR2는 음수입니다. May의 사후 최저 Chronos도 모델 재선택 근거로 쓰지 않았습니다.

세 정규화 모두 origin 이전값에서만 residual-center와 scaler를 학습하고
원래 단위로 정확히 역변환했습니다. zscore/Robust/MinMax는 수치 최적화용이며
정답을 임의로288 또는1000으로 나누지 않았습니다. 정답·예측을 함께 같은
단위로 바꿔도R2는 불변입니다. 가이드의kWh 및13모듈 계층/중복 여부는 미확정.
작은MAPE는 큰 평균 대비 작은 변동 때문에 생길 수 있어 높은 예측력의 증명이 아닙니다.

통계/ML103개 + PatchTST8 + iTransformer9 + Chronos76 설정을 검증했습니다.
신경망은64차원2층 공식 코어를 재학습하고 검증 plateau·학습률 감소로
중단했습니다(고정24epoch제한 없음). 실제학습90~155epoch, 최저점1~125epoch.
inner best checkpoint를 저장하고 해당 learning-rate schedule을 outer
refit에 사용했습니다. 최종 epoch/schedule는CV 중앙값으로 고정해April에서
추가 튜닝하지 않았습니다. 고정 Chronos는 기존 로컬120M 공식 checkpoint의
중앙값 예측으로, 가중치 재학습·데이터 전송·새 모델 다운로드는 없습니다.

May는모든선택고정/감사통과이후 딱1회 평가했습니다. 이후에는 가중치/
구성/스케일러/편향을 바꾸지 않았습니다. 이전April/May EDA 이력이 있으므로
완전히 새로운 전향적 맹검 실험으로 설명하지 않습니다. 같은 폴더에서
재튜닝/selection reset/May 재평가를 CLI가 차단합니다. 새로운 사용자 승인
실험을 하려면 별도의실험폴더와 새로운 검증 계획이 필요합니다.

실시간 API:
from api import serving_forecast
result=serving_forecast(hourly_series,origin='2025-03-13 19:00',points=24,module_id='all')
Series는recorded naive hour index,guide provisionalkWh 값이며완료된 과거만
제공합니다.19:43커서는19:00 forecast origin,최종관측label18:00을 사용합니다.
API는April최종모델을March/December에재사용하지 않습니다. 원점까지의
독립 과거 평균 baseline으로재적합하고모델명을정확히표시합니다. 선택한
개별모듈값을input으로제공하면그모듈의과거에서독립적으로fit합니다.
전체forecast를비례배분해모듈모델인것처럼표현하지 않습니다.
최근과거7개의24h경로 replay metric은제공하되미래정확도를보장하지않습니다.
검증된미래interval이없어lower/upper는null입니다. 부족한과거는cold-start로표시.
serving_api_verification.json은미래값오염·빈시간·마지막미완료시간·coldstart
및13개독립모듈 테스트PASS를기록합니다.

고정 가이드 예측:
api.guide_forecast()는April30 00~May31 00의745개원래단위point를 제공합니다.
May실제정답은May10~May31 00의505개뿐입니다. 앞240시간은실제평가값을
보간하지않습니다. 이경로는실시간replay baseline과역할이다릅니다.

재현성감사(학습/선택변경없음):
"E:\AI work\AWS_프로젝트\output\modeling\.venv\Scripts\python.exe" "E:\AI work\AWS_프로젝트\demo_platform\analytics\forecast\verify_forecast.py"
API감사:
위Python으로verify_api.py실행. 기존원본·이전산출물·기존venv는읽기만합니다.
