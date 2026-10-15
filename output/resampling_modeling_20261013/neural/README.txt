H/D 신경망 및 고정 Chronos 실험

선택: 2025-02-15,02-22,02-25 각 원점의 P745 중 offset240..744
505시간 MAE를 구한 후 세 원점 평균만 사용했습니다. 최근 검증 정답은
2025-03-28 00:00이며, 내부 시험 원점2025-03-29와 겹치지 않습니다.
세 검증 경로가 겹치므로1515개 점수는 고유745개 정답 시간에서 나옵니다.
시험이나 May 정답으로 후보를 재선택하지 않았습니다. 이 작업의 코드는
May 정답 파일을 열지 않았으나 이전 작업에서 April/May EDA가 이루어졌으므로
완전히 새로운 전향적 맹검 시험이라고 주장하지 않습니다.

시간 모델: PatchTST 문맥24/72와 iTransformer 문맥24/72 및 과거 완료된
D/W/MS rate·완료 플래그 다변량 후보. iTransformer는 explicit target_rate
출력 채널에만 손실을 적용합니다. 원시 과거 시간 관측은 해당 hour+1h에
완료된 것으로 취급하며, 완료되지 않은 전체 기간 값은 입력으로 쓰지 않습니다.
첫 완료 기간 이전 fallback은 당시까지 관측된 target의 expanding mean입니다.

일 모델: 완전히 관측된 하루의 energy/24 rate를 문맥7일에서32일 직접
예측합니다. PatchTST는 patch_len4/stride2입니다. 첫 CV의 내부 fit44일에서
문맥7+출력32로 겹치는 학습 창6개만 생깁니다. 문맥14는 불가능하므로 제외했습니다.
일 예측 rate를24시간 균등 분배하여745시간을 출력합니다. 마지막 예측일은
요청된 첫1시간만 포함합니다. 미래의 실제 일 총량/일 평균을 사용하지 않습니다.

epoch는 원점 과거 데이터 내부에서 H745h/D32일을 제외한 fit으로 선택합니다.
상한24, patience4이며 일부 iTransformer는 최저점이 상한에 있어 수렴이
확정되지 않았습니다. 최종 April29까지의 refit은 선택 후보의 세 CV epoch
중앙값을 고정하며 April에서 epoch/구성을 새로 선택하지 않았습니다.

Chronos는 amazon/chronos-2 공식120M 고정 체크포인트의 기존 로컬 snapshot
29ec3766d36d6f73f0696f85560a422f50e8498c를 해시 검증 후 읽습니다.
가중치 재학습/다운로드/데이터 업로드는 없습니다. H 문맥full/720,
D 문맥full/28 및 과거 native 평균과의 alpha0/.25/1 혼합을 CV로만 선택합니다.
공식 point는0.5분위 중앙값입니다. H 선택alpha0는 단순 과거 평균과 같고
foundation 모델의 예측력이 향상되었다는 근거가 아닙니다.

요약 results.json; 분리된 selection_lock.json; 원점별 scaler/가중치;
validation/test/future CSV; native 예측 CSV; verification.json;
8개 PNG/SVG의 plot_manifest.json이 있습니다. 모든 예측 에너지는 가이드
단위 가정의 잠정 kWh이고13개 계측의 계층/중복 여부는 미확정입니다.

재학습(모델 파일 동결 후 검토 시에는 재학습을 실행하지 마세요):
"E:\AI work\AWS_프로젝트\output\modeling\.venv\Scripts\python.exe" "E:\AI work\AWS_프로젝트\output\resampling_modeling_20261013\neural\neural_resampling.py" --family patchtst --resolution daily
family=patchtst/itransformer, resolution=hourly/daily.

신경망 저장 문맥 재실행:
"E:\AI work\AWS_프로젝트\output\modeling\.venv\Scripts\python.exe" "E:\AI work\AWS_프로젝트\output\resampling_modeling_20261013\neural\neural_resampling.py" --predict --checkpoint "E:\AI work\AWS_프로젝트\output\resampling_modeling_20261013\neural\daily\patchtst\saved_models\future_unlabelled.pt" --output "E:\AI work\AWS_프로젝트\output\resampling_modeling_20261013\neural\replay\patch_cli.csv"
임의 새 과거 시계열은 --history CSV로 전달합니다. timestamp,energy_kwh가
연속 시간 간격이며 저장 scaler/가중치는 변경하지 않습니다. 일 모델의
새 원점은 완전히 완료된 하루 직후00시여야 합니다. 다중해상도 새 과거
문맥은 전달된 과거 데이터에서만 계산하므로 충분한 연속 과거를 제공하세요.

Chronos 공식 체크포인트 실제 재로드/forward:
"E:\AI work\AWS_프로젝트\output\modeling\.venv\Scripts\python.exe" "E:\AI work\AWS_프로젝트\output\resampling_modeling_20261013\neural\chronos_resampling.py" --predict --metadata "E:\AI work\AWS_프로젝트\output\resampling_modeling_20261013\neural\daily\chronos2\saved_models\future_unlabelled.json" --recompute --output "E:\AI work\AWS_프로젝트\output\resampling_modeling_20261013\neural\replay\daily_chronos2_cli.csv"
Chronos 함수 predict(metadata,output,recompute=True,pipe=None)는 raw중앙값
forward가 기존 캐시와 정확히 같은지, 혼합 및745시간 매핑이 같은지 검증합니다.
--recompute 생략 시 캐시 SHA를 확인하고 고정 혼합만 재실행합니다.
이 Chronos CLI는 저장된 과거 문맥의 재현용이며 임의 신규 CSV를 받지 않습니다.

감사 재실행: 위Python으로 verify_neural_resampling.py 실행.
27 CV +8 신경망 시험/최종 +4 공식 Chronos 경로의 재실행,
scaler/trainprefix/완료시계/32일→745시간 매핑이 정확히 일치했습니다.
재현성 통과가 실제 장기 예측 성능의 보장을 뜻하지는 않습니다.
