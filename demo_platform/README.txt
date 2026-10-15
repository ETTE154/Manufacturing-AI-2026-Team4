FactoryLens · 제출용 실행 안내
============================
프로젝트 설명·화면 구성·설치·데이터 준비·모델 성능과 제한은 저장소 루트 README.md를 확인하세요.

1. 저장소 루트 .venv를 만들고 requirements.txt를 설치합니다.
2. frontend에서 pnpm install --frozen-lockfile 및 pnpm build를 실행합니다.
3. traing-dataset/rtu_data_full.csv를 준비합니다. docs/data_manifest.json 또는 scripts/download_data.py를 사용하세요.
4. 저장소 루트에서 .\demo_platform\launch.ps1 실행 후 http://127.0.0.1:8099 접속.
   Python 실행 파일이 별도 위치라면 launch.ps1 -PythonPath <경로>를 사용합니다.

원본 CSV·프로젝트 가이드 PDF·실행 DB·키·웹훅·로그·설치환경은 제출본에 포함하지 않습니다.
키는 OPENAI_API_KEY 서버 환경변수로만 읽고, 미설정이면 로컬 기능을 제공합니다.
SLACK_WEBHOOK_URL과 SLACK_ALERTS_ENABLED는 Slack 선택 연동용 환경변수입니다.
자동 .env 로딩과 가이드/OCR 키 판독은 지원하지 않습니다.

설정 탭: 데이터·시연 / 알림·리포트 / 모델 검증.
운영현황·설비분석·고장진단·예측·리포트는 생산 업체 운영자 관점으로 구성했습니다.
원본의 5초 기록 재생이며 실제 공장의 현재 상태를 의미하지 않습니다.
