FactoryLens frontend

구현: React / Vite / Recharts / Lucide. 원본 CSV를 프런트엔드에 복사하거나 정답·가짜 데이터를 번들에 넣지 않습니다.
백엔드: http://127.0.0.1:8099. Vite의 /api 프록시로 연결합니다.
미리보기: http://127.0.0.1:5173. 프로덕션 dist는 백엔드 same-origin 정적 제공이 가능합니다.

번들 런타임 사용 예시 (PowerShell, frontend 폴더):
& 'C:\Users\jeon\.cache\codex-runtimes\codex-primary-runtime\dependencies\bin\fallback\pnpm.cmd' install
& 'C:\Users\jeon\.cache\codex-runtimes\codex-primary-runtime\dependencies\bin\fallback\pnpm.cmd' run build
& 'C:\Users\jeon\.cache\codex-runtimes\codex-primary-runtime\dependencies\bin\fallback\pnpm.cmd' run dev

백엔드 요청/상태 정규화는 src/api.js에 있습니다. API 계약은 ../docs/api_contract.txt를 따릅니다.
SSE 수신 이벤트는 module_id+recorded time으로 중복 제거합니다. 모든 숫자는 실제 응답에서 표시하며, 미연결/미설정 상태에 샘플 값을 만들지 않습니다.
원본 기록 시각은 문자열/명시적14자리 형식으로 표시합니다. ISO-naive 기록 시각을 브라우저 시간대로 변환하지 않습니다.
주간리포트는 종료 시각 기준 최근7일 데이터이며 시작일 필터를 임의 적용했다고 표시하지 않습니다. 일정은 선택 모듈과 분 단위 시각을 서버에 저장합니다.
Slack 테스트 버튼은 사용자가 명시적으로 누르는 경우에만 서버의 테스트 전송 API를 호출합니다. GPT와 Slack 비밀 값은 서버에만 보관합니다.

