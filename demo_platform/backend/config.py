"""Server-only integration configuration from the process environment.

Secrets stay in memory. No document parsing, OCR, credential probing, or
implicit .env loading is performed. Missing keys leave local features available.
"""
from pathlib import Path
import os,threading
APP=Path(__file__).resolve().parents[1];ROOT=APP.parent
DATA=APP/'data';DATA.mkdir(exist_ok=True)
MODEL='gpt-5-mini'
class IntegrationConfig:
    def __init__(self):
        self.lock=threading.RLock();self._key=None;self._key_source='not_configured';self.openai_status='not_configured'
        self._slack=os.environ.get('SLACK_WEBHOOK_URL','').strip();self.slack_enabled=os.environ.get('SLACK_ALERTS_ENABLED','false').lower()=='true';self.slack_status='not_tested'if self._slack else 'not_configured'
    def api_key(self):
        with self.lock:
            if self._key:return self._key
            key=os.environ.get('OPENAI_API_KEY','').strip()
            if not key:return None
            self._key=key;self._key_source='server_environment'
            if self.openai_status=='not_configured':self.openai_status='not_tested'
            return self._key
    def slack_url(self):
        with self.lock:return self._slack if self.slack_enabled else ''
    def configure_slack(self,url=None,enabled=None):
        from urllib.parse import urlparse
        with self.lock:
            if url:
                parsed=urlparse(url)
                if parsed.scheme!='https'or parsed.hostname!='hooks.slack.com'or not parsed.path.startswith('/services/')or parsed.query or parsed.fragment:
                    raise ValueError('Slack의 HTTPS Incoming Webhook URL만 사용할 수 있습니다.')
                self._slack=url;self.slack_status='not_tested'
            if enabled is not None:self.slack_enabled=bool(enabled)
            return self.public()
    def public(self):
        self.api_key()
        from store import DB
        with self.lock:return {'openai':{'configured':bool(self._key),'model':MODEL,'status':self.openai_status,'source':self._key_source,'data_transfer_approved':DB.get('gpt_data_transfer_approved',False),'setup_note':'GPT를 사용하려면 서버 실행 환경에 OPENAI_API_KEY를 설정한 뒤 재시작하세요. 미설정 상태에서는 로컬 통계·검색을 사용합니다. .env 파일은 자동으로 읽지 않습니다.'},'slack':{'configured':bool(self._slack),'enabled':self.slack_enabled,'status':self.slack_status,'storage':'server_memory_or_environment'}}
CONFIG=IntegrationConfig()
