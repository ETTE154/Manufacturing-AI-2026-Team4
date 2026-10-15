"""Offline contract verification: no real keys, webhooks, or network requests."""
import sys,json,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
import services
checks=[];captured=[]
def check(name,condition):checks.append({'name':name,'passed':bool(condition)});assert condition,name
class Response:
    def __init__(self,body):self.body=body.encode()
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self,*args):return self.body
def intercepted(request,timeout):
    captured.append(request)
    if request.full_url=='https://api.openai.com/v1/responses':return Response(json.dumps({'status':'completed','output':[{'content':[{'type':'output_text','text':json.dumps({'answer':'mock only'})}]}]}))
    if request.full_url=='https://hooks.slack.com/services/TEST/DEMO/MOCK':return Response('ok')
    raise AssertionError('Unexpected transport address; offline test refused')
services.urllib.request.urlopen=intercepted
services.CONFIG.api_key=lambda:'offline-test-placeholder'
services.DB.get=lambda key,default=None:True if key=='gpt_data_transfer_approved'else default
services.CONFIG.slack_url=lambda:'https://hooks.slack.com/services/TEST/DEMO/MOCK'
schema={'type':'object','properties':{'answer':{'type':'string'}},'required':['answer'],'additionalProperties':False}
answer=services.gpt_json('test only',{'energy_kwh':123.4},schema,'offline_contract')
body=json.loads(captured[-1].data);check('exact_user_selected_gpt5mini',body['model']=='gpt-5-mini');check('responses_store_false',body['store']is False);check('strict_json_schema',body['text']['format']['strict']is True);check('structured_response_parse',answer['answer']=='mock only')
services.notify_alarm({'title':'의심 후보','equipment':'17(6호기)','time':'2025-03-14 13:02:10','source_kind':'synthetic','evidence':{'score':1.2}})
slack=json.loads(captured[-1].data);check('slack_synthetic_provenance_in_message','증강 시연'in json.dumps(slack,ensure_ascii=False));check('slack_no_confirmed_fault_claim','미확인'in json.dumps(slack,ensure_ascii=False));check('slack_block_kit_payload',len(slack['blocks'])==4)
def refused(request,timeout):raise urllib.error.HTTPError(request.full_url,401,'denied',{},None)
services.urllib.request.urlopen=refused
try:services.gpt_json('test',{},schema,'test')
except services.IntegrationError as e:check('openai_401_error_redacted','offline-test-placeholder'not in str(e)and'401'in str(e))
else:check('openai_401_error_redacted',False)
result={'status':'passed','checks':checks,'network_requests':0,'real_openai_generation_verified':False,'real_slack_delivery_verified':False,'credentials':'synthetic placeholders only'}
(ROOT/'tests/transport_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({'status':'passed','checks':len(checks),'network_requests':0}))
