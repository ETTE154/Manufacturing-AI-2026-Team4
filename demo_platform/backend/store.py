"""Durable replay logs and notification outbox. No credentials are persisted."""
import sqlite3,json,threading,datetime,hashlib
from config import DATA
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='milliseconds')
def dumps(v):return json.dumps(v,ensure_ascii=False,allow_nan=False,default=str)
def alarm_id(event):
    """The existing persistent ten-minute deduplication identity, unchanged."""
    scenario_key=event.get('scenario_id','synthetic')if event['source_kind']=='synthetic'else'recorded'
    key=f"sustained-v1:{event['source_kind']}:{scenario_key}:{event['module_id']}:{event['type']}:{event['time'][:15]}"
    return hashlib.sha256(key.encode()).hexdigest()[:20]
class Database:
    def __init__(self):
        self.path=DATA/'platform.sqlite';self.lock=threading.RLock()
        with self.connect() as c:
            c.executescript('''PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS telemetry(id INTEGER PRIMARY KEY,session_id TEXT,source_time TEXT,received_at TEXT,module_id TEXT,equipment TEXT,status TEXT,payload TEXT);
            CREATE INDEX IF NOT EXISTS telemetry_time ON telemetry(source_time,module_id);
            CREATE TABLE IF NOT EXISTS alarms(id TEXT PRIMARY KEY,module_id TEXT,source_time TEXT,type TEXT,status TEXT,slack_status TEXT,created_at TEXT,payload TEXT);
            CREATE TABLE IF NOT EXISTS outbox(alarm_id TEXT PRIMARY KEY,status TEXT,attempts INTEGER DEFAULT 0,next_attempt TEXT,last_error TEXT);
            CREATE TABLE IF NOT EXISTS reports(id TEXT PRIMARY KEY,created_at TEXT,payload TEXT);
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,payload TEXT);
            ''')
    def connect(self):
        c=sqlite3.connect(self.path,timeout=20);c.row_factory=sqlite3.Row;return c
    def log(self,rows,session,received):
        with self.lock,self.connect() as c:
            c.executemany('INSERT INTO telemetry(session_id,source_time,received_at,module_id,equipment,status,payload) VALUES(?,?,?,?,?,?,?)',[(session,r['time'],received,r['module_id'],r['equipment'],r.get('status','normal'),dumps(r))for r in rows])
    def log_count(self):
        with self.connect()as c:return c.execute('SELECT COUNT(*) FROM telemetry').fetchone()[0]
    def logs(self,module='all',limit=100):
        with self.connect()as c:
            sql='SELECT * FROM telemetry';args=[]
            if module!='all':sql+=' WHERE module_id=?';args.append(str(module))
            total=c.execute(sql.replace('SELECT *','SELECT COUNT(*)'),args).fetchone()[0]
            rows=c.execute(sql+' ORDER BY id DESC LIMIT ?',args+[int(limit)]).fetchall()
            return {'total':total,'rows':[{'id':r['id'],'module_id':r['module_id'],'equipment':r['equipment'],'source_time':r['source_time'],'received_at':r['received_at'],'status':r['status'],'activePower':json.loads(r['payload'])['activePower']}for r in rows]}
    def alarm(self,event):
        event=dict(event);event['alarm_policy']='3_consecutive_ticks'
        event['id']=alarm_id(event);event.update(status='open',slack_status='pending_configuration',created_at=now())
        with self.lock,self.connect()as c:
            n=c.execute('INSERT OR IGNORE INTO alarms VALUES(?,?,?,?,?,?,?,?)',(event['id'],event['module_id'],event['time'],event['type'],'open','pending_configuration',event['created_at'],dumps(event))).rowcount
            if n:c.execute('INSERT OR IGNORE INTO outbox(alarm_id,status,attempts,next_attempt) VALUES(?,?,?,?)',(event['id'],'pending',0,now()))
            return bool(n)
    def alarm_by_id(self,event_id):
        with self.connect()as c:
            row=c.execute("SELECT * FROM alarms WHERE id=? AND json_extract(payload,'$.alarm_policy')='3_consecutive_ticks'",(str(event_id),)).fetchone()
            if row is None:return None
            event=json.loads(row['payload']);event['status']=row['status'];event['slack_status']=row['slack_status'];return event
    def alarms(self,module='all',limit=1000,start=None,end=None,source_kind=None):
        with self.connect()as c:
            where=" WHERE json_extract(payload,'$.alarm_policy')='3_consecutive_ticks'";args=[]
            if module!='all':where+=' AND module_id=?';args.append(str(module))
            if start is not None:where+=' AND source_time>=?';args.append(str(start))
            if end is not None:where+=' AND source_time<?';args.append(str(end))
            if source_kind is not None:
                if source_kind not in['recorded','synthetic']:raise ValueError('알 수 없는 원천 종류입니다.')
                where+=" AND json_extract(payload,'$.source_kind')=?";args.append(source_kind)
            rows=c.execute('SELECT * FROM alarms'+where+' ORDER BY source_time DESC LIMIT ?',args+[int(limit)]).fetchall()
            out=[]
            for r in rows:
                item=json.loads(r['payload']);item['status']=r['status'];item['slack_status']=r['slack_status'];out.append(item)
            return {'rows':out,'total':c.execute('SELECT COUNT(*) FROM alarms'+where,args).fetchone()[0]}
    def ack(self,alarm_id):
        with self.lock,self.connect()as c:
            if not c.execute('SELECT 1 FROM alarms WHERE id=?',(alarm_id,)).fetchone():raise ValueError('알람을 찾을 수 없습니다.')
            c.execute('UPDATE alarms SET status=? WHERE id=?',('acknowledged',alarm_id))
        return {'id':alarm_id,'status':'acknowledged'}
    def active_alarm_count(self):
        with self.connect()as c:return c.execute("SELECT COUNT(*) FROM alarms WHERE status='open' AND json_extract(payload,'$.alarm_policy')='3_consecutive_ticks'").fetchone()[0]
    def due_notifications(self):
        with self.connect()as c:return [dict(r)for r in c.execute("SELECT a.payload,o.attempts FROM outbox o JOIN alarms a ON a.id=o.alarm_id WHERE o.status IN (?,?) AND o.next_attempt<=? AND a.status=? AND json_extract(a.payload,'$.alarm_policy')='3_consecutive_ticks' ORDER BY a.created_at LIMIT 3",('pending','retry',now(),'open'))]
    def notification_result(self,alarm_id,success,attempts,error=None):
        retry=not success and attempts<3;status='sent'if success else'retry'if retry else'failed'
        future=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=30*2**attempts)).isoformat(timespec='milliseconds')
        with self.lock,self.connect()as c:
            c.execute('UPDATE outbox SET status=?,attempts=?,next_attempt=?,last_error=? WHERE alarm_id=?',(status,attempts,future,error,alarm_id));c.execute('UPDATE alarms SET slack_status=? WHERE id=?',(status,alarm_id))
    def set(self,key,value):
        if any(s in key.lower()for s in ['key','secret','webhook','token']):raise ValueError('Credential persistence is prohibited')
        with self.lock,self.connect()as c:c.execute('INSERT OR REPLACE INTO settings VALUES(?,?)',(key,dumps(value)))
    def get(self,key,default=None):
        with self.connect()as c:r=c.execute('SELECT payload FROM settings WHERE key=?',(key,)).fetchone();return json.loads(r[0])if r else default
    def save_report(self,report):
        with self.lock,self.connect()as c:c.execute('INSERT OR REPLACE INTO reports VALUES(?,?,?)',(report['id'],report['generated_at'],dumps(report)))
        return report
    def reports(self):
        with self.connect()as c:return [json.loads(r[0])for r in c.execute('SELECT payload FROM reports ORDER BY created_at DESC LIMIT 30')]
DB=Database()
