"""Scan exactly Git-tracked submission bytes; never display matched credentials."""
from pathlib import Path
import argparse,gzip,hashlib,json,re,subprocess,zipfile,io,zlib
ROOT=Path(__file__).resolve().parents[1]
PATTERNS={
 'openai':rb'sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}',
 'slack_webhook':rb'https://hooks\.slack\.com/services/[A-Za-z0-9/_-]{15,}',
 'github_token':rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})',
 'aws_access_key':rb'(?:AKIA|ASIA)[A-Z0-9]{16}',
 'private_key':rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
 'signed_download':rb'X-Amz-Signature=[A-Za-z0-9]{32,}'
}
RULES={kind:re.compile(pattern)for kind,pattern in PATTERNS.items()}
FORBIDDEN={'.venv','node_modules','runtime_packages','.model_cache','__pycache__','revisions','logs','downloads'}
def secret_hits(content):return sorted(kind for kind,rule in RULES.items()if rule.search(content))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);args=parser.parse_args()
    names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf8').split('\0')
    files=[];findings=[];compressed=0
    for name in filter(None,names):
        path=ROOT/name
        if not path.is_file():findings.append(dict(path=name,category='missing_tracked_file'));continue
        parts=Path(name).parts
        if any(part in FORBIDDEN for part in parts)or path.suffix.lower()=='.pdf'or(path.name.startswith('.env')and path.name!='.env.example')or path.name.startswith('platform.sqlite')or name.startswith(('traing-dataset/','may-dataset/')):
            findings.append(dict(path=name,category='forbidden_submission_path'))
        data=path.read_bytes()
        if len(data)>=100*1024**2:findings.append(dict(path=name,category='github_large_file_limit'))
        for category in secret_hits(data):findings.append(dict(path=name,category=category))
        # Inspect compressed artifacts without unpickling or executing model code.
        if path.suffix=='.gz':
            with gzip.open(path,'rb')as stream:
                tail=b''
                while block:=stream.read(1024*1024):
                    for category in secret_hits(tail+block):findings.append(dict(path=name,category=category+'_gzip'))
                    tail=block[-512:]
            compressed+=1
        elif zipfile.is_zipfile(io.BytesIO(data)):
            with zipfile.ZipFile(io.BytesIO(data))as archive:
                for item in archive.infolist():
                    if item.file_size>512*1024**2:continue
                    with archive.open(item)as stream:
                        tail=b''
                        while block:=stream.read(1024*1024):
                            for category in secret_hits(tail+block):findings.append(dict(path=name,category=category+'_archive'))
                            tail=block[-512:]
            compressed+=1
        elif path.suffix in {'.joblib','.pkl'}and data[:1]==b'x':
            try:
                decoded=zlib.decompress(data)
                for category in secret_hits(decoded):findings.append(dict(path=name,category=category+'_compressed_pickle'))
                compressed+=1
            except zlib.error:pass
        files.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    report=dict(status='PASS'if not findings else'FAIL',tracked_files=len(files),bytes=sum(row['bytes']for row in files),compressed_artifacts_inspected=compressed,findings=findings,files=files)
    if args.report:args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({key:report[key]for key in ['status','tracked_files','bytes','compressed_artifacts_inspected','findings']},ensure_ascii=False))
    raise SystemExit(bool(findings))
if __name__=='__main__':main()
