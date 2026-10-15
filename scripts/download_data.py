"""Explicit public dataset download with byte size/SHA verification and no overwrite."""
from pathlib import Path
import hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    result=hashlib.sha256()
    with path.open('rb')as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):result.update(chunk)
    return result.hexdigest()
def main():
    manifest=json.loads((ROOT/'docs/data_manifest.json').read_text(encoding='utf8'))
    for item in manifest['datasets']:
        path=ROOT/item['path'];path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if path.stat().st_size!=item['bytes']or digest(path)!=item['sha256']:raise RuntimeError('Existing file differs; preserved: '+item['path'])
            print('Already verified: '+item['path']);continue
        pending=path.with_suffix(path.suffix+'.part')
        if pending.exists():raise RuntimeError('Partial file already exists; inspect it before retrying: '+str(pending))
        print('Downloading: '+item['path'])
        with urllib.request.urlopen(item['url'],timeout=120)as response,pending.open('xb')as target:
            while chunk:=response.read(4*1024*1024):target.write(chunk)
        if pending.stat().st_size!=item['bytes']or digest(pending)!=item['sha256']:raise RuntimeError('Downloaded bytes do not match the audited dataset; partial file retained.')
        if path.exists():raise RuntimeError('Destination appeared during download; preserved.')
        pending.rename(path);print('Verified: '+item['path'])
if __name__=='__main__':main()
