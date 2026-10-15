"""Direct official-public-source download and byte integrity audit."""
from pathlib import Path, PurePosixPath
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request, urllib.error, json, hashlib, time, datetime, sys
sys.stdout.reconfigure(encoding="utf-8")
ROOT=Path(r"E:\AI work\AWS_프로젝트")
OUT=ROOT/"output/optimization_20261013/source_audit"
BLOCK=8*1024*1024

def timestamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
def safe_target(key):
    relative=PurePosixPath(key)
    if relative.is_absolute() or ".." in relative.parts or any(":" in part for part in relative.parts):
        raise ValueError("Unsafe source object path")
    path=(OUT/"downloads"/Path(*relative.parts)).resolve()
    path.relative_to((OUT/"downloads").resolve())
    return path
def sha_file(path):
    started=time.monotonic();digest=hashlib.sha256();total=0
    before=path.stat()
    with path.open("rb") as handle:
        while True:
            data=handle.read(BLOCK)
            if not data:break
            digest.update(data);total+=len(data)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise RuntimeError("Local source changed during read")
    return {"path":str(path),"bytes":total,"sha256":digest.hexdigest(),
            "mtime_ns":before.st_mtime_ns,"elapsed_seconds":time.monotonic()-started}
def hash_originals(objects):
    records=[]
    for obj in objects:
        key=obj["key"];record=sha_file(ROOT/Path(*PurePosixPath(key).parts));record["source_key"]=key
        records.append(record)
        print("ORIGINAL_HASH_COMPLETE",key,record["bytes"],record["sha256"],flush=True)
    return records
def download(obj):
    key=obj["key"];target=safe_target(key);target.parent.mkdir(parents=True,exist_ok=True)
    partial=target.with_name(target.name+".part")
    digest=hashlib.sha256();received=0;started=time.monotonic();headers_seen=[]
    expected=int(obj["bytes"]);etag=obj["etag"]
    for attempt in range(1,4):
        if partial.exists():
            received=partial.stat().st_size
            digest=hashlib.sha256()
            with partial.open("rb") as prior:
                while True:
                    block=prior.read(BLOCK)
                    if not block:break
                    digest.update(block)
        req_headers={"User-Agent":"Mozilla/5.0 (official source integrity audit)","If-Match":etag}
        if received:req_headers["Range"]=f"bytes={received}-"
        request=urllib.request.Request(obj["download_url"],headers=req_headers)
        try:
            with urllib.request.urlopen(request,timeout=120) as response:
                status=response.status
                headers={k:response.headers.get(k) for k in ["Content-Length","Content-Type","ETag","Last-Modified","Content-Range","x-amz-version-id"] if response.headers.get(k)}
                headers_seen.append({"attempt":attempt,"status":status,"headers":headers,"started_at_utc":timestamp()})
                if headers.get("ETag")!=etag:raise RuntimeError("Remote ETag changed after listing")
                if received and status!=206:
                    if status!=200:raise RuntimeError("Unexpected range response")
                    received=0;digest=hashlib.sha256()
                mode="ab" if received else "wb"
                print("DOWNLOAD_START",key,"expected_bytes",expected,"status",status,"resume",received,flush=True)
                last=time.monotonic()
                with partial.open(mode) as handle:
                    while True:
                        block=response.read(BLOCK)
                        if not block:break
                        handle.write(block);digest.update(block);received+=len(block)
                        now=time.monotonic()
                        if now-last>=15:
                            print("DOWNLOAD_PROGRESS",key,"bytes",received,"percent",round(received/expected*100,2),"MiB_per_second",round(received/(now-started)/1024**2,2),flush=True)
                            last=now
                if received!=expected:raise RuntimeError(f"Length mismatch {received} != {expected}")
            partial.replace(target)
            record={"source_key":key,"download_url":obj["download_url"],"downloaded_path":str(target),
                    "bytes":received,"sha256":digest.hexdigest(),"elapsed_seconds":time.monotonic()-started,
                    "response_history":headers_seen,"completed_at_utc":timestamp(),
                    "etag_note":"ETag records source object version evidence. It is not assumed to be a cryptographic content checksum or MD5; multipart and encryption behavior can change its meaning. Independently computed SHA-256 is used for byte identity."}
            print("DOWNLOAD_COMPLETE",key,received,record["sha256"],flush=True)
            return record
        except (urllib.error.URLError,TimeoutError,ConnectionError,OSError,RuntimeError) as exc:
            print("DOWNLOAD_RETRY",key,"attempt",attempt,"error",str(exc),flush=True)
            if attempt==3:raise
    raise RuntimeError("Unreachable")

def main():
    manifest=json.loads((OUT/"source_manifest.json").read_text(encoding="utf-8"))
    data=[o for o in manifest["objects"] if o["key"].lower().endswith(".csv")]
    if len(data)!=2:raise ValueError("Source dataset inventory changed; review before download")
    downloads=[];local=None
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_map={executor.submit(download,obj):("download",obj["key"]) for obj in data}
        future_map[executor.submit(hash_originals,data)]=("originals",None)
        for future in as_completed(future_map):
            kind,key=future_map[future];result=future.result()
            if kind=="originals":local=result
            else:downloads.append(result)
            dump("download_progress.json",{"completed_downloads":downloads,"originals":local,"recorded_at_utc":timestamp()})
    lookup={item["source_key"]:item for item in local};pairs=[]
    for remote in downloads:
        original=lookup[remote["source_key"]]
        pair={"source_key":remote["source_key"],"original":original,"downloaded":remote,
              "bytes_match":original["bytes"]==remote["bytes"],
              "sha256_match":original["sha256"]==remote["sha256"]}
        pairs.append(pair)
        print("BYTE_COMPARISON",remote["source_key"],"identical",pair["bytes_match"] and pair["sha256_match"],flush=True)
    manifest["downloaded_files"]=downloads;manifest["download_completed_at_utc"]=timestamp()
    dump("source_manifest.json",manifest)
    core={"official_source_retrieval":True,"compared_at_utc":timestamp(),"files":pairs,
          "all_byte_identical":all(p["bytes_match"] and p["sha256_match"] for p in pairs),
          "read_original_source_only":True,"max_parallel_workers":3}
    dump("byte_comparison.json",core)
    print("BYTE_AUDIT_COMPLETE",json.dumps({"all_byte_identical":core["all_byte_identical"],"files":len(pairs)}),flush=True)

if __name__=="__main__":main()

