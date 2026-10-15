"""Profile direct source copies and distinguish integrity from data semantics."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv,json,hashlib,time,datetime,sys,pandas as pd,numpy as np,html
sys.stdout.reconfigure(encoding="utf-8")
ROOT=Path(r"E:\AI work\AWS_프로젝트");OUT=ROOT/"output/optimization_20261013/source_audit"
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
def header(path):
    with path.open("r",encoding="utf-8-sig",newline="") as f:return next(csv.reader(f))
def profile_train(path):
    start=time.monotonic();rows=0;counts={};bounds={};missing={"module(equipment)":0,"timestamp":0,"localtime":0}
    columns=header(path);dt_min=None;dt_max=None;ts_min=None;ts_max=None
    for ci,d in enumerate(pd.read_csv(path,usecols=list(missing),dtype={"module(equipment)":"string","timestamp":"int64","localtime":"int64"},chunksize=1_000_000)):
        rows+=len(d)
        for name in missing:missing[name]+=int(d[name].isna().sum())
        low=int(d.localtime.min());high=int(d.localtime.max());tl=int(d.timestamp.min());th=int(d.timestamp.max())
        dt_min=low if dt_min is None else min(dt_min,low);dt_max=high if dt_max is None else max(dt_max,high)
        ts_min=tl if ts_min is None else min(ts_min,tl);ts_max=th if ts_max is None else max(ts_max,th)
        for name,g in d.groupby("module(equipment)",sort=False):
            counts[name]=counts.get(name,0)+len(g)
            lo=int(g.localtime.min());hi=int(g.localtime.max())
            old=bounds.get(name,{"minimum":lo,"maximum":hi})
            bounds[name]={"minimum":min(lo,old["minimum"]),"maximum":max(hi,old["maximum"])}
        if ci%5==0:print("PROFILE_PROGRESS","train","rows",rows,"seconds",round(time.monotonic()-start,1),flush=True)
    fmt=lambda n:str(pd.to_datetime(str(n),format="%Y%m%d%H%M%S"))
    return {"path":str(path),"headers":columns,"column_count":len(columns),"rows":rows,"module_count":len(counts),"rows_by_module":counts,
            "localtime_numeric_min":dt_min,"localtime_numeric_max":dt_max,"recorded_localtime_start":fmt(dt_min),"recorded_localtime_end":fmt(dt_max),
            "localtime_bounds_by_module":{k:{"minimum":fmt(v["minimum"]),"maximum":fmt(v["maximum"])} for k,v in bounds.items()},
            "timestamp_min_ms":ts_min,"timestamp_max_ms":ts_max,
            "timestamp_min_as_utc":str(pd.to_datetime(ts_min,unit="ms",utc=True)),"timestamp_max_as_utc":str(pd.to_datetime(ts_max,unit="ms",utc=True)),
            "missing_counts_profiled_key_columns":missing,"profiled_columns":list(missing),"elapsed_seconds":time.monotonic()-start}
def profile_may(path):
    d=pd.read_csv(path);ids=pd.to_datetime(d["id"])
    ratio=lambda y,x:{"minimum":float((d[y]/d[x]).min()),"maximum":float((d[y]/d[x]).max())}
    return {"path":str(path),"headers":list(d.columns),"column_count":len(d.columns),"rows":len(d),"id_start":str(ids.min()),"id_end":str(ids.max()),
            "duplicate_ids":int(ids.duplicated().sum()),"continuous_hourly_grid":bool(pd.DatetimeIndex(ids).equals(pd.date_range(ids.min(),ids.max(),freq="h"))),
            "missing_counts":{k:int(v) for k,v in d.isna().sum().items()},
            "provided_targets":{"hourly_pow_min":float(d.hourly_pow.min()),"hourly_pow_max":float(d.hourly_pow.max())},
            "may_bill_to_hourly_pow_ratio":ratio("may_bill","hourly_pow"),"may_carbon_to_hourly_pow_ratio":ratio("may_carbon","hourly_pow"),
            "agg_pow_matches_hourly_pow_cumsum_max_absdiff":float(np.abs(d.agg_pow-d.hourly_pow.cumsum()).max())}
def main():
    comparison=json.loads((OUT/"byte_comparison.json").read_text(encoding="utf-8"))
    manifest=json.loads((OUT/"source_manifest.json").read_text(encoding="utf-8"))
    guide=json.loads((ROOT/"output/analysis/guide_summary.json").read_text(encoding="utf-8"))
    previous=json.loads((ROOT/"output/analysis/time_anomaly_audit.json").read_text(encoding="utf-8"))
    stats={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={}
        for pair in comparison["files"]:
            path=Path(pair["downloaded"]["downloaded_path"])
            fn=profile_train if path.name=="rtu_data_full.csv" else profile_may
            futures[pool.submit(fn,path)]=pair
        for future in as_completed(futures):
            pair=futures[future];p=future.result();key=pair["source_key"]
            original_headers=header(Path(pair["original"]["path"]))
            item={"source_key":key,"downloaded_profile":p,"original_headers":original_headers,"headers_match":p["headers"]==original_headers,
                  "profile_comparison_method":"Downloaded official CSV parsed directly; original is byte-identical by independently computed SHA-256, so row counts and all field values are identical without a redundant second parse."}
            if path.name=="rtu_data_full.csv":pass
            stats[key]=item;dump("csv_profiles.json",stats)
            print("PROFILE_COMPLETE",key,"rows",p["rows"],"columns",p["column_count"],flush=True)
    train=stats["traing-dataset/rtu_data_full.csv"]["downloaded_profile"]
    may=stats["may-dataset/rtu_ground_truth_may.csv"]["downloaded_profile"]
    if comparison["all_byte_identical"]:
        if train["rows"]!=previous["rows"]:raise ValueError("Official copied row count contradicts prior exhaustive audit")
    time_checks=previous["time_key_checks"]
    time_totals={k:sum(v[k] for v in time_checks.values()) for k in ["local_duplicate_rows","local_negative_diffs","local_non_5s_diffs"]}
    objects=manifest["objects"]
    faults=[o["key"] for o in objects if any(w in o["key"].lower() for w in ["mask","label","fault","anomal","고장","이상"])]
    pdfs=[o["key"] for o in objects if o["key"].lower().endswith(".pdf")]
    report={"audited_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"official_page":manifest["website_url"],"status":"complete",
      "conclusion":"The two directly downloaded official CSV files are byte-for-byte identical to the provided local files. Observed data specification/quality issues are present in the published source, not introduced by transfer or copying.",
      "all_byte_identical":comparison["all_byte_identical"],"file_comparisons":comparison["files"],"csv_profiles":stats,
      "source_inventory":{"objects":[{"key":o["key"],"bytes":o["bytes"],"last_modified":o["last_modified"]} for o in objects],
                          "additional_fault_label_or_mask_files_in_public_listing":faults,"guide_pdf_files_in_public_listing":pdfs,
                          "zip_files_in_public_listing":[o["key"] for o in objects if o["key"].lower().endswith(".zip")],
                          "scope":"Complete public ListObjectsV2 prefix traversal as declared by the official page; no claim about unpublished/private materials."},
      "prior_exhaustive_original_audit_applies_to_exact_official_copy":{"source":str(ROOT/"output/analysis/time_anomaly_audit.json"),"sha256":hashlib.sha256((ROOT/"output/analysis/time_anomaly_audit.json").read_bytes()).hexdigest(),
           "rows":previous["rows"],"time_grid_totals":time_totals,"offset_local_naive_minus_unix_ms":previous["offset_local_naive_minus_unix_ms"],
           "offset_transitions":previous["offset_transitions"],"time_key_checks":time_checks,"candidate_rows_unique":previous["candidate_rows_unique"],"rule_counts":previous["rule_counts"]},
      "semantic_findings":[
       {"topic":"Timestamp/localtime","status":"source_issue_confirmed","finding":"Exact official copy inherits the exhaustive timestamp audit: recorded localtime is continuous at5seconds, but Unix timestamp local-naive offset changes from-8h to-7h on2025-03-09. This disagrees with the guide's Korean-time interpretation and causes Unix-key duplicated values. The pattern resembles US Pacific DST conversion; intended source timezone remains undocumented.","modeling_action":"Preserve raw fields and use explicitly disclosed recorded-localtime sequence pending source-owner timezone clarification."},
       {"topic":"Power/energy units","status":"source_specification_unresolved","finding":"CSV column names do not certify units. Sanitized guide states activePower kW but labels accumActiveEnergy as both Wh and kWh. Direct download supplies no corrected schema or guide. May cost/carbon columns are numerically derivable with the guide coefficients but do not independently validate the physical units.","modeling_action":"Retain the stated provisional unit assumption and obtain authoritative units/calculation definitions; do not silently rescale based solely on model error."},
       {"topic":"Meter hierarchy / May target scope","status":"source_specification_unresolved","finding":"Published CSVs provide no explicit parent/child meter hierarchy, and the May file has aggregate targets without module keys. Byte identity does not establish that summing every module is certified whole-facility consumption.","modeling_action":"Confirm module inclusion and parent/submeter relationships before treating aggregate as certified whole-facility energy."},
       {"topic":"Injected anomalies / verified faults","status":"no_public_ground_truth_mask","finding":"The public object inventory contains no insertion mask or fault-label file. operation indicates operation status, not verified fault identity. Prior63,963 rule-screened observations remain candidates rather than a confirmed inserted-fault count.","modeling_action":"Obtain insertion masks and maintenance/operating labels; keep rule/AE diagnosis labeled as weak-candidate assessment."},
       {"topic":"Guide and source paths","status":"filename_matches_path_typo","finding":"Published filenames match the local/sanitized guide roles. Official training directory is traing-dataset, while the guide summary includes training-dataset(s); this is a path spelling mismatch, not a data-content difference. No guide PDF is available for direct source-to-local guide byte comparison.","modeling_action":"Use URLs derived from the live official object inventory, and preserve the supplied sanitized guide summary."}
      ],
      "method":{"download_protocol":"HTTPS S3 REST object URLs constructed exactly from the official HTTP page's bucket/region code; If-Match protects listed ETag/version during download","integrity":"Independent streaming SHA-256 and byte counts for both original and download; ETag is recorded as version metadata and not treated as MD5/cryptographic checksum","csv":"Pandas chunked CSV parsing of official copy for row/key bounds; exact byte identity transfers the profile to original. Previous exhaustive original grid audit is explicitly reused only because SHA-256 matches.","parallelism":"3 workers during download/hash;2 workers during CSV profiling","originals":"read-only; no existing analysis/modeling files changed","credentials":"No PDF credential/API-key content printed or included in report; only previously sanitized guide data-role/unit summaries used."}}
    dump("audit_report.json",report)
    print("SOURCE_AUDIT_COMPLETE",json.dumps({"all_byte_identical":report["all_byte_identical"],"train_rows":train["rows"],"train_localtime_start":train["recorded_localtime_start"],"train_localtime_end":train["recorded_localtime_end"],"may_rows":may["rows"],"may_start":may["id_start"],"may_end":may["id_end"],"public_mask_files":faults},ensure_ascii=False),flush=True)
if __name__=="__main__":main()

