"""Reconcile descriptive figures/tables, decomposition and final deliverables."""
from pathlib import Path
import sys,json,hashlib,datetime,uuid,zipfile
sys.dont_write_bytecode=True
import pandas as pd
import numpy as np
from pypdf import PdfReader
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def main():
    checks={};facts=read(OUT/'report_facts.json');e=read(OUT/'evidence/evidence_summary.json');d=read(OUT/'decomposition/results.json');v=read(OUT/'visuals/results.json');details={}
    checks['evidence_audit_passed']=read(OUT/'evidence/evidence_verification.json')['status']=='pass'
    checks['visuals_QA_passed']=read(OUT/'visuals/visual_qa.json')['status']=='pass'
    checks['decomposition_QA_passed']=read(OUT/'decomposition/verification.json')['status']=='passed'
    checks['full_scope_reconciled']=facts['full']['rows']==e['full']['rows']==33696013 and len(facts['full']['columns'])==e['full']['columns']==19 and facts['full']['module_count']==e['full']['module_count']==13
    checks['full_period_cadence_reconciled']=all(facts['full'][k]==e['full'][k]for k in ['first','last','sampling_seconds','missing_total'])
    checks['May_scope_reconciled']=facts['may']['rows']==e['may']['rows']==505 and len(facts['may']['columns'])==e['may']['columns']==5 and all(facts['may'][k]==e['may'][k]for k in ['first','last','sampling_seconds','missing_total'])
    checks['raw_5second_prefix_verified']=facts['raw_prefix']['rows_read']==17280 and facts['raw_prefix']['cadence_seconds']==5 and facts['raw_prefix']['localtime_parser'].endswith('%Y%m%d%H%M%S')
    fd=pd.read_csv(OUT/'evidence/feature_dictionary.csv');checks['feature_dictionary24_rows']=len(fd)==24 and (fd.dataset=='full').sum()==19 and (fd.dataset=='may').sum()==5
    nr=pd.DataFrame(facts['original_numeric_statistics']);ff=fd[fd.dataset=='full'].set_index('name')
    err={}
    for field in ['mean','std','min','max']:
        actual=ff.loc[nr.name,field].astype(float).to_numpy();ref=nr[field].to_numpy(float);delta=float(np.max(abs(actual-ref)));err[field]=delta
        assert np.allclose(actual,ref,rtol=1e-10,atol=1e-7),(field,delta)
    checks['full15_sensor_population_statistics_reconciled']=True;details['sensor_statistic_max_absolute_errors']=err
    total=facts['target']['total_provisional_kwh'];period_checks=[]
    for grain,expected,complete in [('hourly',3600,3600),('daily',150,150),('weekly',23,21),('monthly',5,4)]:
        f=pd.read_csv(OUT/'data'/f'{grain}_dataset.csv',float_precision='round_trip');good=f.complete.astype(str).str.lower()=='true'
        assert len(f)==expected and int(good.sum())==complete
        enerr=float(abs(f.energy_kwh.sum()-total));rateerr=float(abs(f.energy_kwh/f.observed_hours-f.rate_kwh_per_hour).max());assert enerr<1e-6 and rateerr<1e-7
        assert f.period_start.nunique()==len(f) and f.observed_hours.sum()==3600 and not f.energy_kwh.isna().any()
        period_checks.append(dict(grain=grain,rows=len(f),complete=int(good.sum()),sum_error=enerr,rate_error=rateerr))
    checks['all_resolution_energy_coverage_and_rate_consistency']=True;details['resampling']=period_checks
    for key,c in d['cases'].items():
        f=pd.read_csv(c['components_csv'],float_precision='round_trip');cols=list(f)
        # Independent CSV fields and additive identity, respecting native units.
        obs=next(x for x in ['observed','observed_kwh','observed_rate_kwh_per_hour']if x in cols)
        trend=next(x for x in ['trend','trend_kwh','trend_rate']if x in cols)
        resid=next(x for x in ['resid','residual','residual_kwh']if x in cols)
        seasons=[x for x in cols if x.startswith('seasonal')]
        if 'seasonal_total'in cols:seasons=['seasonal_total']
        elif 'seasonal'in cols:seasons=['seasonal']
        if 'seasonal_24'in cols:assert float(abs(f.seasonal-f.seasonal_24-f.seasonal_168).max())<1e-7
        assert seasons,(key,cols)
        residual_err=float(abs(f[obs]-f[trend]-f[resid]-f[seasons].sum(axis=1)).max());assert residual_err<1e-7
        assert len(f)==(150 if key=='daily_stl7' else 3600)
    checks['all4_decompositions_additive_reconstruction']=True
    checks['no_annual_week_month_decomposition']=not d['short_series_limits']['weekly']['decomposition_performed'] and not d['short_series_limits']['monthly']['decomposition_performed']
    checks['no_decomposition_May_fit_or_settings_search']=not d['May_data_used'] and not d['settings_search_performed']
    pdf=OUT/'pdf/basic_data_report.pdf';reader=PdfReader(pdf);checks['pdf18pages_text_and_unicode']=len(reader.pages)==18 and all(p.extract_text()and'\ufffd'not in p.extract_text()for p in reader.pages)
    text='\n'.join(p.extract_text()for p in reader.pages)
    checks['pdf_contains_expected_topics']=all(t in text for t in ['33,696,013','2024-12-01','2025-04-30','2025-05-10','5초','19개 피처','24시간','168시간','7일','MSTL','원본','localtime'])
    checks['pdf_font_embedding']=all('/FontDescriptor'in f.get_object()and'/FontFile2'in f.get_object()['/FontDescriptor'].get_object() for p in reader.pages for _,f in p['/Resources']['/Font'].get_object().items()if'/BaseFont'in f.get_object()and'Malgun'in str(f.get_object()['/BaseFont']))
    render=read(OUT/'render_pack_qa.json');checks['pdf18pages_rendered']=render['rendered_pages']==18
    with zipfile.ZipFile(OUT/'figures_and_tables.zip')as z:
        checks['archive_integrity']=z.testzip()is None and len(z.namelist())==64 and len([n for n in z.namelist()if n.endswith('.png')])==20 and len([n for n in z.namelist()if n.endswith('.svg')])==20
        assert all(not n.startswith('/')and '..'not in Path(n).parts for n in z.namelist())
    b=read(OUT/'browser_qa.json');checks['html_QA_passed']=b['pass'];details['browser']=b
    q=OUT/'pdf_visual_qa.json'
    checks['PDF_visual_QA_record_exists']=q.exists()
    if q.exists():details['PDF_visual_QA']=read(q)
    originals=[]
    for row in read(ROOT/'output/optimization_20261013/source_audit/byte_comparison.json')['files']:
        old=row['original'];p=Path(old['path']);s=p.stat();okay=s.st_size==old['bytes'] and s.st_mtime_ns==old['mtime_ns'];assert okay
        originals.append(dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,preserved_size_and_mtime=okay,prior_full_sha256=old['sha256']))
    checks['original_data_preserved']=True;details['originals']=originals
    checks={k:bool(value)for k,value in checks.items()}
    result={'pass':all(checks.values()),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'details':details,
      'limitations':['Physical target units and meter hierarchy unresolved','Recorded localtime is not independently confirmed Korea timezone','Month4 and week21 cannot support annual seasonal decomposition','Descriptive full-history decomposition contains future smoothing and is not a historical forecast input'],
      'original_copies_created':0,'original_moves':0,'original_deletes':0,'source_registry_modified':False,'model_training_performed':False}
    write(OUT/'verification.json',result)
    inventory=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file()or p.name=='artifact_inventory.json'or any(x in p.parts for x in ['tmp','.mplconfig','__pycache__']):continue
        rel=p.relative_to(OUT).as_posix();inventory.append(dict(file_id=str(uuid.uuid5(uuid.NAMESPACE_URL,'ai-work-e/AWS/data_overview_20261013/'+rel)),project_id=None,workspace_id='ai-work-e',relative_path=rel,bytes=p.stat().st_size,sha256=sha(p),original_path=None,copy_status='generated_not_original_copy',verification_status='size_and_SHA256_recorded',generated_by_ai=True))
    write(OUT/'artifact_inventory.json',dict(artifacts=inventory,source_registry_unchanged=True,original_copies=0))
    print(json.dumps({'pass':result['pass'],'checks':checks,'artifact_count':len(inventory)},ensure_ascii=True))
    if not result['pass']:sys.exit(1)
if __name__=='__main__':main()
