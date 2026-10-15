"""One read-only full19-column scan with six rules and full AE; IF on candidates.

Dense raw windows are not copied. Events retain exact source indices/timestamps
and peak original observations for the parent mmap replay loader.
"""
from pathlib import Path
import sys,io,json,time,hashlib,gzip
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from detector import OUT,RAW,TYPE_IDS,TYPE_LABELS,prepare,Detector
from train_system import write,sha
ROOT=OUT.parents[2];SOURCE=ROOT/'traing-dataset/rtu_data_full.csv'
EXPECTED='3b4e1ff2125ca64612837530a8612fa7f044f398e508e00026097fc788d0dce5'

class HashingRaw(io.RawIOBase):
    def __init__(self,path):self.handle=open(path,'rb');self.digest=hashlib.sha256();self.bytes=0
    def readable(self):return True
    def readinto(self,b):
        data=self.handle.read(len(b));n=len(data);b[:n]=data;self.digest.update(data);self.bytes+=n;return n
    def close(self):
        self.handle.close();super().close()

def main():
    detector=Detector();start=time.monotonic();before=SOURCE.stat();model_hash={n:sha(OUT/n) for n in ['models.joblib','ae.pt','synthetic_xgb.json']}
    dtype={c:'float64' for c in RAW};dtype.update({'module(equipment)':'string','timestamp':'int64','localtime':'int64','operation':'int8','accumActiveEnergy':'int64'})
    source=HashingRaw(SOURCE);buffer=io.BufferedReader(source,buffer_size=1<<20);rows=0;counts={t:0 for t in TYPE_IDS};ae_count=0
    module_counts={};last_seconds={};invalid_diffs={};missing_counts={};events=[];open_events={};source_eq={};peaks=[]
    def finish(key):
        if key not in open_events:return
        ev=open_events.pop(key);ev['duration_seconds']=ev['point_count']*5;ev['sustained_3points']=ev['point_count']>=3
        ev['status']='unconfirmed_candidate';ev['source_kind']='recorded';ev['confidence']='repeated_context_deviation' if ev['sustained_3points'] else 'isolated_point_review'
        events.append(ev)
    try:
        for chunk_index,chunk in enumerate(pd.read_csv(buffer,chunksize=250000,dtype=dtype,float_precision='round_trip'),1):
            for c,n in chunk.isna().sum().items():missing_counts[c]=missing_counts.get(c,0)+int(n)
            f=prepare(chunk);f['source_row_index']=np.arange(rows,rows+len(f),dtype=np.int64)
            for m,g in f.groupby('module_id',sort=False):
                sec=g.dt.to_numpy(dtype='datetime64[s]').astype(np.int64);previous=last_seconds.get(m)
                bad=int(np.sum(np.diff(sec)!=5))+(int(sec[0]-previous!=5) if previous is not None else 0)
                invalid_diffs[m]=invalid_diffs.get(m,0)+bad;last_seconds[m]=int(sec[-1]);module_counts[m]=module_counts.get(m,0)+len(g)
                source_eq[m]=str(g['module(equipment)'].iloc[0])
            scored=detector.scores(f,False);flags=scored['rule_flags'].copy();flags[:,3] |= scored['ae_alarm']
            ae_count+=int(scored['ae_alarm'].sum())
            for col,t in enumerate(TYPE_IDS):
                mask=flags[:,col];counts[t]+=int(mask.sum())
                loc=np.flatnonzero(mask)
                for i in loc:
                    row=f.iloc[i];m=str(row.module_id);key=(m,t);now=row['dt'];rawscore=float(scored['rule_scores'][i,col]);threshold=float(scored['rule_thresholds'][i,col])
                    if t=='abnormal_pattern' and scored['ae_alarm'][i]:rawscore=max(rawscore,float(scored['ae_score'][i]/detector.saved['thresholds']['ae']*threshold))
                    normalized=rawscore/max(threshold,1e-12)
                    if key in open_events and (now-pd.Timestamp(open_events[key]['end_time'])).total_seconds()!=5:finish(key)
                    if key not in open_events:
                        open_events[key]={'type':t,'label':TYPE_LABELS[col],'module_id':m,'equipment':source_eq[m],'start_time':now.isoformat(),'end_time':now.isoformat(),
                             'start_source_row':int(row.source_row_index),'end_source_row':int(row.source_row_index),'point_count':0,'score':-1.,'threshold':threshold,'normalized_score':-1.}
                    ev=open_events[key];ev['point_count']+=1;ev['end_time']=now.isoformat();ev['end_source_row']=int(row.source_row_index)
                    if normalized>ev['normalized_score']:
                        ev.update(score=rawscore,normalized_score=normalized,peak_time=now.isoformat(),peak_source_row=int(row.source_row_index),
                                  ae_score=float(scored['ae_score'][i]),ae_alarm=bool(scored['ae_alarm'][i]),
                                  raw_features={c:(int(row[c]) if c in ['timestamp','localtime','operation','accumActiveEnergy'] else float(row[c])) for c in chunk.columns if c!='module(equipment)'},
                                  derived_features={c:float(v) for c,v in scored['derived'].iloc[i].items()},
                                  evidence='Exceeds frozenJanuary contextual threshold. Actual fault label/type remains unknown; peak original features retained for replay.')
            rows+=len(f)
            if chunk_index%4==0:print(f'fullscan rows{rows:,} elapsed{time.monotonic()-start:.1f}s candidate_points{sum(counts.values()):,}',flush=True)
        for key in list(open_events):finish(key)
        source_hash=source.digest.hexdigest();source_bytes=source.bytes
    finally:buffer.close()
    after=SOURCE.stat();assert rows==33696013 and source_bytes==before.st_size and source_hash==EXPECTED
    assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns and not any(invalid_diffs.values()) and not any(missing_counts.values())
    assert model_hash=={n:sha(OUT/n) for n in model_hash}
    # Keep complete event audit separate from the compact frontend candidate list.
    flat=[]
    for ev in events:
        f={k:v for k,v in ev.items() if k not in ['raw_features','derived_features']};f['raw_features_json']=json.dumps(ev['raw_features'],ensure_ascii=False);f['derived_features_json']=json.dumps(ev['derived_features'],ensure_ascii=False);flat.append(f)
    pd.DataFrame(flat).to_csv(OUT/'all_rule_events.csv.gz',index=False,compression='gzip',float_format='%.17g')
    candidates=[];summary={}
    for t in TYPE_IDS:
        group=[e for e in events if e['type']==t];sustained=[e for e in group if e['sustained_3points']]
        ranking=sorted(group,key=lambda e:(e['sustained_3points'],e['normalized_score'],e['point_count']),reverse=True)
        selected=ranking[:30]
        for e in selected:
            e['event_id']=f'{t}_{len(candidates)+1:04d}'
            e.update(id=e['event_id'],title=e['label']+' 후보',start=e['start_time'],end=e['end_time'],time=e['peak_time'])
            e['evaluation_scope']='chronological_after_calibration' if pd.Timestamp(e['peak_time'])>=pd.Timestamp('2025-02-01') else 'retrospective_fit_or_calibration_period'
            candidates.append(e)
        summary[t]={'raw_alarm_points':counts[t],'all_contiguous_events':len(group),'sustained_3point_events':len(sustained),'frontend_review_candidates':len(selected),
                    'status':'unconfirmed_natural_candidates' if sustained else 'no_sustained_candidate_found',
                    'coverage_note':'Single-point candidates, if present, are not a found fault; unmatched forms use separately tagged synthetic demonstrations.'}
    # Full IF/classifier scores only for the exact peak observations shown in UI.
    if candidates:
        peakrows=[{'module(equipment)':e['equipment'],**e['raw_features']} for e in candidates]
        peak_ai=detector.diagnose_batch(peakrows)
        for e,ai in zip(candidates,peak_ai):e['ai']=ai['ai'];e['rules_at_peak']=ai['rules'];e['confidence_detail']=ai['confidence']
    write(OUT/'candidates.json',{'status':'fullscan_complete','source_kind':'recorded','actual_fault_labels_available':False,'ground_truth_injected_count':None,
                   'source':str(SOURCE),'source_sha256':source_hash,'types':summary,'candidates':candidates,
                   'selection':'Up to30 strongest evidence-ranked events pertype, sustained3point events first; count is not optimized to match six faults.',
                   'replay':'Load dense original5s rows by module_id,start/end/peak time or exactzero-basedsource indices. No full raw dataset copy.'})
    pd.DataFrame([{k:v for k,v in e.items() if k not in ['ai','rules_at_peak','raw_features','derived_features','confidence_detail']} for e in candidates]).to_csv(OUT/'candidates.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    result=json.loads((OUT/'results.json').read_text(encoding='utf-8'));result['status']='complete';result['full_scan']={
        'source':str(SOURCE),'rows_read':rows,'columns_read':list(dtype),'explicit_dtype':dtype,'module_rows':module_counts,'source_sha256':source_hash,
        'source_bytes':source_bytes,'source_size_mtime_preserved':True,'localtime_format':'%Y%m%d%H%M%S,naive recordedtime; Unix offset key ignored',
        'per_module_non5s_diffs':invalid_diffs,'missing_values':missing_counts,'all_six_rules_scored_on_all_rows':True,'AE_scored_on_all_rows':True,
        'IF_and_classifier_coverage':'187200stratified original sample for calibration/comparison and exact peak rows of up to180frontend candidates; not all33.7M rows.',
        'AE_alarm_rows':ae_count,'point_and_event_counts':summary,'all_events_csv':str(OUT/'all_rule_events.csv.gz'),
        'candidates_json':str(OUT/'candidates.json'),'candidates_csv':str(OUT/'candidates.csv'),'elapsed_seconds':time.monotonic()-start,
        'no_actual_precision_recall':'Unknown inserted type/timestamps. Neither rawalarm count nor synthetic score reveals actual precision, recall or number of real faults.'}
    write(OUT/'results.json',result);write(OUT/'full_scan_audit.json',result['full_scan']);write(OUT/'scan_summary.json',result['full_scan'])
    print(json.dumps({'status':'fullscan_complete','rows':rows,'sourceSHA':source_hash,'types':summary,'elapsed':time.monotonic()-start},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
