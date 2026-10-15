"""Indexed, immutable access to actual 5-second source rows.

The source is module-blocked and strictly ordered by recorded localtime. A
validated byte index + binary search reads dense windows without scanning or
copying the 4.9GB source on every UI request. Unix clock errors are not used as
the ordering key. Historical hourly sensor summaries are separately audited.
"""
from pathlib import Path
import csv,io,json,mmap,threading,hashlib
import numpy as np
import pandas as pd
from config import APP,ROOT,DATA
SOURCE=ROOT/'traing-dataset/rtu_data_full.csv'
SENSOR_KEYS=['voltageR','voltageS','voltageT','voltageRS','voltageST','voltageTR','currentR','currentS','currentT','activePower','powerFactorR','powerFactorS','powerFactorT','reactivePowerLagging','accumActiveEnergy','operation']
LABELS={'voltageR':'R상 전압','voltageS':'S상 전압','voltageT':'T상 전압','voltageRS':'R-S 선간전압','voltageST':'S-T 선간전압','voltageTR':'T-R 선간전압','currentR':'R상 전류','currentS':'S상 전류','currentT':'T상 전류','activePower':'유효전력','powerFactorR':'R상 역률','powerFactorS':'S상 역률','powerFactorT':'T상 역률','reactivePowerLagging':'무효전력','accumActiveEnergy':'누적 유효전력량','operation':'가동 상태','energy_kwh':'시간 전력량','pf_mean':'평균 역률','current_unbalance_pct':'전류 불균형 지표'}
def unit(key):
    if key.startswith('voltage'):return 'V'
    if key.startswith('current')and key!='current_unbalance_pct':return 'A'
    if key.startswith('powerFactor')or key in ['pf_mean','current_unbalance_pct']:return '%'
    return {'activePower':'kW · 가이드 기준','reactivePowerLagging':'kVar · 가이드 기준','accumActiveEnergy':'원자료 누적량','operation':'상태','energy_kwh':'kWh · 잠정'}.get(key,'원자료')
def iso(t):return pd.Timestamp(t).strftime('%Y-%m-%d %H:%M:%S')
def clean_row(row):
    r={str(k):(v.item()if isinstance(v,np.generic)else v)for k,v in row.items()}
    for k,v in list(r.items()):
        if isinstance(v,float)and not np.isfinite(v):r[k]=None
        if isinstance(v,pd.Timestamp):r[k]=iso(v)
    return r
class SourceStore:
    def __init__(self):
        self.lock=threading.RLock();self.file=SOURCE.open('rb');self.mm=mmap.mmap(self.file.fileno(),0,access=mmap.ACCESS_READ)
        headend=self.mm.find(b'\n');self.header=next(csv.reader([self.mm[:headend].decode('utf-8-sig').strip()]))
        self.profile=json.loads((ROOT/'output/analysis/training_profile.json').read_text(encoding='utf8'))
        self.modules=[{'id':tag.split('(')[0],'name':tag.split('(',1)[1][:-1],'source_tag':tag}for tag in self.profile['equipment_blocks']]
        self.bounds=self.build_index();self.hourly=self.load_hourly();self.complete_target=self.hourly.groupby('time').energy_kwh.sum()
        self.cache={};self.series_availability=self.build_availability()
    def parse_line(self,start,end=None):
        if end is None:end=self.mm.find(b'\n',start)
        if end<0:end=len(self.mm)
        return next(csv.reader([self.mm[start:end].decode('utf8').rstrip('\r')]))
    def build_index(self):
        s=SOURCE.stat();dest=DATA/'source_index.json'
        if dest.exists():
            old=json.loads(dest.read_text(encoding='utf8'))
            if old['bytes']==s.st_size and old['mtime_ns']==s.st_mtime_ns:return old['bounds']
        starts=[];cursor=0
        for m in self.modules:
            found=self.mm.find(b'\n'+m['source_tag'].encode('utf8')+b',',cursor)
            if found<0:raise RuntimeError('원본 모듈 블록을 확인할 수 없습니다.')
            starts.append(found+1);cursor=found+1
        result={}
        for i,m in enumerate(self.modules):
            begin=starts[i];end=starts[i+1]if i+1<len(starts)else len(self.mm)
            lastend=end
            while lastend>begin and self.mm[lastend-1]in [10,13]:lastend-=1
            laststart=self.mm.rfind(b'\n',begin,lastend)+1
            first=self.parse_line(begin);last=self.parse_line(laststart,lastend)
            assert first[0]==last[0]==m['source_tag']
            assert len(first[2])==len(last[2])==14 and first[2].isdigit()and last[2].isdigit()and first[2]<=last[2]
            result[m['id']]={'start':begin,'end':end,'first':first[2],'last':last[2],'source_tag':m['source_tag']}
        dest.write_text(json.dumps({'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sort_key':'recorded localtime YYYYMMDDHHMMSS','bounds':result,'original_modified':False},ensure_ascii=False,indent=2),encoding='utf8')
        return result
    def raw_range(self,module_id='all'):
        ids=list(self.bounds)if str(module_id)=='all'else[str(module_id)]
        if any(mid not in self.bounds for mid in ids):raise ValueError('알 수 없는 설비입니다.')
        first=max(pd.to_datetime(self.bounds[mid]['first'],format='%Y%m%d%H%M%S')for mid in ids)
        last=min(pd.to_datetime(self.bounds[mid]['last'],format='%Y%m%d%H%M%S')for mid in ids)
        return first,last
    def build_availability(self):
        by_module={}
        for mid in ['all']+list(self.bounds):
            first,last=self.raw_range(mid);end=last+pd.Timedelta(seconds=5)
            dates=[str(d.date())for d in pd.date_range(first.normalize(),last.normalize(),freq='D')]
            day_samples=[]
            for day in pd.date_range(first.normalize(),last.normalize(),freq='D'):
                a=max(first,day);b=min(end,day+pd.Timedelta(days=1));n=max(0,int((b-a).total_seconds()//5))
                day_samples.append({'date':str(day.date()),'observed_samples':n,'complete':n==17280})
            raw={'available':first<=last,'min_date':dates[0],'max_date':dates[-1],'available_dates':dates,
                 'first_timestamp':iso(first),'last_timestamp':iso(last),'end_exclusive':iso(end),
                 'coverage_start':iso(first),'period_count':int((last-first).total_seconds()//5)+1,
                 'complete_period_count':int((last-first).total_seconds()//5)+1,'max_query_hours':24,
                 'partial_first_period':False,'partial_last_period':False,'day_coverage':day_samples,
                 'partial_first_day':not day_samples[0]['complete'],'partial_last_day':not day_samples[-1]['complete'],
                 'first_day_samples':day_samples[0]['observed_samples'],'last_day_samples':day_samples[-1]['observed_samples'],
                 'timestamp_semantics':'recorded_sample_time','count_basis':'audited continuous five-second grid, per simultaneous timestamp'}
            scopes={'raw':raw}
            f=self.hourly if mid=='all'else self.hourly[self.hourly.module_id==mid]
            times=pd.DatetimeIndex(sorted(f.time.unique()))
            if mid=='all':
                counts=f.groupby('time').module_id.nunique();times=pd.DatetimeIndex(counts[counts==len(self.bounds)].index).sort_values()
            for grain,freq in [('hour','h'),('day','D'),('week','W-SUN'),('month','MS')]:
                if times.empty:
                    scopes[grain]={'available':False,'min_date':None,'max_date':None,'available_dates':[],'period_count':0,'complete_period_count':0}
                    continue
                g=pd.Series(1,index=times).resample(freq).count();g=g[g>0]
                expected=pd.Series([1 if grain=='hour'else 24 if grain=='day'else 168 if grain=='week'else int(t.days_in_month)*24 for t in g.index],index=g.index)
                complete=g.eq(expected);valid=g.index[complete];days=sorted({str(t.date())for t in times})
                coverage_start=times[0];coverage_end=times[-1]+pd.Timedelta(hours=1)
                scopes[grain]={'available':True,'min_date':days[0],'max_date':days[-1],'available_dates':days,
                   'first_timestamp':iso(g.index[0]),'last_timestamp':iso(g.index[-1]),'coverage_start':iso(coverage_start),'end_exclusive':iso(coverage_end),
                   'period_count':len(g),'complete_period_count':int(complete.sum()),'partial_first_period':not bool(complete.iloc[0]),'partial_last_period':not bool(complete.iloc[-1]),
                   'first_period_records':int(g.iloc[0]),'last_period_records':int(g.iloc[-1]),'first_period_expected_records':int(expected.iloc[0]),'last_period_expected_records':int(expected.iloc[-1]),
                   'first_complete_period_label':iso(valid[0])if len(valid)else None,'last_complete_period_label':iso(valid[-1])if len(valid)else None,
                   'timestamp_semantics':'week_end_Sunday_label'if grain=='week'else'period_start_label','record_unit':'completed_hour',
                   'includes_partial_periods':True,'partial_period_note':'주·월 끝 라벨은 실제 관측 끝과 다를 수 있습니다. 선택 날짜는 실제 완료시간 관측 범위입니다.'}
            by_module[mid]=scopes
        return {'by_module':by_module,'timezone':'recorded_naive_localtime','source_interval_seconds':5,
                'all_module_scope':'common simultaneous coverage','calendar_end_is_inclusive_day':True,
                'calendar_request_mode':'range_mode=calendar; start/end at midnight; end is selected last date plus one day',
                'timestamp_request_mode':'range_mode=timestamps; [start,end) must be within actual coverage'}
    def validate_series_range(self,module_id,grain,start,end,range_mode='timestamps',explicit=True):
        if range_mode not in ['timestamps','calendar']:raise ValueError('알 수 없는 날짜 범위 방식입니다.')
        scope=self.series_availability['by_module'].get(str(module_id))
        if scope is None:raise ValueError('알 수 없는 설비입니다.')
        if grain not in scope:raise ValueError('알 수 없는 집계 간격입니다.')
        available=scope[grain]
        if not available['available']:
            if explicit:raise ValueError('해당 설비·집계의 완료 관측 자료가 없습니다.')
            return None,None
        first=pd.Timestamp(available['coverage_start']);last=pd.Timestamp(available['end_exclusive'])
        start,end=pd.Timestamp(start),pd.Timestamp(end)
        if pd.isna(start)or pd.isna(end)or start.tzinfo is not None or end.tzinfo is not None:raise ValueError('시간대 없는 원본 기록 시각을 입력하세요.')
        if not explicit:
            end=min(max(end,first),last);start=max(min(start,end),first)
            return start,end
        if start>=end:raise ValueError('조회 시작은 종료보다 빨라야 합니다.')
        if grain=='raw'and end-start>pd.Timedelta(hours=24):raise ValueError('원본5초 간격은24시간 이하로 조회하세요. 긴 기간은 집계를 선택하세요.')
        if range_mode=='calendar':
            if start!=start.normalize()or end!=end.normalize():raise ValueError('날짜 선택 범위는 자정 시작과 포함 종료일 다음 자정으로 지정하세요.')
            if str(start.date())not in available['available_dates']or str((end-pd.Timedelta(days=1)).date())not in available['available_dates']:raise ValueError('선택 날짜가 해당 설비·집계의 실제 보유기간 밖입니다.')
            return max(start,first),min(end,last)
        if start<first or end>last:raise ValueError('조회 범위가 해당 설비·집계의 실제 보유기간 밖입니다.')
        return start,end
    def lower_bound(self,module_id,timecode):
        b=self.bounds[str(module_id)];lo=int(b['start']);hi=int(b['end']);begin=lo
        if timecode<=int(b['first']):return lo
        if timecode>int(b['last']):return hi
        steps=0
        while lo<hi:
            steps+=1
            if steps>80:raise RuntimeError('원본 시간 인덱스 탐색이 수렴하지 않았습니다.')
            mid=(lo+hi)//2;before=self.mm.rfind(b'\n',begin,mid);start=begin if before<0 else before+1
            if start<lo:start=lo
            stop=self.mm.find(b'\n',start,int(b['end']));stop=int(b['end'])if stop<0 else stop
            fields=self.mm[start:stop].split(b',',3);key=int(fields[2])
            if key<timecode:lo=min(stop+1,int(b['end']))
            else:hi=start
        return lo
    def window(self,start,end,module_ids=None,max_hours=24):
        start,end=pd.Timestamp(start),pd.Timestamp(end)
        if end<=start:return []
        if (end-start)>pd.Timedelta(hours=max_hours):raise ValueError('원본 조회는 한 번에24시간 이하로 지정하세요.')
        ids=[m['id']for m in self.modules]if module_ids is None else [str(i)for i in module_ids]
        allrows=[]
        for mid in ids:
            if mid not in self.bounds:raise ValueError('알 수 없는 설비입니다.')
            a=self.lower_bound(mid,int(start.strftime('%Y%m%d%H%M%S')));b=self.lower_bound(mid,int(end.strftime('%Y%m%d%H%M%S')))
            if b<=a:continue
            text=self.mm[a:b].decode('utf8');reader=csv.reader(io.StringIO(text))
            for values in reader:
                if not values:continue
                raw=dict(zip(self.header,values));t=pd.to_datetime(raw['localtime'],format='%Y%m%d%H%M%S')
                if not(start<=t<end):raise RuntimeError('원본 시간 범위가 인덱스와 다릅니다.')
                r={'module_id':mid,'equipment':raw['module(equipment)'],'module(equipment)':raw['module(equipment)'],'time':iso(t),'localtime':int(raw['localtime']),'timestamp':int(raw['timestamp'])}
                for key in SENSOR_KEYS:r[key]=int(raw[key])if key in ['operation','accumActiveEnergy']else float(raw[key])
                currents=np.array([r['currentR'],r['currentS'],r['currentT']]);mean=currents.mean()
                r['pf_mean']=sum(r[f'powerFactor{x}']for x in 'RST')/3;r['current_unbalance_pct']=float(np.max(abs(currents-mean))/mean*100)if mean>0 else 0.
                r['energy_kwh']=r['activePower']*5/3600
                allrows.append(r)
        allrows.sort(key=lambda r:(r['time'],int(r['module_id'])));return allrows
    def load_hourly(self):
        f=pd.read_csv(ROOT/'output/feature_revision_20261013/data/hourly_features.csv',float_precision='round_trip');times=pd.to_datetime(f.hour);parts=[]
        for module in self.modules:
            mid=module['id'];df=pd.DataFrame({'time':times,'module_id':mid,'equipment':module['source_tag']})
            for sensor in SENSOR_KEYS:
                key=f'm{mid}__{sensor}_mean'
                if key in f:df[sensor]=f[key].to_numpy()
            df['energy_kwh']=f[f'm{mid}__energy_kwh'].to_numpy();df['pf_mean']=f[f'm{mid}__pf_mean'].to_numpy();df['current_unbalance_pct']=f[f'm{mid}__current_unbalance_mean'].to_numpy();df['records']=720;df['operation']=1
            parts.append(df)
        result=pd.concat(parts,ignore_index=True);assert len(result)==46800
        counters=pd.read_csv(ROOT/'output/analysis/hourly_by_equipment.csv',usecols=['module(equipment)','hour','counter_last','complete_720']).rename(columns={'module(equipment)':'equipment'})
        counters=counters[counters.complete_720.astype(str).str.lower().eq('true')].copy();counters['time']=pd.to_datetime(counters.hour)
        result=result.merge(counters[['equipment','time','counter_last']],on=['equipment','time'],how='left',validate='one_to_one').rename(columns={'counter_last':'accumActiveEnergy'})
        return result
    def history_energy(self,module_id='all',cursor=None):
        f=self.hourly
        if module_id!='all':f=f[f.module_id==str(module_id)]
        if cursor is not None:f=f[f.time+pd.Timedelta(hours=1)<=pd.Timestamp(cursor)]
        s=f.groupby('time').energy_kwh.sum().sort_index();return s
    def series(self,module_id,sensor,start,end,grain='hour'):
        if sensor not in SENSOR_KEYS+['energy_kwh','pf_mean','current_unbalance_pct']:raise ValueError('허용된 센서 항목을 선택하세요.')
        if str(module_id)!='all'and str(module_id)not in self.bounds:raise ValueError('알 수 없는 설비입니다.')
        if grain not in ['raw','hour','day','week','month']:raise ValueError('알 수 없는 집계 간격입니다.')
        ids=None if str(module_id)=='all'else[str(module_id)];start,end=pd.Timestamp(start),pd.Timestamp(end)
        if grain=='raw':
            rows=self.window(start,end,ids);df=pd.DataFrame(rows)
            if df.empty:return []
            agg='sum'if sensor in ['activePower','reactivePowerLagging','energy_kwh']and module_id=='all'else'mean'
            g=df.groupby('time')[sensor].agg(agg);return [{'timestamp':str(t),'value':float(v)}for t,v in g.items()]
        f=self.hourly[(self.hourly.time>=start)&(self.hourly.time+pd.Timedelta(hours=1)<=end)].copy()
        if module_id!='all':f=f[f.module_id==str(module_id)]
        if f.empty:return []
        if sensor not in f:raise ValueError('이 항목의 시간 집계는 제공되지 않습니다. 원본 간격으로 조회하세요.')
        agg='sum'if sensor in ['activePower','reactivePowerLagging','energy_kwh']and module_id=='all'else'mean'
        g=f.groupby('time')[sensor].agg(agg)
        freq={'hour':'h','day':'D','week':'W-SUN','month':'MS'}.get(grain)
        if not freq:raise ValueError('알 수 없는 집계 간격입니다.')
        z=g.resample(freq).agg(['sum','mean','min','max','count'])
        out=[]
        for t,row in z.iterrows():
            expected=1 if grain=='hour' else 24 if grain=='day' else 168 if grain=='week' else (t+pd.offsets.MonthBegin(1)-t)/pd.Timedelta(hours=1)
            out.append({'timestamp':iso(t),'value':float(row['sum']if sensor=='energy_kwh'else row['mean']),'minimum':float(row['min']),'maximum':float(row['max']),'records':int(row['count']),'complete':int(row['count'])==expected})
        return out
