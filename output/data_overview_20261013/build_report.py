"""Create PDF and HTML data overview from audited full-population summaries.

No model training, cleaning, imputation or source-file modification occurs.
The report's resampling/decomposition inputs are complete observed hours.
"""
from pathlib import Path
import sys,json,math,html,hashlib,datetime
sys.dont_write_bytecode=True
import pandas as pd
import numpy as np
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
WIDTH,HEIGHT=A4;CONTENT_WIDTH=WIDTH-84
BLUE='#123d66';TEAL='#0f766e';INK='#172b42';GRAY='#526477'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def num(x,places=2):
    if x is None:return '-'
    return f'{float(x):,.{places}f}' if places else f'{int(x):,}'

def basic_facts():
    profile=read(ROOT/'output/analysis/training_profile.json');audit=read(ROOT/'output/analysis/time_anomaly_audit.json')
    may=pd.read_csv(ROOT/'may-dataset/rtu_ground_truth_may.csv',float_precision='round_trip');may['id']=pd.to_datetime(may.id)
    allprofiles=list(profile['profiles'].values());count=sum(p['rows']for p in allprofiles);assert count==33696013
    data=[]
    for col in ['voltageR','voltageS','voltageT','voltageRS','voltageST','voltageTR','currentR','currentS','currentT','activePower','powerFactorR','powerFactorS','powerFactorT','reactivePowerLagging','accumActiveEnergy']:
        s=sum(p['sum'][col]for p in allprofiles);s2=sum(p['sumsq'][col]for p in allprofiles);m=s/count
        data.append(dict(name=col,mean=m,std=math.sqrt(max(0,s2/count-m*m)),min=min(p['min'][col]for p in allprofiles),max=max(p['max'][col]for p in allprofiles),missing=profile['missing'][col]))
    source=ROOT/'output/analysis/baseline_audit/complete_hour_module_sum.csv';f=pd.read_csv(source,float_precision='round_trip');y=pd.Series(f.iloc[:,1].to_numpy(float),index=pd.to_datetime(f.iloc[:,0]))
    assert len(y)==3600 and y.index.equals(pd.date_range('2024-12-01',periods=3600,freq='h'))
    sys.path.insert(0,str(ROOT/'output/resampling_modeling_20261013'))
    from resample_common import periods
    tabledir=OUT/'data';tabledir.mkdir(exist_ok=True);resampled={};y.attrs['origin']=pd.Timestamp('2025-04-30')
    for label,freq in [('hourly','H'),('daily','D'),('weekly','W-SUN'),('monthly','MS')]:
        g=periods(y,freq,complete_only=False);g.to_csv(tabledir/(label+'_dataset.csv'),index=False,encoding='utf-8-sig',float_format='%.17g')
        assert abs(g.energy_kwh.sum()-y.sum())<1e-6
        resampled[label]=dict(rows=len(g),complete=int(g.complete.sum()),partial=int((~g.complete).sum()),energy_sum=float(g.energy_kwh.sum()),source='Audited complete-hour target, freshly reaggregated; no imputation')
    modules=pd.read_csv(ROOT/'output/analysis/hourly_by_equipment.csv',float_precision='round_trip');modules=modules[modules.complete_720.astype(str).str.lower()=='true']
    module_stats=modules.groupby('module(equipment)').agg(energy_kwh=('energy_kwh','sum'),mean_hour_energy=('energy_kwh','mean'),hours=('hour','size'))
    assert len(module_stats)==13 and module_stats.hours.eq(3600).all()
    total=module_stats.energy_kwh.sum();module_stats['share_percent']=100*module_stats.energy_kwh/total
    assert abs(total-y.sum())<1e-6
    cadence=[dict(module=k,local_duplicates=v['local_duplicate_rows'],local_negative=v['local_negative_diffs'],local_non_5s=v['local_non_5s_diffs'],unix_duplicates=v['unix_duplicate_rows'])for k,v in audit['time_key_checks'].items()]
    out=dict(full=dict(rows=count,columns=profile['columns'],module_count=len(allprofiles),rows_per_module=2592001,first='2024-12-01 00:00:00',last='2025-04-30 00:00:00',sampling_seconds=5,span_days=150,missing_total=sum(profile['missing'].values()),bytes=profile['source_bytes']),
      may=dict(rows=len(may),columns=list(may),first=str(may.id.iloc[0]),last=str(may.id.iloc[-1]),sampling_seconds=3600,mean=float(may.hourly_pow.mean()),std=float(may.hourly_pow.std(ddof=1)),missing_total=int(may.isna().sum().sum()),
          bill_factor_max_error=float(abs(may.may_bill-may.hourly_pow*180).max()),carbon_factor_max_error=float(abs(may.may_carbon-may.hourly_pow*.424).max()),agg_unique=int(may.agg_pow.nunique()),agg_sum_error=float(abs(may.agg_pow-may.hourly_pow.sum()).max())),
      original_numeric_statistics=data,original_statistic_scope='Full population33,696,013 rows; moments from previous full-scan training_profile.json; std ddof0',
      time_checks=cadence,unix_duplicate_rows=sum(c['unix_duplicates']for c in cadence),local_duplicate_rows=sum(c['local_duplicates']for c in cadence),
      local_non5s_intervals=sum(c['local_non_5s']for c in cadence),operation_constant_one=True,
      target=dict(complete_hours=len(y),first=str(y.index[0]),last=str(y.index[-1]),total_provisional_kwh=float(y.sum()),mean=float(y.mean()),std=float(y.std(ddof=1)),source_sha256=sha(source)),
      source_time_offsets=audit['offset_local_naive_minus_unix_ms'],source_offset_transition='2025-03-09 03:00:00',
      module_statistics=[dict(module=k,**{a:float(b)if a!='hours'else int(b)for a,b in v.items()})for k,v in module_stats.iterrows()],
      raw_prefix=read(OUT/'raw_views/results.json'),analysis_type='Descriptive EDA, not forecasting evaluation',
      statistical_sources=['output/analysis/training_profile.json','output/analysis/time_anomaly_audit.json','output/analysis/baseline_audit/complete_hour_module_sum.csv','output/analysis/hourly_by_equipment.csv'],
      resampled_datasets=resampled,sources_modified=False)
    pd.DataFrame(data).to_csv(OUT/'evidence/pooled_sensor_statistics.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    module_stats.to_csv(OUT/'evidence/complete_hour_module_statistics.csv',encoding='utf-8-sig',float_format='%.17g')
    write(OUT/'report_facts.json',out);return out

def figure_manifest():
    # Report integration lists explicit authored assets, never scans unrelated folders.
    return read(OUT/'report_figure_manifest.json')

class Report:
    def __init__(self):
        pdfmetrics.registerFont(TTFont('Malgun','C:/Windows/Fonts/malgun.ttf'))
        pdfmetrics.registerFont(TTFont('MalgunBold','C:/Windows/Fonts/malgunbd.ttf'))
        pdfmetrics.registerFontFamily('Malgun',normal='Malgun',bold='MalgunBold',italic='Malgun',boldItalic='MalgunBold')
        self.styles={
          'title':ParagraphStyle('title',fontName='MalgunBold',fontSize=27,leading=38,textColor=colors.HexColor(BLUE),spaceAfter=16,wordWrap='CJK'),
          'h1':ParagraphStyle('h1',fontName='MalgunBold',fontSize=18,leading=26,textColor=colors.HexColor(BLUE),spaceAfter=13,wordWrap='CJK'),
          'h2':ParagraphStyle('h2',fontName='MalgunBold',fontSize=12,leading=19,textColor=colors.HexColor(TEAL),spaceBefore=8,spaceAfter=6,wordWrap='CJK'),
          'body':ParagraphStyle('body',fontName='Malgun',fontSize=9.5,leading=15,textColor=colors.HexColor(INK),spaceAfter=9,wordWrap='CJK'),
          'caption':ParagraphStyle('caption',fontName='Malgun',fontSize=8,leading=12,textColor=colors.HexColor(GRAY),spaceAfter=8,wordWrap='CJK'),
          'cell':ParagraphStyle('cell',fontName='Malgun',fontSize=8,leading=11,wordWrap='CJK',textColor=colors.HexColor(INK)),
          'thead':ParagraphStyle('thead',fontName='MalgunBold',fontSize=8,leading=11,wordWrap='CJK',textColor=colors.white)}
        self.story=[];self.sections=[];self.current=None;self.figure_ids=[]
    def paragraph(self,text,style='body'):
        self.story.append(Paragraph(text,self.styles[style]));browser_text=text.replace('<link ','<a ').replace('</link>','</a>');self.current['html'].append(f'<p class="{style}">{browser_text}</p>')
    def page(self,title,slug):
        if self.current is not None:self.story.append(PageBreak())
        self.current=dict(title=title,slug=slug,html=[]);self.sections.append(self.current)
        self.story.append(Paragraph(title,self.styles['h1']))
    def table(self,headers,rows,widths=None):
        cols=len(headers);widths=widths or [CONTENT_WIDTH/cols]*cols
        arr=[[Paragraph(html.escape(str(c)),self.styles['thead'])for c in headers]]+[[Paragraph(html.escape(str(c)),self.styles['cell'])for c in row]for row in rows]
        t=Table(arr,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([
          ('BACKGROUND',(0,0),(-1,0),colors.HexColor(BLUE)),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
          ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f1f5f9'),colors.white]),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#cbd5e1'))]))
        self.story.extend([t,Spacer(1,9)])
        self.current['html'].append('<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(str(c))+'</th>'for c in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>'for c in row)+'</tr>'for row in rows)+'</tbody></table></div>')
    def figure(self,path,caption,max_height=330):
        p=OUT/path;assert p.exists(),p
        from PIL import Image as PILImage
        with PILImage.open(p)as im:iw,ih=im.size
        scale=min(CONTENT_WIDTH/iw,max_height/ih);self.story.append(Image(str(p),width=iw*scale,height=ih*scale,hAlign='CENTER'))
        self.story.append(Spacer(1,4));self.paragraph(caption,'caption');self.figure_ids.append(path)
        self.current['html'].append(f'<figure><a href="{html.escape(path)}"><img src="{html.escape(path)}" alt="{html.escape(caption)}"></a><figcaption>{caption}</figcaption></figure>')
    def footer(self,canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#cbd5e1'));canvas.line(42,HEIGHT-34,WIDTH-42,HEIGHT-34)
        canvas.setFont('Malgun',7);canvas.setFillColor(colors.HexColor(GRAY));canvas.drawString(42,HEIGHT-27,'AWS 프로젝트 | 제공 데이터 기본 분석');canvas.drawRightString(WIDTH-42,HEIGHT-27,'2026-10-13')
        canvas.line(42,33,WIDTH-42,33);canvas.drawString(42,22,'기록된 localtime 기준 | 전력량은 가이드 정의에 따른 잠정값');canvas.drawRightString(WIDTH-42,22,str(doc.page));canvas.restoreState()
    def save(self):
        dest=OUT/'pdf';dest.mkdir(exist_ok=True)
        pdf=dest/'basic_data_report.pdf';doc=SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=48,bottomMargin=44,title='제공 데이터 기본 분석 보고서',author='AWS 프로젝트 데이터 분석')
        doc.build(self.story,onFirstPage=self.footer,onLaterPages=self.footer)
        nav=''.join(f'<a href="#{s["slug"]}">{html.escape(s["title"])}</a>'for s in self.sections)
        sections=''.join(f'<section id="{s["slug"]}"><h2>{html.escape(s["title"])}</h2>'+''.join(s['html'])+'</section>'for s in self.sections)
        sections+='<section id="downloads"><h2>개별 자료 다운로드</h2><p><a href="figures_and_tables.zip">그래프 PNG/SVG 20쌍과 분석 CSV 24개 묶음</a></p><p>'+ ' · '.join(f'<a href="data/{name}_dataset.csv">{label} 집계 CSV</a>'for name,label in [('hourly','시간'),('daily','일'),('weekly','주간'),('monthly','월')])+'</p><p>'+ ' · '.join(f'<a href="decomposition/{name}_components.csv">{label} 성분 CSV</a>'for name,label in [('hourly_stl24','STL 24시간'),('hourly_stl168','STL 168시간'),('daily_stl7','STL 7일'),('hourly_mstl24_168','MSTL 24/168시간')])+'</p></section>'
        doc='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>제공 데이터 기본 분석 보고서</title><style>html{{scroll-behavior:smooth}}body{{margin:0;background:#eef3f7;font:16px/1.75 "Malgun Gothic",sans-serif;color:{INK}}header,main{{max-width:1160px;margin:auto;padding:28px}}header{{background:{BLUE};color:white;border-radius:0 0 18px 18px}}h1{{font-size:32px;line-height:1.35}}h2{{font-size:25px;color:{BLUE};line-height:1.4}}nav{{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}}nav a{{background:#e4edf5;color:{BLUE};padding:4px 10px;border-radius:6px;font-size:13px;text-decoration:none}}section{{background:white;border-radius:12px;padding:24px;margin:24px 0;scroll-margin-top:12px}}.table{{overflow-x:auto;max-width:100%}}p,figcaption,td,th{{overflow-wrap:anywhere}}table{{border-collapse:collapse;width:100%;font-size:14px;line-height:1.5}}th,td{{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid #dbe3ec}}th{{background:{BLUE};color:white}}figure{{margin:16px 0}}img{{max-width:100%;height:auto}}figcaption,.caption{{font-size:14px;color:{GRAY}}.h2{{font-size:19px;color:{TEAL};font-weight:bold}}a{{color:#1d4ed8}}header a{{color:white}}.download{{display:inline-block;margin:5px 18px 5px 0}}@media(max-width:600px){{header,main{{padding:14px}}section{{padding:16px}}h1{{font-size:26px}}h2{{font-size:21px}}}}@media print{{header,main{{max-width:none;padding:0}}nav{{display:none}}section{{break-before:page;background:white;padding:0}}}}</style></head><body><header><h1>제공 데이터 기본 분석 보고서</h1><p>피처·기간·샘플링·데이터 품질·시간 해상도별 시각화·시계열 분해</p><a class="download" href="pdf/basic_data_report.pdf">PDF 보고서</a><a class="download" href="evidence/feature_dictionary.csv">전체 피처 사전 CSV</a><a class="download" href="verification.json">검증 기록</a></header><main><nav>{nav}</nav>{sections}</main></body></html>'''
        doc=doc.replace('{INK}',INK).replace('{BLUE}',BLUE).replace('{TEAL}',TEAL).replace('{GRAY}',GRAY).replace('{{','{').replace('}}','}').replace('{nav}',nav).replace('{sections}',sections)
        (OUT/'data_report.html').write_text(doc,encoding='utf8')
        write(OUT/'report_manifest.json',dict(pdf=str(pdf),html=str(OUT/'data_report.html'),section_count=len(self.sections),figure_paths=self.figure_ids,report_scope='Descriptive data overview, multi-resolution EDA and decomposition',report_generated=datetime.datetime.now(datetime.timezone.utc).isoformat()))

def populate(r,f,m,decomp):
    numeric={v['name']:v for v in f['original_numeric_statistics']};features=pd.read_csv(OUT/'evidence/feature_dictionary.csv').fillna('')
    if 'dataset'in features:features=features[features.dataset.astype(str).str.lower().isin(['full','training','rtu_data_full.csv'])]
    # Feature dictionary may carry richer independent audit columns; use canonical names.
    namecol=next(c for c in ['name','feature','column']if c in features)
    meaningcol=next(c for c in ['meaning_ko','meaning','description','description_ko']if c in features)
    unitcol=next(c for c in ['unit','unit_guide','unit_status']if c in features)
    dtypecol=next(c for c in ['dtype_observed','dtype','observed_dtype']if c in features)
    r.page('01  数据 개요와 핵심 확인 사항'.replace('数据','데이터'),'overview')
    r.paragraph('제공 데이터 기본 분석 보고서','title')
    r.paragraph('학습 원본과 5월 평가 자료를 확인하고, 원본의 피처·시간축·관측 간격 및 집계 자료의 추세·계절성·잔차를 분석했습니다. 이 문서는 예측 성능 보고서와 별도의 탐색적 데이터 분석 자료입니다.')
    r.table(['항목','Full 원본','May 정답'],[
      ['행 수',num(f['full']['rows'],0),'505'],['열 수','19','5'],['기간','2024-12-01 00:00 ~ 2025-04-30 00:00','2025-05-10 00:00 ~ 2025-05-31 00:00'],['관측 간격','모듈별 5초','1시간'],['계측 범위','13개 모듈 식별자','전체 시간별 정답 집계'],['센서 정보','전압·전류·전력·역률·누적량','센서와 모듈별 값 없음']], [78,CONTENT_WIDTH*.43,CONTENT_WIDTH-78-CONTENT_WIDTH*.43])
    r.figure('raw_views/dataset_timeline.png','그림 1. 두 제공 파일의 실제 시간 범위. Full의 마지막 시점과 완료된 시간 버킷의 끝을 구분합니다.',190)
    r.paragraph('<b>핵심:</b> 결측값은 없고 기록된 localtime은 모듈별 5초 간격으로 연속입니다. 다만 Unix timestamp와의 시간대 불일치, 계측기 포함 관계, 일부 전력량 단위는 확인이 필요합니다. 기간을 집계할수록 변동이 감소하며, 월 데이터는 완전한 4개 관측치뿐입니다.')

    r.page('02  원본 데이터의 19개 피처','features')
    r.paragraph('19개 열은 식별자 1개, 시간 필드 2개, 운전 상태 1개, 계측값 15개로 구성됩니다. 아래 dtype은 원본 앞부분의 CSV 파싱 추론이며 고정 스키마가 제공된 것은 아닙니다. 의미와 단위는 가이드 정의를 별도로 표시했습니다.')
    rows=[[v[namecol],v[meaningcol],v[unitcol],v[dtypecol]]for _,v in features.iterrows()]
    assert len(rows)==19
    r.table(['원본 피처명','의미','단위·주의','관측 dtype'],rows,[130,145,151,CONTENT_WIDTH-426])
    r.paragraph('localtime은 20241201000000처럼 YYYYMMDDhhmmss를 숫자로 인코딩한 필드입니다. 문자열로 바꾼 후 format=%Y%m%d%H%M%S로 해석했습니다. 정수 값을 날짜 함수에 바로 넣으면 1970년의 나노초 시각으로 오해할 수 있습니다. operation은 전체에서 1로 고정되어 운전 전환을 설명하는 피처로는 정보가 없습니다.','caption')

    r.page('03  계측 피처의 기초 통계','statistics')
    r.paragraph('원본 전체 33,696,013행의 이전 전수 스캔 합계·제곱합·최솟값·최댓값을 사용해 통계를 재계산했습니다. 표준편차는 모집단 기준(ddof=0)입니다. 표본 센서 그림 및 상관계수의 표본 범위와 구분했습니다.')
    rows=[[v['name'],num(v['mean'],2),num(v['std'],2),num(v['min'],2),num(v['max'],2)]for v in numeric.values()]
    r.table(['피처','평균','표준편차','최솟값','최댓값'],rows,[135,94,94,94,CONTENT_WIDTH-417])
    r.paragraph('표의 단위는 피처 사전을 따릅니다. accumActiveEnergy는 누적 카운터로, 값의 크기와 장기 증가가 피처 분포를 지배합니다. 누적량의 원수치와 구간 소비량을 혼동하지 않아야 합니다. 전압·전류와 activePower 사이의 물리적 단위 일치 및 누적량의 Wh/kWh 표기는 별도 확인 대상입니다.')
    r.paragraph('원본 모든 피처의 결측 수: 0. operation=1: 전체 33,696,013행. 결측이 없다는 사실만으로 센서 교정·단위·시간대·원래 삽입된 이상치가 모두 정상이라고 판정할 수는 없습니다.')

    r.page('04  계측 모듈의 범위와 관측량','modules')
    r.paragraph('단일 설비 1개만의 데이터가 아닙니다. 13개 모듈 이름에는 분전반·전등·기계·호이스트 등이 함께 포함돼 있습니다. 다만 계측기의 상하위 포함 관계가 없어 13개의 독립적인 최종 설비 또는 공장 총량으로 확정할 수 없습니다.')
    rows=[[v['module'],'2,592,001',num(v['energy_kwh'],0),num(v['share_percent'],3)+'%']for v in f['module_statistics']]
    r.table(['모듈 식별자','원본 행 수','완료된 시간 전력량 합계(kWh)','13모듈 합계 내부 비중'],rows,[155,87,164,CONTENT_WIDTH-406])
    r.figure(m['modules'],'그림 2. 모듈별 완료된 시간의 관측 전력량. 비중은 데이터 내 13개 모듈 합계의 구성비이며 실제 공장 에너지 점유율로 확정하지 않습니다.',210)
    r.paragraph('모듈별 시작·끝은 동일하고 각각 2,592,001행입니다. (150일×24시간×720개/시간)+종료 시점 1개에 해당합니다. 완료된 모듈별 시간은 3,600×13=46,800행이며, 13모듈을 합치면 3,600개 시간 시각이 됩니다. 원본 종료 시점 1개는 완료된 1시간 소비량으로 집계하지 않았습니다.','caption')

    r.page('05  샘플링 간격과 시간축 품질','sampling')
    r.figure('raw_views/raw_5second_example.png','그림 3. 원본 첫 PM-3 하루 17,280개의 5초 관측과 처음 120초 확대. 한 모듈의 첫날 예시이며 전체 모집단의 분포를 대표하는 통계 표본은 아닙니다.',355)
    r.table(['검사','결과','해석'],[
      ['모듈별 localtime 비5초 간격',num(f['local_non5s_intervals'],0),'기록된 현지 시각은 연속'],['모듈별 localtime 중복',num(f['local_duplicate_rows'],0),'모듈+localtime 키로 구분'],['모듈별 Unix timestamp 중복',num(f['unix_duplicate_rows'],0),'13개 모듈×720행; localtime 중복과 다름'],['Unix timestamp 역순 전환','13회','모듈별1회; localtime 역순은0회'],['시간대 오프셋 전환','2025-03-09 03:00','localtime - Unix: -8시간 → -7시간']],[155,126,CONTENT_WIDTH-281])
    r.paragraph('가이드는 한국시간이라고 설명하지만 Unix 시각과 localtime의 오프셋은 이 설명과 일치하지 않습니다. 첫 localtime은 2024-12-01 00:00이나 Unix 값을 변환하면 UTC 08:00, 한국시간 17:00입니다. 본 분석은 연속성이 검증된 기록 localtime을 기준으로 집계했으며 실제 시간대가 확인됐다는 뜻이 아닙니다. 서로 다른 모듈의 동일 시각은 정상 동시 관측이며 전체 행의 시각만으로 중복 제거하면 자료가 손상됩니다.','caption')

    r.page('06  리샘플링 기준과 시간 단위 시각화','hourly')
    r.paragraph('가이드 activePower(kW)는 해당 5초의 평균값입니다. 기록별 전력량=activePower×5/3600이며, 완료된 시간에서는 모듈별 720개 관측을 합칩니다. 최종 시계열은 13모듈을 합한 잠정 시간 전력량입니다. 원본을 보간하거나 이상치 후보를 삭제하지 않았습니다.')
    r.table(['단위','기준','전체 버킷','완료','부분'],[['H','1시간','3,600','3,600','0'],['D','00시~다음 00시','150','150','0'],['W-SUN','월요일~다음 월요일','23','21','2'],['MS','월초~다음 월초','5','4','1']],[55,206,83,83,CONTENT_WIDTH-427])
    r.figure(m['hourly'],'그림 4. 시간별 전력량 3,600개와 완료된 시간의 평균 부하. 1시간 버킷에서는 kWh와 kWh/h의 숫자가 같지만 물리량은 구분됩니다.',340)
    r.paragraph('전체 시간 전력량 평균 '+num(f['target']['mean'],3)+' kWh, 표본 표준편차 '+num(f['target']['std'],3)+' kWh. 각 버킷에 관측 시간과 기대 시간, coverage 및 완료 여부를 저장해 실제 관측 합계와 완전한 기간 총량을 구분했습니다.','caption')

    r.page('07  일 단위 시각화','daily')
    r.figure(m['daily'],'그림 5. 완료된 150일의 일 전력량 합계와 기간 평균 부하(일 전력량/24시간). 두 축은 각각 kWh/일과 kWh/h입니다.',385)
    r.figure(m['hourly_zoom'],'그림 6. 시간 시계열의 72시간 확대. 일정한 평균 수준 주위의 시간별 변동과 집계에 의한 평활화를 구분해 봅니다.',220)
    r.paragraph('집계는 관측값의 변동을 평균화합니다. 일별 곡선이 매끈해지는 것만으로 일별 예측이 더 쉬워졌거나 새로운 설명 피처가 생겼다고 결론 내리지는 않습니다.','caption')

    r.page('08  주간·월 단위 시각화','weekly_monthly')
    r.figure(m['weekly'],'그림 7. 주간 23개 버킷 중 21개는 168시간 모두 관측됐습니다. 처음 주는 12월 1일의 24시간, 마지막 주는 4월 28~29일의 48시간만 있습니다.',250)
    r.figure(m['monthly'],'그림 8. 월 합계와 월 평균 부하. 12~3월은 완료된 4개월이며, 4월은 29일(696/720시간)만 집계됐습니다. 월 합계 변화에는 28일·30일·31일의 달 길이 차이가 포함됩니다.',250)
    r.paragraph('부분 기간의 합계를 완전한 주·월과 직접 비교하면 소비 감소처럼 보일 수 있습니다. 기간 평균 부하=관측 전력량/관측시간으로 길이를 보정하되, 미관측 시간의 실제값을 추정한 완전 기간 총량으로 해석하지 않습니다. 주 라벨 일요일은 이용 가능 시점인 다음 월요일 00시와 다릅니다.','caption')

    r.page('09  시간대·요일·해상도별 패턴','calendar_patterns')
    r.figure(m['calendar'],'그림 9. 시간대×요일 평균의 전체 평균 대비 편차와 셀별 관측 수. 작은 상대 편차를 확대해서 표시한 결과이며 큰 운전 주기가 확인됐다는 뜻은 아닙니다.',290)
    r.figure(m['variability'],'그림 10. 해상도별 기간 평균 부하의 변동계수. 서로 다른 기간 길이의 에너지 합계 표준편차를 섞어서 비교하지 않았습니다.',235)
    r.paragraph('달력별 평균만으로 계절성을 확정하지 않고 아래의 원 시계열 ACF, 스펙트럼 및 순열 기반 진단을 함께 확인했습니다. 월 값 4개의 작은 변동은 충분한 계절 반복을 관측했다는 근거가 아닙니다.','caption')

    r.page('10  센서 분포와 동시점 관계','sensor_distribution')
    r.figure(m['sensors'],'그림 11. 모듈×완료된 시간별 층화 센서 표본 187,200행의 분포·동시점 상관 분석. 전수 통계 표와 다른 표본 결과이며 인과관계 또는 미래 예측력을 의미하지 않습니다.',465)
    r.paragraph('각 모듈의 완료된 시간마다 720개 5초 슬롯 중 4개를 균등·비복원으로 추출한 기존 표본을 사용했습니다. 13모듈×3,600시간×4=187,200행이며 고정 난수 시드를 사용하고 이상치 후보를 과표집하지 않았습니다. 동시점 상관이 커도 미래의 센서가 예측 시점에 제공되지 않으면 미래 목표 예측 피처로 그대로 사용할 수 없습니다.')

    r.page('11  시계열 분해 방법과 해석','decomposition_method')
    r.paragraph('가법적 분해: 관측값 = 추세 + 계절 성분 + 잔차. 추세는 완만한 수준 변화, 계절 성분은 지정한 주기로 반복하도록 추출한 성분, 잔차는 나머지입니다. 이 데이터에서는 robust STL과 두 주기를 동시에 분석하는 MSTL을 적용했습니다.')
    r.table(['입력','분해','지정 주기','관측 수'],[['시간 전력량','robust STL','24시간','3,600'],['시간 전력량','robust STL','168시간','3,600'],['일 평균 부하','robust STL','7일','150'],['시간 전력량','robust MSTL','24·168시간','3,600'],['주간 평균 부하','연간 계절 분해 제외','52주 필요','완료 21주'],['월 평균 부하','연간 계절 분해 제외','12개월 필요','완료 4개월']],[133,130,154,CONTENT_WIDTH-417])
    r.paragraph('STL은 주기를 지정하면 계절 성분을 추출하므로 그래프에 계절선이 나타난 사실 자체는 실제 반복 주기의 증거가 아닙니다. 계절성 강도는 잔차와 계절+잔차 분산의 비율로 계산한 기술적 지표이며 예측 R² 또는 미래 정확도가 아닙니다.')
    r.paragraph('이번 분해는 전체 과거 기간을 사용한 탐색적 설명입니다. 양쪽 관측을 이용하는 추세 필터에는 미래 값이 포함되므로 이 성분을 과거 시점의 예측 피처로 바로 사용할 수 없습니다. 실제 모델 적용에서는 각 학습 구간 안에서 다시 계산해야 합니다.')
    r.paragraph('참고: <link href="https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html" color="#1d4ed8">statsmodels STL 공식 문서</link>, <link href="https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.MSTL.html" color="#1d4ed8">MSTL 공식 문서</link>. 전체 설정·진단 결과는 decomposition/results.json과 개별 성분 CSV에 저장했습니다.')

    for i,key,title in [(12,'stl24','시간 시계열: 24시간 주기 STL'),(13,'stl168','시간 시계열: 168시간 주기 STL'),(14,'daily_stl7','일 시계열: 7일 주기 STL'),(15,'mstl','시간 시계열: 24·168시간 MSTL')]:
        r.page(f'{i:02d}  '+title,key)
        r.figure(m[key],'그림 '+str(i)+'. 관측값, 추세, 계절 성분과 잔차. 성분의 축 단위는 입력 단위와 동일하며 관측값과 성분별 세로축 범위는 다릅니다.',470)
        case=m.get('decomposition_case_text',{}).get(key,'고정 주기·창 설정으로 분해했습니다. 성분별 변동 크기와 실제 주기성 진단을 함께 해석해야 합니다.')
        r.paragraph(case)
        r.paragraph('성분 CSV에는 원 시계열, 추세, 계절 성분, 잔차를 저장했습니다. 합성 재구성 오차와 원 자료의 시간 정렬은 별도 검증했습니다. 전체기간 사후 분해 그림이며 미래 예측 결과를 표시한 그림이 아닙니다.','caption')

    r.page('16  주기성 진단과 분석 결론','diagnostics')
    r.figure(m['diagnostics'],'그림 16. 원 시계열의 자기상관·스펙트럼과 지정 주기 반복성 진단. 분해에 앞서 실제 데이터에서 주기를 뒷받침하는지 확인했습니다.',335)
    r.paragraph(m.get('diagnostic_text','주기성을 판단할 때 ACF와 스펙트럼, 반복성 검정을 함께 봅니다. 지정 주기 분해는 주기성의 자동 입증 절차가 아닙니다.'))
    r.paragraph('<b>프로젝트에 주는 의미:</b> 현재 자료는 13개 전력 계측 모듈을 높은 빈도로 관측한 이력입니다. 모듈별 센서 관계와 이상치 후보 분석에는 활용할 수 있으나, operation이 일정하고 평가 파일에 미래 센서가 없어 미래 변동을 설명하는 정보는 제한적입니다. 정격용량·생산량·운영 계획·계측기 계층과 시간대 확인이 추가 분석의 우선 자료입니다.')
    r.paragraph('주간·월간 시계열에는 반복 계절을 확인할 기간이 부족합니다. 특히 완료된 월 4개로 월별 계절 예측이나 계절분해의 안정성을 검증했다는 주장은 하지 않았습니다. 이번 보고서는 사실 확인과 탐색 목적이며 고장 정답이나 데이터 생성 방식을 추정해 확정하지 않았습니다.')

    r.page('17  5월 정답 데이터와 자료 사용 범위','may')
    r.table(['피처','내용','관측 사실'],[['id','평가 시각','1시간 간격, 505개'],['hourly_pow','집계 전력량 정답','평균 '+num(f['may']['mean'],3)+'; 표본 std '+num(f['may']['std'],3)],['may_bill','전기요금','hourly_pow × 180'],['may_carbon','탄소량','hourly_pow × 0.424'],['agg_pow','평가 범위 합계','505개 hourly_pow 합계가 모든 행에 반복']],[97,151,CONTENT_WIDTH-248])
    r.paragraph('May의 5개 열은 독립적인 센서 피처 5개가 아닙니다. 시각과 목표 전력량, 그 전력량에서 파생된 요금·탄소·합계이며, 전압·전류·운전 상태·모듈별 시계열은 제공되지 않습니다. 정답 파생 열을 미래 예측의 입력으로 사용하면 목표 정보가 유입됩니다.')
    r.paragraph('May 관측 구간은 5월 전체가 아닙니다. 5월 10~30일의 완료된 21일 및 5월 31일 00시 버킷 1개입니다. 완전한 주는 2개이고 완전한 월은 없습니다. agg_pow를 5월 전체 총량으로 설명하지 않습니다.')
    r.paragraph('공개 원본에서 내려받은 파일과 제공 파일의 바이트 크기·SHA-256은 이전 검증에서 동일했습니다. 이번 작업은 원본을 수정하거나 이동하거나 복사하지 않았으며, 새 그래프·분해 성분·통계·보고서만 별도 폴더에 만들었습니다.')
    r.paragraph('<b>새 산출물:</b> PDF 및 HTML 보고서, 피처 사전·전수 통계 CSV, H/D/W/MS 집계 CSV, 분해 성분 CSV, 모든 그래프의 PNG/SVG, 재현 소스와 검증·파일 색인.')

    r.page('18  데이터 근거와 재현 파일','references')
    r.paragraph('원본 근거는 프로젝트가이드.pdf, traing-dataset/rtu_data_full.csv 및 may-dataset/rtu_ground_truth_may.csv입니다. 가이드의 단위 설명과 실제 시간·형식·값에서 확인한 사실을 구분했습니다.')
    r.table(['근거','확인 내용'],[['evidence/evidence_summary.json','원본 스키마·행수·기간·모듈·샘플링·품질 독립 확인'],['evidence/feature_dictionary.csv','19개 피처의 의미·dtype·단위·결측·범위'],['evidence/pooled_sensor_statistics.csv','원본 전수 통계 재계산'],['visuals/results.json','시간 해상도별 집계·관측률·차트·표본 범위'],['decomposition/results.json','STL/MSTL 설정·성분·강도·주기성 진단'],['report_facts.json','보고서에 사용한 수치와 통계 근거'],['verification.json','수치·시간 정렬·재구성·문서·브라우저 검증'],['artifact_inventory.json','새 산출물의 지속 식별자·크기·SHA-256']],[240,CONTENT_WIDTH-240])
    r.paragraph('재현 시 기본 자료 보존을 위해 원본·기존 산출물은 읽기 전용으로 다뤘습니다. full 전수 통계는 기존 검증된 33,696,013행 전체 스캔 결과를 재사용했고, 이번에 원본 첫 17,280행을 직접 읽어 날짜 파싱과 5초 간격 예시를 확인했습니다. 모듈×시간 층화 센서 표본은 전수 통계와 구분했고 실제 출처·추출 방식·시드를 시각화 결과 JSON에 기록했습니다.')
    r.paragraph('시계열 분해 공식 방법 문서: https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html 및 https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.MSTL.html. 검정의 가정과 재구성 허용 오차는 분해 결과 JSON에 기록했습니다.','caption')

def main():
    (OUT/'evidence').mkdir(exist_ok=True);facts=basic_facts();manifest=figure_manifest();decomp=read(OUT/'decomposition/results.json')
    r=Report();populate(r,facts,manifest,decomp);r.save()
    print(json.dumps(dict(pdf=str(OUT/'pdf/basic_data_report.pdf'),html=str(OUT/'data_report.html'),sections=len(r.sections),figures=len(r.figure_ids)),ensure_ascii=True))
if __name__=='__main__':main()
