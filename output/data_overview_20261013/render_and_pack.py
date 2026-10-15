"""Render final PDF for visual QA and package only newly generated visuals/CSV."""
from pathlib import Path
import subprocess,json,zipfile,sys
sys.dont_write_bytecode=True
from pypdf import PdfReader
from PIL import Image,ImageOps,ImageDraw
OUT=Path(__file__).resolve().parent
def main():
    pdf=OUT/'pdf/basic_data_report.pdf';temp=OUT/'tmp/pdfs';temp.mkdir(parents=True,exist_ok=True)
    reader=PdfReader(pdf);assert len(reader.pages)==18
    text={}
    for i,page in enumerate(reader.pages):
        t=page.extract_text();assert t and '\ufffd'not in t;text[i+1]=t
    (temp/'extracted_pages.json').write_text(json.dumps(text,ensure_ascii=False,indent=2),encoding='utf8')
    exe=Path('C:/Users/jeon/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.EXE')
    subprocess.run([str(exe),'-r','110','-png',str(pdf),str(temp/'page')],check=True)
    pages=sorted(temp.glob('page-*.png'));assert len(pages)==18
    for j in range(0,18,6):
        sheet=Image.new('RGB',(1200,1710),'#e2e8f0');draw=ImageDraw.Draw(sheet)
        for n,path in enumerate(pages[j:j+6]):
            im=Image.open(path);im.thumbnail((570,790));x=15+(n%2)*600;y=30+(n//2)*565
            # Each thumbnail keeps all of the page, with sufficient review area.
            im.thumbnail((570,530));sheet.paste(im,(x,y));draw.text((x,y-18),f'Page{j+n+1}',fill='black')
        sheet.save(temp/f'contact-{j//6+1}.png')
    files=[]
    for folder in ['raw_views','visuals/figures','decomposition','evidence','data']:
        for p in (OUT/folder).rglob('*'):
            if p.is_file() and p.suffix.lower()in ['.png','.svg','.csv'] and 'tmp'not in p.parts and '.mplconfig'not in p.parts:
                files.append(p)
    archive=OUT/'figures_and_tables.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6)as z:
        for p in sorted(files):z.write(p,p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive)as z:assert z.testzip()is None;assert len(z.namelist())==len(files)
    result=dict(pdf_pages=18,rendered_pages=len(pages),archive_members=len(files),archive_png_count=sum(p.suffix=='.png'for p in files),archive_svg_count=sum(p.suffix=='.svg'for p in files),archive_csv_count=sum(p.suffix=='.csv'for p in files),pdf_text_unicode_replacement_chars=0)
    (OUT/'render_pack_qa.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
