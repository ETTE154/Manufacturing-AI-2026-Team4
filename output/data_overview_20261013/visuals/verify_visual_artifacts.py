"""Read-only source/hash and generated-figure checks, plus recorded visual QA."""
from pathlib import Path
import sys,json,hashlib,xml.etree.ElementTree as ET
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'output/analysis/.plot_deps'))
from PIL import Image

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    r=json.loads((OUT/'results.json').read_text(encoding='utf8'))
    captions=json.loads((OUT/'chart_caption.json').read_text(encoding='utf8'))
    sources=[]
    for source in r['sources']:
        current=sha(source['path']);assert current==source['sha256']
        sources.append({'path':source['path'],'unchanged_sha256':current})
    figures=[]
    for key,chart in captions['charts'].items():
        png=Path(chart['paths']['png']);svg=Path(chart['paths']['svg'])
        with Image.open(png) as im:
            size=list(im.size);assert size[0]>=2200 and size[1]>=1000;im.verify()
        content=svg.read_text(encoding='utf8');assert '\ufffd' not in content
        root=ET.fromstring(content);assert root.tag.endswith('svg')
        # Any rasterized scatter points are embedded data, not external files.
        hrefs=[value for node in root.iter() for attr,value in node.attrib.items() if attr.endswith('href')]
        assert all(value.startswith('#') or value.startswith('data:') for value in hrefs)
        figures.append({'key':key,'png_dimensions':size,'png_sha256':sha(png),'svg_sha256':sha(svg),
                        'PNG_decodes':True,'SVG_parses':True,'SVG_external_image_dependencies':False,'replacement_characters':0})
    assert len(figures)==10
    checked=['hourly_energy_rate','hourly_72h','daily_energy_rate','weekly_energy_rate','monthly_energy_rate',
             'weekday_hour_pattern','module_energy_distribution','module_energy_comparison','resolution_variability','sensor_distribution_correlation']
    out={'status':'pass','figure_count':10,'formats':['png','svg'],'sources_preserved':sources,'figures':figures,
          'visual_inspection':{'inspected_PNG_keys':checked,'Korean_font_normal':True,'titles_labels_legends_not_clipped':True,
          'partial_week_labels_corrected_and_rechecked':True,'partial_month_hollow_marker_and_hatching':True,
          'zero_axes_and_expanded_axes_explicit':True,'sensor_log_hist_and_observed_correlation_only':True},
          'scope':'Descriptive EDA only; no model training and no May-label read'}
    (OUT/'visual_qa.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
    print('PASS10PNG+10SVG; source SHA unchanged; visual inspection recorded')
if __name__=='__main__':main()
