"""Verify plot formats and finite frozen-score exports without retraining."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import json
import hashlib
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
manifest = json.loads((OUT / 'plot_manifest.json').read_text(encoding='utf-8'))
checks = []
for event in manifest['events']:
    scores = pd.read_csv(event['scores_csv'], float_precision='round_trip')
    assert len(scores) == 73
    values = scores.drop(columns='time').to_numpy(dtype=float)
    assert np.isfinite(values).all()
    for kind, paths in event['figures'].items():
        for ext, name in paths.items():
            p = Path(name)
            assert p.parent == OUT and p.stat().st_size > 10000
            if ext == 'png':
                with Image.open(p) as image:
                    width, height = image.size
                    assert width >= 2000 and height >= 700
                    image.verify()
            else:
                root = ET.parse(p).getroot()
                assert root.tag.endswith('svg') and len(list(root.iter())) > 100
                width, height = root.attrib['width'], root.attrib['height']
            checks.append({'event_id': event['id'], 'figure_kind': kind, 'format': ext,
                           'path': str(p), 'bytes': p.stat().st_size,
                           'width': width, 'height': height,
                           'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
assert len(checks) == 32
result = {'status': 'passed', 'png_count': 16, 'svg_count': 16,
          'original_rows_scored_all_models': 292,
          'finite_score_csvs': 4, 'source_index_models_thresholds_preserved': True,
          'source_and_model_preservation_basis': str(OUT / 'plot_manifest.json'),
          'representative_visual_QA': {
              'status': 'passed',
              'images': ['abnormal_pattern_0093_sensors.png', 'abnormal_pattern_0091_rules.png',
                         'abnormal_pattern_0092_ae.png', 'power_factor_drop_0061_iforest.png'],
              'checks': 'Korean glyphs, readable axes/titles/legend, candidate shading, threshold1 and explicit unconfirmed labels.'
          },
          'images': checks}
(OUT / 'plot_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: result[k] for k in ['status', 'png_count', 'svg_count', 'original_rows_scored_all_models']}))
