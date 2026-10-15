"""Vendor pinned official PatchTST sources, preserving originals and hashes."""
from pathlib import Path
import hashlib
import json
import urllib.request

HERE = Path(__file__).resolve().parent
COMMIT = '204c21efe0b39603ad6e2ca640ef5896646ab1a9'
REPOSITORY = 'https://github.com/yuqinie98/PatchTST'
FILES = ['PatchTST_supervised/models/PatchTST.py',
         'PatchTST_supervised/layers/PatchTST_backbone.py',
         'PatchTST_supervised/layers/PatchTST_layers.py',
         'PatchTST_supervised/layers/RevIN.py','LICENSE']
records = []
for filename in FILES:
    url = f'https://raw.githubusercontent.com/yuqinie98/PatchTST/{COMMIT}/{filename}'
    with urllib.request.urlopen(url,timeout=30) as response:
        original = response.read()
    path = HERE/'upstream'/filename
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(original)
    text = original.decode('utf-8')
    if filename.endswith('.py'):
        local_relative = Path(filename).relative_to('PatchTST_supervised')
        destination = HERE/'patchtst_vendor'/local_relative
        # Only namespace import statements change; architecture code is untouched.
        modified = text.replace('from layers.','from patchtst_vendor.layers.').replace('import layers.','import patchtst_vendor.layers.')
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(modified.encode('utf-8'))
    else:
        destination = HERE/'patchtst_vendor'/'LICENSE'
        destination.parent.mkdir(parents=True,exist_ok=True)
        modified = text
        destination.write_bytes(original)
    records.append({'upstream_path':filename,'source_url':url,
                    'original_sha256':hashlib.sha256(original).hexdigest(),
                    'vendored_path':str(destination),'vendored_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
                    'modification':'Only from/import layers namespace changed to patchtst_vendor.layers' if filename.endswith('.py') else 'Unmodified license'})
for directory in ['patchtst_vendor','patchtst_vendor/models','patchtst_vendor/layers']:
    (HERE/directory/'__init__.py').write_text('',encoding='utf-8')
provenance = {'repository':REPOSITORY,'commit':COMMIT,'files':records,
              'architecture_scope':'Official supervised PatchTST core with namespace-only import changes; compact univariate config, not original-paper benchmark reproduction',
              'algorithmic_modifications':False}
(HERE/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
print(json.dumps(provenance,indent=2))
