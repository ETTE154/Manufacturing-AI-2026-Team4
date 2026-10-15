"""Fetch selected official iTransformer model modules; does not execute them."""
from pathlib import Path
import urllib.request, json, hashlib

DEST = Path(__file__).resolve().parent / 'itransformer' / 'vendor'
SHA = 'c2426e68ca13f74aaec08045c5c724d8ad328124'
FILES = ['model/iTransformer.py','layers/Embed.py','layers/SelfAttention_Family.py',
         'layers/Transformer_EncDec.py','utils/masking.py','LICENSE']

def main():
    provenance = []
    for name in FILES:
        url = f'https://raw.githubusercontent.com/thuml/iTransformer/{SHA}/{name}'
        data = urllib.request.urlopen(url, timeout=60).read()
        path = DEST / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        provenance.append({'upstream_url':url,'path':name,'upstream_sha256':hashlib.sha256(data).hexdigest()})
    for folder in [DEST, DEST/'model',DEST/'layers',DEST/'utils']:
        (folder/'__init__.py').write_text('',encoding='utf-8')
    (DEST/'provenance.json').write_text(json.dumps({'repository':'https://github.com/thuml/iTransformer','revision':SHA,'files':provenance},indent=2),encoding='utf-8')
    print('Downloaded selected official modules for source review.')

if __name__ == '__main__':
    main()
