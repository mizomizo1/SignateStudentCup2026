from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
EXPECTED=json.loads((ROOT/'frozen_manifest.json').read_text(encoding='utf-8'))
OUT=ROOT/'outputs'

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

for filename, expected_count in EXPECTED['counts'].items():
    path=OUT/filename
    if not path.exists(): raise FileNotFoundError(path)
    df=pd.read_csv(path,header=None,names=['企業ID','購入フラグ'])
    got_count=int(df['購入フラグ'].sum())
    if got_count!=expected_count: raise AssertionError((filename,'positive_count',got_count,expected_count))
    got_sha=sha256(path)
    expected_sha=EXPECTED['sha256'][filename]
    if got_sha!=expected_sha: raise AssertionError((filename,'sha256',got_sha,expected_sha))
print('FINAL_V2 VERIFIED: all five output counts and SHA256 values match frozen_manifest.json')
