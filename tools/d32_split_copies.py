"""Build a D32 sliding reduction with one private workspace copy per split."""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path

BASE = 'muse_decode_layer/cand/g1b4_stage_release.py'
BASE_SHA = 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'

HOST = '''
_d32_copies_parent_plan = _plan7
_D32_COPY_PLANS = set()

def _plan7(B,pairs,L,window,max_pages,P,device):
    plan=_d32_copies_parent_plan(B,pairs,L,window,max_pages,P,device)
    if 17<=B<=32 and pairs==1 and L==1 and window>0 and P==1 and not plan['flat'] and plan['cs']==0:
        plan=dict(plan,nc=plan['n_splits'])
        key=(B,L,window,plan['n_splits'],plan['nc'])
        if key not in _D32_COPY_PLANS:
            print('D32_SPLIT_COPIES',key,flush=True)
            _D32_COPY_PLANS.add(key)
    return plan
'''


def build():
    source = (Path(__file__).resolve().parents[1] / BASE).read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == BASE_SHA
    result = source+'\n'+HOST
    ast.parse(result)
    assert len(result.encode()) < 1_000_000
    return result, dict(base=BASE, base_sha256=BASE_SHA,
                        sha256=hashlib.sha256(result.encode()).hexdigest(), bytes=len(result.encode()),
                        native_source_unchanged=True, compile_flags_unchanged=True,
                        scope='17..32 batch, q_len1, sliding, page1, one KV pair, noncluster plan',
                        mechanism='unique split copies followed by existing fixed-order sum, no shared atomic destinations')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    args = p.parse_args()
    source, manifest = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(source.encode())
    args.archive.write_bytes(gzip.compress(source.encode(), mtime=0))
    args.archive.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))
