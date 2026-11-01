"""Transparent owned band32 CHECK artifact; no GPU dispatch or public candidate."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys


def build(source, moe_tools):
    sys.path.insert(0,str(moe_tools))
    from moe_skinny_band import patch
    from moe_make_multiply_check import limited
    from moe_owned_cooperative_pair import cooperative_source
    control=limited(cooperative_source(source))
    candidate=limited(patch(source,cooperative=True))
    original=ast.parse(source)
    patched=ast.parse(patch(source))
    def other(tree):
        return ast.dump(ast.Module([n for n in tree.body if not
            (isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_SKINNY_BLOCK')],[]),include_attributes=False)
    assert other(original)==other(patched),'Unrelated source changed'
    manifest=dict(original_lf_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  control_lf_sha256=hashlib.sha256(control.encode()).hexdigest(),
                  candidate_lf_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
                  patcher_sha256=hashlib.sha256((moe_tools/'moe_skinny_band.py').read_bytes()).hexdigest(),
                  cooperative_sha256=hashlib.sha256((moe_tools/'moe_owned_cooperative_pair.py').read_bytes()).hexdigest(),
                  body_sha256=hashlib.sha256(Path(__file__).with_name('moe_band_check_body.py').read_bytes()).hexdigest(),
                  seeds=[993701,993703,993709],treatment_m=[16,32],shared_duplicate_binaries=True)
    out='# okc: problem=muse_moe_w4a16\n# CHECK ONLY: owned band32, common cooperative launch and BF16 effective gamma.\nimport torch,time\n'
    out+='_BT_CONTROL='+repr(control)+'\n_BT_CANDIDATE='+repr(candidate)+'\n'
    out+='_BAND_SOURCE_MANIFEST='+repr(manifest)+'\n'
    out+=Path(__file__).with_name('moe_band_check_body.py').read_text(encoding='utf8')
    ast.parse(out);assert len(out.encode())<1000000
    return out,manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--moe-tools',type=Path,required=True);a=p.parse_args()
    out,manifest=build(a.source.read_text(encoding='utf8'),a.moe_tools.resolve())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(out,encoding='utf8',newline='\n')
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8',newline='\n')
    print(len(out.encode()),hashlib.sha256(out.encode()).hexdigest())
