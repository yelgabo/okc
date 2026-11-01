"""Build private stage-retirement component A/B; never dispatches or submits."""
import argparse,ast,hashlib,importlib.util,sys
from pathlib import Path

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def build(root,upstream,body):
    sys.path.insert(0,str(root/'tools'))
    make=load(root/'tools/moe_make_sc6_compare.py','_stage_sc6_make').make
    stage=load(root/'tools/dl_g1_stage_retire.py','_stage_patch')
    source=make(upstream.read_text(encoding='utf8'),(root/'muse_moe_w4a16/baseline.py').read_text(encoding='utf8'),body)
    tree=ast.parse(source);n=next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='CUDA_SRC')
    old=ast.literal_eval(n.value);new=stage.patch_device(old,switch=True)
    assert new.count('OKC_STAGE_RETIRE')==2 and '-DOKC_STAGE_NULL' not in new
    lines=source.splitlines(keepends=True);lines[n.lineno-1:n.end_lineno]=['CUDA_SRC = '+repr(new)+'\n'];out=''.join(lines)
    ast.parse(out);assert len(out.encode())<1_000_000
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--root',type=Path,required=True);p.add_argument('--upstream',type=Path);a=p.parse_args()
    upstream=a.upstream or a.root/'.build/today/kel/kernels/muse_moe_w4a16.py'
    s=build(a.root,upstream,Path(__file__).with_name('dl_g1_stage_ab_body.py').read_text(encoding='utf8'));a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(s.encode());print('STAGE_AB',len(s.encode()),hashlib.sha256(s.encode()).hexdigest())
