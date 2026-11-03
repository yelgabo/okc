"""Generate the private, fail-closed C2 SK10 XOR paired diagnostic.

Input is the exact C2 source plus reviewed proj_sparse_xor helper suffix. This
module never fetches source or changes production. No GPU work at generation.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
from proj_make_bench_sk_xor_body import BODY

AUGMENTED_SHA256 = 'd23aa17be4cf9e7c5869803b3b4603d8e72395953687adc0287fb1c75738d586'
C2_SHA256 = '5610b8a64e25f70c639ac0523fe7c14754a6c54d349eb005210123b4e19fb3b4'


from proj_sk_xor_codec import decode_report, encode_report


def score_helper(source):
    tree=ast.parse(source)
    names=('LAYERS','M_VALUES','SEEDS','ROUNDS','TARGETS')
    chosen=[n for n in tree.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)) or (isinstance(n,ast.FunctionDef) and n.name=='score_model')]
    assert len(chosen)==6
    mapping={n:'_SXS_'+n for n in names};mapping.update(score_model='_sx_score_model',math='_sxmath')
    class Rename(ast.NodeTransformer):
        def visit_Name(self,n):
            if n.id in mapping:n.id=mapping[n.id]
            return n
        def visit_FunctionDef(self,n):
            if n.name in mapping:n.name=mapping[n.name]
            return self.generic_visit(n)
    return ast.unparse(Rename().visit(ast.Module(body=chosen,type_ignores=[])))


def make(c2_bytes, augmented_bytes, score_source):
    assert hashlib.sha256(augmented_bytes).hexdigest()==AUGMENTED_SHA256, 'reviewed XOR helper source drift'
    assert hashlib.sha256(c2_bytes).hexdigest()==C2_SHA256, 'not exact public C2 pin'
    c2=c2_bytes.decode('utf8').replace('\r\n','\n');aug=augmented_bytes.decode('utf8').replace('\r\n','\n')
    assert aug.startswith(c2), 'helper source must preserve exact C2 byte prefix'
    tree=ast.parse(aug)
    suffix=ast.parse(aug[len(c2):])
    for node in suffix.body:
        if isinstance(node,ast.Assign):
            assert all(isinstance(t,ast.Name) and t.id in ('_XOR_CUDA','_XOR_CPP','_XOR_CONFIG') for t in node.targets), 'unexpected suffix binding'
        else:
            assert isinstance(node,ast.FunctionDef) and node.name in ('xor_expected_layout','xor_build','xor_pack','_xor_pack_bits','xor_runner'), 'unexpected suffix side effect'
    for name in ('xor_build','xor_pack','xor_runner'):
        assert sum(isinstance(n,ast.FunctionDef) and n.name==name for n in tree.body)==1,name
    # Helper may add builders/runners, never replace judged dispatch or preparation.
    for name in ('prepare','kernel_fn'):
        assert sum(isinstance(n,ast.FunctionDef) and n.name==name for n in tree.body)==1,name
    pins={'c2_sha256':C2_SHA256,'augmented_sha256':hashlib.sha256(augmented_bytes).hexdigest(),
          'c2_lf_sha256':hashlib.sha256(c2.encode()).hexdigest(),
          'body_sha256':hashlib.sha256(BODY.encode()).hexdigest(),
          'scope':'private C2 normal judged control; SK10-only XOR comparison; no public permission'}
    codec=ast.get_source_segment(Path(__file__).with_name('proj_sk_xor_codec.py').read_text(encoding='utf8'),next(n for n in ast.parse(Path(__file__).with_name('proj_sk_xor_codec.py').read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name=='encode_report')).replace('def encode_report(', 'def _sx_encode(',1)
    model=score_helper(score_source)
    pins['score_model_sha256']=hashlib.sha256(score_source.encode()).hexdigest()
    pins['codec_sha256']=hashlib.sha256(codec.encode()).hexdigest()
    emitted=aug+'\n'+BODY.replace('@@PIN@@',repr(pins)).replace('@@CODEC@@',codec).replace('@@MODEL@@',model)
    ast.parse(emitted)
    assert len(emitted.encode())<1000000, 'official one-file source limit'
    return emitted,pins


def main():
    p=argparse.ArgumentParser();p.add_argument('c2',type=Path);p.add_argument('augmented',type=Path);p.add_argument('output',type=Path);p.add_argument('--score-reader',type=Path,default=Path(__file__).with_name('proj_sk_xor_read.py'));a=p.parse_args()
    src,pins=make(a.c2.read_bytes(),a.augmented.read_bytes(),a.score_reader.read_text(encoding='utf8'))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(src.encode('utf8'))
    pins['physical_sha256']=hashlib.sha256(a.output.read_bytes()).hexdigest();pins['bytes']=a.output.stat().st_size
    a.output.with_suffix('.manifest.json').write_text(json.dumps(pins,indent=2)+'\n',encoding='utf8')
    print(json.dumps(pins,indent=2))

if __name__=='__main__':main()

