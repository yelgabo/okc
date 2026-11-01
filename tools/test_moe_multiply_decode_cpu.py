import ast
from pathlib import Path
import unittest

import torch
from moe_multiply_decode import M,MASK,derive_positions,prove_carry_freedom,raw_decode,patch,OLD,NEW,OLD_NS,NEW_NS

ROOT=Path(__file__).resolve().parents[1]


def load(source):
    tree=ast.parse(source);fs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    needed={'repack_expert','repack_expert_fat','w4_repack_values'}
    while True:
        old=set(needed)
        for name in list(needed):
            needed.update(n.func.id for n in ast.walk(fs[name]) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in fs)
        if old==needed:break
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in needed]
    table=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='W4_BITPOS')
    ns={'torch':torch,'W4_BITPOS':table}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual-owned-packers>','exec'),ns)
    return ns


def old_decode(w):return w&M,(w<<6)&M,((w>>6)|(w<<26))&M,((w>>4)&0x00400040)|((w<<4)&0x81808180)


def tensor_decode(w,new):
    if new:return w&M,((w<<10)|(w>>22))&M,((w<<13)|(w>>19))&M,((w&MASK)*16648)&M
    return old_decode(w)


def pack(codes,table):
    return sum(((code>>b)&1)<<table[i][b] for i,code in enumerate(codes) for b in range(4))


class DecodeTests(unittest.TestCase):
    def test_bijection_carry_and_all_code_pairs(self):
        prove_carry_freedom()
        source=(ROOT/'muse_moe_w4a16/submission.py').read_text(encoding='utf-8')
        old=load(source)['W4_BITPOS'];new=derive_positions()
        self.assertEqual(sorted(sum((list(x) for x in new),[])),list(range(32)))
        for reg in range(4):
            for pair in range(256):
                codes=[(3*i+pair)%16 for i in range(8)]
                codes[2*reg]=pair&15;codes[2*reg+1]=pair>>4
                self.assertEqual(old_decode(pack(codes,old)),raw_decode(pack(codes,new)))
        print('MULTIPLY_CPU pairfixtures1024 carrysubsets256 exact')

    def test_actual_full_shape_gate_up_down_and_legacy_bytes(self):
        source=(ROOT/'muse_moe_w4a16/submission.py').read_text(encoding='utf-8')
        states=[load(source),load(patch(source))]
        gen=torch.Generator().manual_seed(4108)
        checked=0
        for kind,n in ((1,8192),(2,4096)):
            packed=torch.randint(0,256,(n,2048),dtype=torch.uint8,generator=gen)
            scales=torch.randint(8,127,(n,256),dtype=torch.uint8,generator=gen)
            for name in ('repack_expert','repack_expert_fat','w4_repack_values'):
                if name=='w4_repack_values':
                    results=[ns[name](packed,kind).reshape(-1) for ns in states]
                else:
                    values=[ns[name](packed,scales,kind).reshape(-1,36864) for ns in states]
                    self.assertTrue(torch.equal(values[0][:,32768:],values[1][:,32768:]))
                    results=[v[:,:32768].contiguous().view(torch.int32).reshape(-1).long()&0xffffffff for v in values]
                for a,b in zip(tensor_decode(results[0],False),tensor_decode(results[1],True)):
                    self.assertTrue(torch.equal(a,b),(kind,name))
                checked+=results[0].numel()*8
                print('MULTIPLY_CPU fullshape',kind,n,4096,name,'codes',results[0].numel()*8,flush=True)
        self.assertEqual(checked,3*(8192+4096)*4096)

    def test_source_only_decoder_and_table(self):
        source=(ROOT/'muse_moe_w4a16/submission.py').read_text(encoding='utf-8')
        revised=patch(source)
        before,after=ast.parse(source),ast.parse(revised)
        for tree in (before,after):
            for node in tree.body:
                if isinstance(node,ast.Assign) and getattr(node.targets[0],'id','')=='W4_BITPOS':node.value=ast.Constant('DERIVED_TABLE')
                elif isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
                    node.value.value=node.value.value.replace(NEW,OLD).replace(NEW_NS,OLD_NS)
        self.assertEqual(ast.dump(before,include_attributes=False),ast.dump(after,include_attributes=False))
        with self.assertRaises(AssertionError):patch(source+'\n')
        with self.assertRaises(AssertionError):patch(revised)


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
