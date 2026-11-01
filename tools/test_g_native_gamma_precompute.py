import ast
import os
from pathlib import Path
import struct
import unittest
import torch
from g_native_gamma_precompute import build,GAMMA_HELPER,HELPER_PINS
from g_native_patch_ladder import flags

ROOT=Path(os.environ.get('OKC_REPO_ROOT',Path(__file__).resolve().parents[1]))


def scalar_reference(bits):
    # BF16 raw is an exactly representable binary32; emulate RN float32 add,
    # then integer ties-to-even BF16 rounding. No native helper involved.
    raw=struct.unpack('<f',struct.pack('<I',bits<<16))[0]
    result=struct.unpack('<I',struct.pack('<f',1.+raw))[0]
    return ((result+0x7fff+((result>>16)&1))>>16)&0xffff


def coefficient_helper(source):
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='FORK_GAMMA_PRECOMPUTE'
           or isinstance(n,ast.FunctionDef) and n.name=='_fork_gamma_native_weight']
    ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual-native-gamma-prepare>','exec'),ns)
    return ns


class Gamma(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=(ROOT/'muse_decode_layer/cand/g1b4_stage_release.py').read_text(encoding='utf-8')
        cls.on,cls.manifest=build(cls.original,True)

    def test_all_finite_bf16_scalars_actual_prepare_helper(self):
        ns=coefficient_helper(self.on);all_bits=torch.arange(65536,dtype=torch.int32)
        finite=(all_bits&0x7f80)!=0x7f80
        safe=all_bits.clone();safe[~finite]=0
        raw=safe.to(torch.int16).view(torch.bfloat16)
        restored=[]
        for chunk in raw.split(4096):
            saved=chunk.view(torch.int16).clone()
            got=ns['_fork_gamma_native_weight']({'post_expert_norm':chunk})
            self.assertEqual(got.dtype,torch.bfloat16)
            self.assertNotEqual(got.untyped_storage().data_ptr(),chunk.untyped_storage().data_ptr())
            self.assertTrue(torch.equal(chunk.view(torch.int16),saved))
            restored.append(got.view(torch.int16).to(torch.int32)&0xffff)
        got=torch.cat(restored)[finite]
        want=torch.tensor([scalar_reference(b) for b in all_bits[finite].tolist()],dtype=torch.int32)
        self.assertEqual(len(want),65280)
        self.assertTrue(torch.equal(got,want))
        for bad in (torch.full((4096,),float('nan'),dtype=torch.bfloat16),torch.ones(4096),torch.ones(4095,dtype=torch.bfloat16)):
            with self.assertRaisesRegex(RuntimeError,'FORK_GAMMA_FORMAT'):ns['_fork_gamma_native_weight']({'post_expert_norm':bad})
        off,_=build(self.original)
        raw=torch.ones(4096,dtype=torch.bfloat16)
        self.assertIs(coefficient_helper(off)['_fork_gamma_native_weight']({'post_expert_norm':raw}),raw)

    def test_complete_consumer_reversal_and_other_AST_unchanged(self):
        original=ast.parse(self.original);patched=ast.parse(self.on)
        for key in HELPER_PINS:
            a=next(n for n in original.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])==key)
            b=next(n for n in patched.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])==key)
            value=ast.literal_eval(b.value).replace(GAMMA_HELPER,'')
            records=self.manifest['consumer_literals'][key]['expressions']
            self.assertEqual(len(records),16)
            for record in records:value=value.replace(record['new'],record['old'],1)
            self.assertEqual(value,ast.literal_eval(a.value))
            b.value=a.value
        for name in ('_fork_native_ma_prepare','_fork_native_moe'):
            a=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
            i=next(i for i,n in enumerate(patched.body) if isinstance(n,ast.FunctionDef) and n.name==name)
            patched.body[i]=a
        patched.body=[n for n in patched.body if not(isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='FORK_GAMMA_PRECOMPUTE')
                      and not(isinstance(n,ast.FunctionDef) and n.name=='_fork_gamma_native_weight')]
        self.assertEqual(ast.dump(patched),ast.dump(original))
        # Actual native-bank slot7 now calls the derived coefficient helper.
        prep=next(n for n in ast.parse(self.on).body if isinstance(n,ast.FunctionDef) and n.name=='_fork_native_ma_prepare')
        bank=next(n for n in ast.walk(prep) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=="st['ma_w']")
        self.assertEqual(ast.unparse(bank.value.elts[7]),'_fork_gamma_native_weight(weights)')

    def test_actual_five_key_selector_format(self):
        function=next(n for n in ast.parse(self.on).body if isinstance(n,ast.FunctionDef) and n.name=='_fork_native_moe')
        definitions=flags(self.original)
        for enabled in (False,True):
            recorded=[]
            class Ext:
                def __init__(self,flags):self.flags=flags
            ns=dict(definitions,FORK_GAMMA_PRECOMPUTE=enabled,MOE_HAS_PDL=False,
                    _RouterExt=lambda x:x,_ma_ext=lambda flags:recorded.append(flags) or Ext(flags),
                    _main_defines=lambda sfx:definitions['MOE_DEFINES'])
            exec(compile(ast.Module(body=[function],type_ignores=[]),'<actual-native-selector>','exec'),ns)
            state={'ma_sfx':('-DOKC_PRMTSC=1',),'ma_sc6':True,'ma_ext':{},'ma_gamma_effective':enabled}
            for tokens,key in ((1,'tiny'),(8,'tiny2'),(16,'mid'),(32,'b4'),(128,'main'),(256,'main')):
                ext=ns['_fork_native_moe'](tokens,state)
                self.assertIs(ext,state['ma_ext'][key])
                self.assertEqual('-DOKC_FORK_GAMMA_EFFECTIVE=1' in ext.flags,enabled)
            self.assertEqual(len(recorded),5)
            state['ma_gamma_effective']=not enabled
            with self.assertRaisesRegex(AssertionError,'state mismatch'):ns['_fork_native_moe'](1,state)


if __name__=='__main__':unittest.main()
