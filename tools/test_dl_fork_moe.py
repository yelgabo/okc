"""Offline source gates. Optional --Torch tests run in the persistent CPU venv."""
import ast
import hashlib
from pathlib import Path
import unittest
import dl_fork_moe as patcher

ROOT=Path(__file__).resolve().parent.parent
SOURCE=Path('C:/Users/gretc/Documents/GitHub/okc/.build/today/kel/kernels/muse_decode_layer.py')


def strings(src):
    out={}
    for n in ast.parse(src).body:
        if isinstance(n,ast.Assign):
            target=ast.unparse(n.targets[0])
            if isinstance(n.value,ast.Constant) and isinstance(n.value.value,str): out[target]=n.value.value
    return out


class NativeRepair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=SOURCE.read_text(encoding='utf-8')
        cls.patched=patcher.patch_moe(cls.source)
        cls.sources=strings(cls.patched)

    def test_scope_and_failclosed_pin(self):
        with self.assertRaises(AssertionError): patcher.patch_moe(self.source+'\n')
        old=strings(self.source)
        changed={key for key in self.sources if key in old and self.sources[key]!=old[key]}
        self.assertEqual(changed,{'MOE_CUDA_SRC',"_MA_ALT_SRCS['moe_helper0']"})
        tree=ast.parse(self.patched)
        df=next(n for n in tree.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=="_MA_ALT_SRCS['moe_helper0df']")
        self.assertEqual(ast.unparse(df.value),"_MA_ALT_SRCS['moe_helper0']")
        self.assertIn('MOE_HAS_PDL = False',self.patched)
        self.assertIn('MFAB_LAYER = False',self.patched)

    def test_all_selected_precision_and_launch_sites(self):
        for key in ('MOE_CUDA_SRC',"_MA_ALT_SRCS['moe_helper0']"):
            code=self.sources[key]
            import re
            pairs=re.findall(r'(?:const )?float gt = (.*?), up = (.*?);',code)
            self.assertEqual(len(pairs),5)
            self.assertTrue(all(g.startswith('fork_bf16(') and u.startswith('fork_bf16(') for g,u in pairs))
            self.assertEqual(code.count('cudaLaunchCooperativeKernel('),1)
            self.assertNotIn('cudaLaunchKernelEx(',code)
            self.assertNotIn('<<<(int)grid',code)
            self.assertIn('resident>=1 && grid<=resident*prop.multiProcessorCount',code)
            self.assertIn('prop.cooperativeLaunch',code)
            self.assertIn('FORK_MOE_UNPROVEN_FALLBACK',code)
            self.assertIn('atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;',code)
            self.assertIn('fence.proxy.async.shared::cta;',code)
        main=self.sources['MOE_CUDA_SRC']
        self.assertIn('#define OKC_MFIN 0',main)
        self.assertIn('#define OKC_YBF 0   // fork:',main)
        self.assertEqual(main.count('fence.proxy.async.global;'),3)
        self.assertIn('k < BW_E ? 1 : k < BX_F ? NT : k < BX_E ? 1 : NT',main)
        self.assertEqual(main.count('mbar_arrive1(S.bars + BX_E + prev);'),2)
        self.assertNotIn('if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);',main)

    def test_sc6_pack_and_fallback_sources_unchanged(self):
        a=ast.parse(self.source); b=ast.parse(self.patched)
        for name in ('_sc6_ok','_sc6_wgs','_wgs_copy','_ma_scales_prmt_exact','_ma_prepare'):
            aa=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name==name)
            other='_fork_native_ma_prepare' if name=='_ma_prepare' else name
            bb=next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name==other)
            bb.name=name
            if name=='_ma_prepare':
                class RestoreState(ast.NodeTransformer):
                    def visit_Subscript(self,node):
                        if ast.unparse(node)=='st[\'ma_sc6\']': return ast.Name(id='_MA_SC6W',ctx=ast.Load())
                        return self.generic_visit(node)
                bb=RestoreState().visit(bb)
            self.assertEqual(ast.dump(aa),ast.dump(bb))
        # Exhaustively check every 6-bit scale and every lane's mixed upper-bit packing.
        for base in range(64):
            v=[(base+13*i)%64 for i in range(8)]
            pieces=[v[6]&3,(v[6]>>2)&3,(v[6]>>4)&3,v[7]&3,(v[7]>>2)&3,(v[7]>>4)&3]
            packed=[(v[n]<<2)|pieces[n] for n in range(6)]
            restored=[p>>2 for p in packed]+[(packed[0]&3)|((packed[1]&3)<<2)|((packed[2]&3)<<4),
                (packed[3]&3)|((packed[4]&3)<<2)|((packed[5]&3)<<4)]
            self.assertEqual([x+0x40 for x in restored],[x+0x40 for x in v])

    def test_state_scales_and_unused_legacy_reachability(self):
        tree=ast.parse(self.patched)
        builder=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_ma_ext')
        self.assertNotIn('_MA_SC6W',ast.unparse(builder))
        for key in ('w13q','w13s','w2q','w2s'):
            self.assertFalse(any(isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant)
                and n.slice.value==key for n in ast.walk(tree)))
        prep=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare'
            and 'FORK_MOE_SCALE' in ast.unparse(n))
        self.assertNotIn('_ma_ext(',ast.unparse(prep))
        pick=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_pick_expert_backend')
        scope={};exec(compile(ast.fix_missing_locations(ast.Module(body=[pick],type_ignores=[])),'backend_gate','exec'),scope)
        self.assertEqual(scope['_pick_expert_backend']('til',False,True),'til')
        # Every branch which selects this unknown label immediately raises.
        self.assertEqual(self.patched.count('if not (use_marlin or use_ma):'),4)
        self.assertEqual(self.patched.count('raise RuntimeError("no expert backend:'),4)
        dispatch=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_fork_native_moe')
        from types import SimpleNamespace
        calls=[]
        def ext(defines): calls.append(defines);return SimpleNamespace(defines=defines)
        scope=dict(_ma_ext=ext,_RouterExt=lambda e:e,MOE_HAS_PDL=False,
            MOE_TINY_DEFINES=('tiny',),MOE_TINY_MAX_M=4,MOE_TINY2_DEFINES=('tiny2',),
            MOE_TINY2_MAX_M=8,MOE_MID_DEFINES=('mid',),MOE_MID_MAX_M=16,
            MOE_DEFINES=('-DOKC_WGS=1',),_main_defines=lambda s:('-DOKC_WGS=1',),_MA_SC6W=False)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[dispatch],type_ignores=[])),'state_dispatch','exec'),scope)
        a=dict(ma_sfx=('prmt',),ma_sc6=True,ma_ext={})
        b=dict(ma_sfx=('prmt',),ma_sc6=False,ma_ext={})
        aa=scope['_fork_native_moe'](128,a)
        scope['_MA_SC6W']=True  # opposite process-global must not affect stateB
        bb=scope['_fork_native_moe'](128,b)
        self.assertIn('-DOKC_SC6W=1',aa.defines)
        self.assertNotIn('-DOKC_SC6W=1',bb.defines)
        self.assertIs(scope['_fork_native_moe'](128,a),aa)
        self.assertEqual(len(calls),2)


if __name__=='__main__': unittest.main()
