"""Executed CPU interface/reachability tests; not a CUDA correctness certificate."""
import ast
import copy
from pathlib import Path
import sys
import unittest
import dl_fork_owned_moe as module

BASE=Path(sys.argv.pop(1)).read_text(encoding='utf-8')
OWNED=Path(sys.argv.pop(1)).read_text(encoding='utf-8')

class Gate(unittest.TestCase):
    def test_wired_reachability_and_source(self):
        result=module.patch_owned_moe(BASE,OWNED)
        tree=ast.parse(result)
        defs=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
        fns={n.name:n for n in defs}
        self.assertEqual(sum(n.name=='_moe' for n in defs),2)
        for n in defs:
            if n.name in ('_ma_ext','_moe_build','_moe_mid','_moe_tiny','_moe_tiny2'):
                self.assertIsInstance(n.body[0],ast.Raise)
        ma=fns['_ma_prepare']
        self.assertIn('_fork_install_owned_moe',ast.unparse(ma.body[0]))
        ns={'_fork_install_owned_moe':lambda w,s:s.update(owned_moe='adapter') or True}
        exec(compile(ast.Module(body=[ma,fns['_fork_owned_moe_for'],fns['_moe']],type_ignores=[]),'actualdispatch','exec'),ns)
        st={};ns['_ma_prepare']({},st)
        self.assertEqual(st['ma_w'],());self.assertEqual(st['ma_scratch'],())
        for m in (1,7,8,16,17,32,128,256,384,385,512):self.assertEqual(ns['_moe'](m,st),'adapter')
        with self.assertRaises(RuntimeError):ns['_moe'](8,{})
        prep=next(n for n in defs if n.name=='prepare' and '_ma_prepare(weights, st)' in ast.unparse(n))
        text=ast.unparse(prep)
        self.assertLess(text.index('_ma_prepare(weights, st)'),text.index('_tune(st'))
        thread=next(n for n in ast.walk(prep) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ths' for t in n.targets))
        self.assertNotIn('_moe',ast.unparse(thread))
        self.assertNotIn('_repack_q',text);self.assertNotIn('_repack_s',text)
        self.assertNotIn("st['g_postexp']",text)
        self.assertLess(len(result.encode()),1_000_000)
        with self.assertRaises(AssertionError):module.module_source(OWNED+'\n')

    def test_actual_adapter_and_gamma_cpu(self):
        import torch
        calls=[]
        class Ext:
            def moe_fused_run(self,*args):
                calls.append(args)
                return torch.full_like(args[0],3)
        ext=Ext();builds=[]
        ns={'torch':torch,'load_inline':lambda **kw:builds.append(kw) or ext}
        exec(module.module_source(OWNED),ns)
        packed={k:torch.ones(1) for k in ('w13','w2','w13_gs','w2_gs','w13f','w2f')}
        scratch={k:torch.ones(1) for k in ('inter','ybuf','ss','ctr','done_a','xa','xb','fix','fixcnt','xready','done_as')}
        ns['_owned_moe1_prepare']=lambda w:packed
        ns['_owned_moe1_scratch']=lambda dev:scratch
        gamma=torch.tensor([.001953125,.00390625,-.001953125,.0078125],dtype=torch.bfloat16).repeat(1024)
        weights={'w13':torch.zeros(1),'post_expert_norm':gamma}
        state={};self.assertTrue(ns['_fork_install_owned_moe'](weights,state))
        adapter=state['owned_moe']
        self.assertTrue(torch.equal(adapter.gamma,(1+gamma.float()).bfloat16().float()))
        self.assertFalse(torch.equal(adapter.gamma,1+gamma.float()))
        for m in (1,8,16,32,128,256,1,17):
            x=torch.zeros((m,4096),dtype=torch.bfloat16)
            global_ids=torch.tensor([128,129,159,160,0,127,255,130]).repeat(m,1)
            local=(global_ids-128)%256
            lids=torch.where(local<32,local,-1).to(torch.int32)
            tw=torch.full((m,8),.125);out=torch.empty_like(x)
            adapter.okc_set_pdl(1)
            adapter.okc_run((),(),x,lids,tw,out,0,32,132)
            a=calls[-1]
            self.assertIs(a[1],lids);self.assertEqual(int(a[3]),0)
            self.assertIs(a[8],adapter.gamma);self.assertEqual(a[22:24],(32,1e-8))
            self.assertEqual(a[24],0 if m<=16 else 1);self.assertIs(a[25],False)
            self.assertTrue(torch.equal(out,torch.full_like(out,3)))
        clone=copy.deepcopy(adapter)
        self.assertIs(clone.module,adapter.module)
        self.assertNotEqual(clone.scratch['ctr'].data_ptr(),adapter.scratch['ctr'].data_ptr())
        self.assertNotEqual(clone.zero.data_ptr(),adapter.zero.data_ptr())
        self.assertEqual(len(builds),1)
        cuda=builds[0]['cuda_sources']
        self.assertIn('case 0: sk_8_3_0::launch<1>',cuda)
        self.assertIn('case 1: launch_main<51>',cuda)
        for unused in ('namespace sk_8_4_0 {','namespace sk_16_4_0 {','namespace sk_8_4_3 {'):
            self.assertNotIn(unused,cuda)

if __name__=='__main__':unittest.main()
