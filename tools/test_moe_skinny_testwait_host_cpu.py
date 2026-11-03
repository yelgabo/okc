"""Execute actual optional stamp/summary branches with real CPU Torch2D scratch.

Run /root/.cache/okc-host-validate/bin/python tools/test_moe_skinny_testwait_host_cpu.py
These are synthetic host fixtures, never GPU measurements.
"""
import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from moe_skinny_testwait_check import build

ROOT=Path(__file__).resolve().parent.parent


def assigned(node,name):
    return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)


class HostStampCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree=ast.parse(build((ROOT/'muse_moe_w4a16/submission.py').read_text()))
        cls.decoder=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_tw_decode_wait_counters')
        cls.snapshot=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_tw_snapshot_wait_counters')
        cls.wrapper=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare'][-1]
        cls.variants=next(ast.literal_eval(n.value) for n in tree.body if assigned(n,'SKINNY_VARIANTS'))
        a=next(i for i,n in enumerate(cls.wrapper.body) if assigned(n,'stamp_complete'))
        z=next(i for i,n in enumerate(cls.wrapper.body) if assigned(n,'summary'))
        cls.optional=cls.wrapper.body[a:z]
        cls.summary=[n for n in cls.wrapper.body[z:] if not isinstance(n,ast.Return)]

    def execute(self,nodes,scope):
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
                     'actual_testwait_stamp_host','exec'),scope)

    def scope(self,kind='full',elapsed=100.,spin=0,win=True):
        # Exact _scratch inter shape from production, not a flattened substitute.
        inter=torch.zeros((4096,4096),dtype=torch.bfloat16)
        ctr=torch.zeros(64,dtype=torch.int32); ctr[8]=spin
        calls=[]
        def run(data,state,variant):
            if kind=='execution_error': raise RuntimeError('simulated CUDA execution error')
            calls.append((data,variant))
            flat=inter.view(torch.int64).reshape(-1)
            d=flat[:132*16].reshape(132,16)
            d.zero_()
            if kind!='zero':
                d[:,0]=1000; d[:,7]=101000
                d[:,12]=2000; d[:,13]=182000
                mode=self.variants[variant][2]
                d[:,8]=1800 if mode==2 else 900
                d[:,9]=3600 if mode==2 else 1800
        facade=SimpleNamespace(int64=torch.int64,isfinite=torch.isfinite,
            cuda=SimpleNamespace(synchronize=lambda:None,
                get_device_properties=lambda dev:SimpleNamespace(multi_processor_count=132)))
        scope=dict(torch=facade,time=SimpleNamespace(monotonic=lambda:elapsed),began=0.,dev='cpu',
            inputs=lambda m,r,s:(m,r,s),_run=run,SKINNY_VARIANTS=self.variants,
            states={'base':{'scratch':{'inter':inter,'ctr':ctr}}},
            flush=SimpleNamespace(zero_=lambda:None),stamps=[],
            core=['TWCORE_RESULT complete16cases resources_equivalentTrue',f'KEEP={win}'],calls=calls)
        self.execute([self.snapshot,self.decoder],scope)
        if kind=='cpu_shape_error':
            # A CPU-only malformed prefix is optional; no CUDA work here.
            scope['_tw_decode_wait_counters']=lambda raw,g:raw[:-1].reshape(g,16)
        if kind=='snapshot_error':
            def failed_snapshot(x,g): raise RuntimeError('simulated CUDA snapshot transfer error')
            scope['_tw_snapshot_wait_counters']=failed_snapshot
        return scope

    def test_complete_real_tensor_decode_and_summary(self):
        for win in (True,False):
            scope=self.scope(win=win)
            self.execute(self.optional,scope)
            self.assertTrue(scope['stamp_complete'])
            self.assertEqual(scope['snapshot'].device.type,'cpu')
            self.assertEqual(scope['snapshot'].dtype,torch.int64)
            self.assertEqual(scope['snapshot'].shape,(132*16,))
            self.assertEqual(len(scope['calls']),16)
            self.assertEqual(len(scope['stamps']),8)
            self.assertTrue(all('(2, 1.0, 2.0)' in line and '(130, 0.5, 1.0)' in line for line in scope['stamps']))
            with contextlib.redirect_stdout(io.StringIO()) as output: self.execute(self.summary,scope)
            self.assertIn(f'KEEP={win}',output.getvalue())
            self.assertIn('TWSTAMP_COMPLETE True',output.getvalue())

    def test_zero_and_shape_decode_failures_preserve_core(self):
        for kind in ('zero','cpu_shape_error'):
            for win in (True,False):
                scope=self.scope(kind=kind,win=win)
                self.execute(self.optional,scope)
                self.assertFalse(scope['stamp_complete'])
                self.assertEqual(len(scope['calls']),1)
                self.assertIn('AssertionError' if kind=='zero' else 'RuntimeError',scope['stamp_error'])
                with contextlib.redirect_stdout(io.StringIO()) as output: self.execute(self.summary,scope)
                self.assertIn('TWCORE_RESULT',output.getvalue())
                self.assertIn(f'KEEP={win}',output.getvalue())
                self.assertIn('noattributionifFalse',output.getvalue())
        # Reproduce the actual old rank2 slicing failure with real Torch data.
        inter=self.scope()['states']['base']['scratch']['inter']
        with self.assertRaises(RuntimeError): inter.view(torch.int64)[:132*16].reshape(132,16)

    def test_stamp_guard_and_execution_fail_closed(self):
        scope=self.scope(elapsed=270.)
        self.execute(self.optional,scope)
        self.assertFalse(scope['stamp_complete'])
        self.assertEqual(scope['stamp_error'],'guard265')
        self.assertEqual(scope['calls'],[])
        with contextlib.redirect_stdout(io.StringIO()) as output: self.execute(self.summary,scope)
        self.assertIn('error=guard265',output.getvalue())
        self.assertIn('TWCORE_RESULT',output.getvalue())
        guard=next(n for n in self.wrapper.body if isinstance(n,ast.FunctionDef) and n.name=='guard')
        scope=self.scope(elapsed=281.)
        self.execute([guard],scope)
        with self.assertRaisesRegex(RuntimeError,'280s'): scope['guard']()
        with self.assertRaisesRegex(RuntimeError,'simulated CUDA'):
            self.execute(self.optional,self.scope(kind='execution_error'))
        with self.assertRaisesRegex(RuntimeError,'snapshot transfer'):
            self.execute(self.optional,self.scope(kind='snapshot_error'))
        with self.assertRaisesRegex(AssertionError,'spin timeout'):
            self.execute(self.optional,self.scope(spin=1))


if __name__=='__main__':
    print(f'Host-only validation Torch={torch.__version__} CUDA_available={torch.cuda.is_available()}')
    unittest.main()
