"""CPU-only structural/ownership checks; pass pinned private donor with --source."""
import ast
import hashlib
from pathlib import Path
import sys
import types
import unittest
from dl_fork_attention import patch_attention

SOURCE=None
if '--source' in sys.argv:
    i=sys.argv.index('--source');SOURCE=Path(sys.argv[i+1]);del sys.argv[i:i+2]


class Buffer:
    def __init__(self,n):self.n=n
    def numel(self):return self.n


class AttentionTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if SOURCE is None:raise unittest.SkipTest('requires --source pinned private donor')
        raw=SOURCE.read_bytes()
        assert hashlib.sha256(raw).hexdigest()=='1ec189dc6f306cec1694946eb62d64b54daf378c6fc1ada2f0f5e501e973ff2a'
        cls.original=raw.decode().replace('\r\n','\n');cls.patched=patch_attention(cls.original)
        cls.tree=ast.parse(cls.patched)

    def test_device_source_and_guards(self):
        def literal(src,name):
            return next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
        self.assertEqual(literal(self.original,'ATT7_CUDA_SRC'),literal(self.patched,'ATT7_CUDA_SRC'))
        self.assertEqual(self.patched.count('cfg.get("kv_scale", 1.0) != 1.0'),3)
        self.assertIn('attention_state=state, **pdl)',self.patched)
        for name in ('qe','qo','ke','ko','v','g'):
            self.assertIn(name+' = '+name+'.to(tl.bfloat16).to(tl.float32)',self.patched)
        self.assertIn('o_in = (a * torch.sigmoid(gate.view(T, 16, 64).float())).to(torch.bfloat16)',self.patched)
        self.assertIn('return _after_attention(data, state, o_in, None, T)',self.patched)
        self.assertNotIn('ties are measure-zero',self.patched)

        compile(self.patched,'patched_layer','exec')
        with self.assertRaises(AssertionError):patch_attention(self.patched)

    def test_private_state_growth_and_fallback(self):
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_current_stream_capturing=lambda:False),float32=object(),int32=object(),zeros=lambda n,**kw:Buffer(n))
        seen=[]
        def base(data,state):seen.append(state);return 'fa3'
        env={'torch':torch,'_guarded':lambda x:x,'_attention_fa3_base':base,'kernel_fn':lambda *a:(_ for _ in ()).throw(AssertionError('recursive global dispatch'))}
        names={'_workspace7','_cluster_bufs7','_kernel_fn_fa3'}
        nodes=[n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'isolated_helpers','exec'),env)
        def owner():return {'rows':{},'cluster':{},'retained':[]}
        a,b=owner(),owner()
        first=env['_workspace7'](10,'cuda',a);other=env['_workspace7'](10,'cuda',b)
        self.assertIsNot(first[0],other[0])
        self.assertIs(env['_workspace7'](10,'cuda',a),first)
        larger=env['_workspace7'](40000,'cuda',a)
        self.assertIsNot(larger,first);self.assertIs(a['retained'][0],first)
        c=env['_cluster_bufs7'](1,8,'cuda',a)
        self.assertIsNot(c[0],env['_cluster_bufs7'](1,8,'cuda',b)[0])
        torch.cuda.is_current_stream_capturing=lambda:True
        self.assertIs(env['_workspace7'](10,'cuda',a),larger)
        with self.assertRaises(AssertionError):env['_workspace7'](100000,'cuda',a)
        state={'att_ok':True,'fa3':object(),'attention_scratch':a}
        self.assertEqual(env['_kernel_fn_fa3']({},state),'fa3')
        self.assertTrue(state['att_ok']);self.assertFalse(seen[-1]['att_ok'])
        self.assertIs(seen[-1]['attention_scratch'],a)
        with self.assertRaises(RuntimeError):env['_kernel_fn_fa3']({}, {'fa3':None})


if __name__=='__main__':unittest.main()
