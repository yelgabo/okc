import ast
import random
import struct
import types
import unittest

import d32_identity_rescale as candidate
from test_attention_layer_continuity import config, environment, invoke


def bits(value):return struct.unpack('<I',struct.pack('<f',value))[0]
def value(raw):return struct.unpack('<f',struct.pack('<I',raw))[0]


class IdentityRescale(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source,cls.control,cls.native,cls.manifest=candidate.build()

    def test_finite_identity_and_vote_model(self):
        raw=[n<<16 for n in range(65536) if n&0x7f80!=0x7f80]
        rng=random.Random(192211)
        raw += [n for n in (rng.getrandbits(32) for _ in range(10000)) if n&0x7f800000!=0x7f800000]
        for n in raw:self.assertEqual(bits(value(n)*1.),n)
        for active in ([True]*32,[False]*32,[i%3==0 for i in range(32)]):
            for alpha in ([1.]*32,[.5]*32,[0. if i%7==0 else 1. for i in range(32)]):
                need=any(a and f!=1. for a,f in zip(active,alpha))
                for i,(a,f) in enumerate(zip(active,alpha)):
                    x=value(raw[i])
                    self.assertEqual(bits(x*f if a else x),bits(x*f if a and need else x))

    def test_selected_entries_and_direct_override_guard(self):
        captures,owners=[],[]
        env,_,_=environment(self.source,captures,owners)
        tree=ast.parse(self.source)
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)
               and n.name in ('_d32_attention_scope','_d32_attention_module')]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'identity-scope','exec'),env)
        module=types.SimpleNamespace(run=lambda *args:captures.append(('identity',args)))
        env['_prepare_attention_modules7']=lambda state:state.update(
            d32_attention_eligible=True,d32_attention_module=module)
        for batch in (1,16,17,32,33):
            for length in (1,8):
                for window in (0,2048):
                    state,data,_=invoke(env,config(batch,length,window))
                    expected=17<=batch<=32 and length==1 and window==0
                    self.assertEqual(captures[-1][0],'identity' if expected else 'control')
                    if not expected:
                        with self.assertRaises(ValueError):
                            env['_launch7'](data,attention_state=state,_mod=module)

    def test_disabled_twin_and_preserved_denominator(self):
        _,control,disabled,_=candidate.build(True)
        self.assertEqual(control,self.control)
        self.assertEqual(disabled,self.native.replace('#define D32_IDENTITY_SKIP 1','#define D32_IDENTITY_SKIP 0'))
        self.assertIn('l_r[0] = l_r[0] * alpha + ls;',self.native)
        self.assertIn('if constexpr (D32_IDENTITY_SKIP && FLAT && WS && !ONE)',self.native)
        self.assertLess(self.native.index('__any_sync(0xffffffffu'),
                        self.native.index('    if (d32_active) {'))
        self.assertIn('D32 identity module metadata scope',self.native)
        self.assertIn('D32 identity module launch scope',self.native)
        self.assertIn('D32 identity module scale scope',self.native)
        self.assertEqual(self.native.count('mbar_arrive('),self.control.count('mbar_arrive('))
        self.assertEqual(self.native.count('fence.proxy.async'),self.control.count('fence.proxy.async'))


if __name__=='__main__':unittest.main()
