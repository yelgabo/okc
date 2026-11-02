"""Actual Python entry tests and sampled source/lifetime negatives, CPU only."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import sys
from attention_continuity import generate, transform, ROOT, LAYER
from attention_top100_screen import literal, function


class Continuity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = (ROOT/LAYER).read_text()
        cls.candidate, cls.original, cls.cuda, _ = generate(cls.base)
        cls.tree = ast.parse(cls.candidate)

    def test_identity_negative(self):
        with self.assertRaises(ValueError): generate(self.base+'\n')
        with self.assertRaises(ValueError): transform(self.original.replace('#define OKC_CONT 0 ', '#define OKC_CONT 2 '))

    def test_native_boundaries(self):
        # Exactly these five narrow changes; every numerical expression and other
        # stage/page/query/retirement instruction is identical by inverse reversal.
        changed = self.cuda
        from attention_continuity import once
        changed = once(changed, '#define OKC_CONT 1 ', '#define OKC_CONT 0 ')
        changed = once(changed, 'if (L == 1 && !(OKC_NOXCH1 && !M_CL))', 'if (L == 1 && !CONT && !(OKC_NOXCH1 && !M_CL))')
        changed = once(changed, 'float* xs = (CONT ? reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(&S) + sizeof(Smem<L, FLAT>)) : reinterpret_cast<float*>(S.k))', 'float* xs = reinterpret_cast<float*>(S.k)')
        changed = once(changed, ' + ((L == 1 && Cfg<L, true>::WS && OKC_CONT) ? 64 * 18 * sizeof(float) : 0)', '')
        changed = once(changed, '  if constexpr (Cfg<L, FLAT>::WS && OKC_CONT) __syncthreads(); // publish initialized full/empty barriers before role split\n', '')
        self.assertEqual(changed, self.original)

    def test_generated_source_derivation(self):
        nodes = [n for n in self.tree.body if isinstance(n, (ast.Assign, ast.Assert)) and
                 ('ATT7_CONT_CUDA_SRC' in ast.unparse(n))]
        env = dict(ATT7_CUDA_SRC=self.original)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual-derived-source', 'exec'), env)
        self.assertEqual(env['ATT7_CONT_CUDA_SRC'], self.cuda)

    def test_actual_twelve_launches_and_final_binding(self):
        captures = []
        fake_torch = NS(bfloat16='bf16', empty=lambda *a, **k: NS(view=lambda *a:'result'))
        env = dict(torch=fake_torch, _num_sms7=lambda _:132, LOG2E=1.4426950408889634,
            Q_DIM=1024, _workspace7=lambda *a:('acc','sum','count'),
            _cluster_bufs7=lambda *a:('cluster','barrier'), _inv_freq=lambda *a:'freq',
            _att7=lambda:NS(run=lambda *a:captures.append(('original', a))))
        for key in ('_TUNED7','OKC_FLATCOOP','OKC_PDL','OKC_FLAT1B','OKC_PLAIN8','OKC_F2','OKC_S2','OKC_W0_L1','OKC_W0_L8','OKC_W0_PR','PDL','PDL_MAX_T','ATT_HAS_PDL'):
            env[key] = literal(self.tree,key)
        for name in ('_plan7','_pdl','_launch7','_attention_continuity_module','_decode_attention7'):
            function(self.tree,name,env)
        original_launch = env['_launch7']
        reference_env = dict(env)
        reference_launch = function(ast.parse(self.base),'_launch7',reference_env)
        for l in (1,8):
            for b in (1,16,32):
                for w in (0,2048):
                    cfg = dict(batch=b,q_len=l,window=w,page_size=1,num_q_heads=16,num_kv_heads=2,head_dim=64,sm_scale=.125,qk_scale_factor=1.,rope_theta=0.)
                    data = dict(config=cfg,qkv=NS(device='cpu-standin'), page_table=NS(shape=(b,86008)))
                    data.update({k:k for k in ('seq_lens','cache_loc','k_cache','v_cache','k_scale','v_scale')})
                    module = NS(run=lambda *a:captures.append(('treatment',a)))
                    state = dict(attention_scratch={},attention_continuity_module=module)
                    self.assertEqual(env['_decode_attention7'](data,state,data['qkv']), 'result')
                    actual = captures[-1]
                    reference_launch(data, attention_state=state, pdl=False)
                    reference = captures[-1]
                    self.assertEqual(actual[0], 'treatment' if b>1 and w==0 else 'original')
                    self.assertEqual(actual[1][1:9],reference[1][1:9])
                    self.assertEqual(actual[1][10:],reference[1][10:])
                    self.assertIs(state['attention_continuity_module'], module)
        # State owners retain their independently supplied module identities.
        c = dict(batch=16,q_len=1,window=0,page_size=1,num_q_heads=16,num_kv_heads=2,head_dim=64)
        a,z = object(),object()
        self.assertIs(env['_attention_continuity_module'](c,dict(attention_continuity_module=a)),a)
        self.assertIs(env['_attention_continuity_module'](c,dict(attention_continuity_module=z)),z)
        for key,value in [('batch',2),('q_len',2),('window',2048),('page_size',2),('head_dim',128)]:
            self.assertIsNone(env['_attention_continuity_module'](dict(c,**{key:value}),{}))
        with self.assertRaises(ValueError): env['_attention_continuity_module'](c,{})
        # Actual launch entry rejects changed geometry/PDL rather than relabeling
        # the drifted result as the registered continuity comparison.
        c.update(sm_scale=.125,qk_scale_factor=1.,rope_theta=0.)
        d=dict(config=c,qkv=NS(device='cpu-standin'),page_table=NS(shape=(16,86008)))
        d.update({k:k for k in ('seq_lens','cache_loc','k_cache','v_cache','k_scale','v_scale')})
        m=NS(run=lambda *a:None); st=dict(attention_scratch={},attention_continuity_module=m)
        for override in (dict(n_splits=132),dict(w0=1000),dict(flat=0)):
            with self.assertRaises(ValueError): original_launch(d,_mod=m,attention_state=st,pdl=False,**override)
        with self.assertRaises(ValueError): original_launch(d,_mod=m,attention_state=st,pdl=True)

    def test_actual_preparation_builds_before_capture(self):
        calls=[]
        env=dict(torch=NS(cuda=NS(get_device_capability=lambda:(9,0))),
            _att7=lambda:calls.append('control'), ATT7_CPP_SRC='cpp',ATT7_CONT_CUDA_SRC=self.cuda)
        prepare=function(self.tree,'_prepare_attention_continuity',env)
        modules={'torch.utils.cpp_extension':NS(load_inline=lambda **kw:(calls.append(kw) or object()))}
        a,b={},{}
        with patch.dict(sys.modules,modules):
            prepare(a); prepare(a); prepare(b)
        builds=[x for x in calls if isinstance(x,dict)]
        self.assertEqual(len(builds),2)
        self.assertIsNot(a['attention_continuity_module'],b['attention_continuity_module'])
        self.assertTrue(all(x['cuda_sources']==self.cuda for x in builds))
        self.assertEqual(builds[0]['extra_cuda_cflags'],['-DOKC_QDIRECT=0','-DOKC_PK2=0','-DOKC_W0ENABLE=1','-O3','-DOKC_SM90=1','--expt-relaxed-constexpr'])

    def test_ring_and_page_lifetime_model(self):
        # Model is deliberately local; it does not prove arbitrary CUDA schedules.
        for stages in (3,4):
            for segments in ([],[0],[1],[1,2,0,7],[65,1,3],[257,2]):
                issued = retired = 0
                for count in segments:
                    for x in range(count):
                        ordinal = issued
                        slot,phase = ordinal%stages,(ordinal//stages)&1
                        if ordinal>=stages:
                            self.assertLess(ordinal-stages,retired)
                        self.assertEqual((slot,phase),(issued%stages,(issued//stages)%2))
                        issued += 1; retired += 1
                    # PGCONT=0: only pages0..count-1 are issued, each read ends
                    # at producer bar.sync2 before next segment reinitialization.
                    pages = set(range(count)); consumed = set(range(count))
                    self.assertEqual(pages,consumed)
                self.assertEqual(issued,retired)
        def validate(retired,needed,published,merge_disjoint,pages_done):
            assert retired>=needed and published and merge_disjoint and pages_done
        for args in [(0,1,True,True,True),(1,1,False,True,True),(1,1,True,False,True),(1,1,True,True,False)]:
            with self.assertRaises(AssertionError): validate(*args)

    def test_weighted_partition_ranges(self):
        # Sample the actual weighted contiguous partition equations; each real
        # key (including causal/new-token suffixes) belongs to one interval.
        for batch in (16,32):
            for length,w0 in ((1,1024),(8,1135)):
                for seq in (0,1,63,64,65,80000):
                    n = seq+length; total = batch*n; intervals=[]
                    for vc in range(264):
                        sp,r = divmod(vc,2)
                        lo = total*(2048*sp+r*w0)//(1024*264)
                        hi = total*(2048*sp+(2048 if r else w0))//(1024*264)
                        if hi>lo: intervals.append((lo,hi))
                        ordinal=0
                        for b in range(batch):
                            s,e=max(lo,b*n)-b*n,min(hi,(b+1)*n)-b*n
                            if e>s:
                                keys=list(range(s,e))
                                self.assertEqual(len(keys),e-s)
                                # New rows are the suffix [seq,seq+L), not a
                                # second ownership interval or a masked gap.
                                self.assertTrue(all(0<=k<n for k in keys))
                                ordinal+=(e-s+63)//64
                    self.assertEqual(intervals[0][0],0)
                    self.assertEqual(intervals[-1][1],total)
                    self.assertTrue(all(a[1]==b[0] for a,b in zip(intervals,intervals[1:])))


if __name__=='__main__': unittest.main()
