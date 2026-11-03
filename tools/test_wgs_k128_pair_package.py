"""State-layout/factory negatives and real CPU packing; not target evidence."""
from pathlib import Path
import tempfile
import unittest
import torch
import wgs_k128_pair_package as p
from test_wgs_k128_full_reuse import pack_env
from test_g_expert_g224_prefix import capture,BASE
from g_expert_g224_prefix import function_source


def environment():
    calls,e=capture(BASE,False)
    e['prepare']=lambda weights:None  # No production prepare is executed here.
    s,m=p.build();exec(s[len(BASE):],e)
    e['_MA_EXTS']={};e['_K128_EXTS']={};calls.clear()
    e['torch']=torch
    # Fixed compiler gateway only; actual CUDA calls are prohibited in CPU tests.
    e['torch_cuda_saved']=torch.cuda.get_device_capability
    return calls,e


def state():
    return dict(ma_ext={},ma_sfx=('-DOKC_PRMTSC=1',),ma_sc6=True,
        ma_w=[torch.zeros(4,dtype=torch.int32) for _ in range(14)],
        _fork_native_eligible=True,_wl_guard_storage=[])


class Pair(unittest.TestCase):
    def test_lexically_closed_original_packers(self):
        s,m=p.build();e=pack_env()
        for n in ('_sc6_wgs',):exec(function_source(BASE,n),e)
        # Only generated OFF definitions: ON helpers retain the new K128 values.
        for n in ('_k128_off_kperm_src','_k128_off_kperm_codes','_k128_off_kperm_rows','_k128_off_wgs_copy'):
            exec(function_source(s,n),e)
        old=pack_env()
        for n in ('_sc6_wgs','_kperm_src','_kperm_codes','_kperm_rows','_wgs_copy'):exec(function_source(BASE,n),old)
        rng=torch.Generator().manual_seed(1102031)
        codes=torch.randint(0,256,(256,2048),dtype=torch.uint8,generator=rng)
        scales=torch.randint(64,128,(256,256),dtype=torch.uint8,generator=rng)
        rows=torch.arange(256)
        self.assertEqual(e['_kperm_src']().numel(),128)
        self.assertEqual(e['_k128_off_kperm_src']().numel(),256)
        for sc6 in (False,True):
            a=e['_k128_off_wgs_copy'](codes,scales,rows,sc6)
            self.assertTrue(torch.equal(a,old['_wgs_copy'](codes,scales,rows,sc6)))
            b=e['_wgs_copy'](codes,scales,rows,sc6)
            self.assertEqual(a.numel(),b.numel())
            self.assertFalse(torch.equal(a,b))

    def test_actual_source_factory_arms_and_weight_negatives(self):
        from unittest.mock import patch
        calls,e=environment()
        old,_=capture(BASE,False)
        with patch.object(torch.cuda,'get_device_capability',return_value=(9,0)):
            controls=[]
            for arm in (False,False,True,True):
                st=state();e['_k128_pair_mark'](st,arm)
                for t in (1,8,16,32,128,256):e['_fork_native_moe'](t,st)
                controls.append(st)
            self.assertEqual(calls[:5],old)
            self.assertEqual(len(calls),7)
            for call in calls[5:]:self.assertEqual(call['cuda_sources'],e['_K128_CUDA'])
            self.assertIsNot(controls[0]['ma_ext'],controls[1]['ma_ext'])
            self.assertIsNot(controls[0]['ma_ext']['main'],controls[2]['ma_ext']['main'])
            st=controls[2];st['_expert_pair_arm']=False
            with self.assertRaisesRegex(RuntimeError,'REBOUND'):e['_fork_native_moe'](128,st)
            st['_expert_pair_arm']=True;st['_k128_pair_layout']='K256'
            with self.assertRaisesRegex(RuntimeError,'LAYOUT'):e['_fork_native_moe'](128,st)
            st['_k128_pair_layout']='K128';st['ma_w'][12].add_(1)
            with self.assertRaisesRegex(RuntimeError,'WEIGHT_OWNER_OR_VERSION'):e['_fork_native_moe'](128,st)

    def test_actual_bind_private_owners_rebind_and_module_negative(self):
        from unittest.mock import patch
        calls,e=environment();e['_GUARDS']=[]
        # Explicit allocation/repack boundaries, small real Torch tensors.
        e['_k128_pair_repack']=lambda w,st,arm:(torch.full((8,),int(arm)),torch.full((8,),int(arm)))
        e['_ma_make_scratch']=lambda dev:[torch.zeros(16)]
        e['_k128_off_ma_make_scratch']=lambda dev:[torch.zeros(32)]
        e['_guarded']=lambda t:t
        weights={'w13':torch.zeros(1)}
        with patch.object(torch.cuda,'get_device_capability',return_value=(9,0)):
            arms=[e['_expert_pair_bind'](state(),i>=2,weights) for i in range(4)]
            for i,a in enumerate(arms):
                self.assertEqual(a['_k128_pair_layout'],'K128' if i>=2 else 'K256')
                for b in arms[:i]:
                    pa={t.untyped_storage().data_ptr() for t in a['ma_w']+a['ma_scratch']}
                    pb={t.untyped_storage().data_ptr() for t in b['ma_w']+b['ma_scratch']}
                    self.assertFalse(pa&pb)
            with self.assertRaisesRegex(RuntimeError,'INVALID_BIND'):e['_expert_pair_bind'](arms[0],True,weights)
            arms[2]['ma_ext']['main']=arms[0]['ma_ext']['main']
            with self.assertRaisesRegex(RuntimeError,'MODULE_OWNER'):e['_fork_native_moe'](128,arms[2])
            with self.assertRaisesRegex(RuntimeError,'HOLD'):e['prepare'](weights)


if __name__=='__main__':unittest.main()
