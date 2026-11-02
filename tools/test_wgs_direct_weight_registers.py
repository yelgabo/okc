"""Actual selected source/address/RS math contracts; CPU only, no GPU calls."""
import re
from pathlib import Path
import tempfile
import unittest
import wgs_direct_weight_registers as g
from g_expert_g224_prefix import function_source
from test_g_expert_g224_prefix import capture,BASE,body,expression


def selected(sc6=True):
    calls,e=capture(BASE,False)
    with tempfile.TemporaryDirectory() as d:s,n,m=g.generate(Path(d))
    exec(s[len(BASE):],e);e['_MA_EXTS']={};e['_DW_EXTS']={};calls.clear()
    state=dict(ma_sfx=('-DOKC_PRMTSC=1',),ma_sc6=sc6,ma_ext={})
    for t in (1,8,16,32,128,256,384):e['_fork_native_moe'](t,state)
    return calls,e,n


def physical_order(pb,g0,g1):
    sp=((g1-1)//16)*16
    if pb or not(g1%16) or sp<=g0:sp=g0
    return list(range(sp,g1))+list(range(g0,sp))


class DirectWeight(unittest.TestCase):
    def test_actual_factories_short_math_packing_and_scratch_invariance(self):
        old,_=capture(BASE,False);original=g.literal(BASE,'MOE_CUDA_SRC')
        for sc6 in (False,True):
            calls,e,n=selected(sc6)
            self.assertEqual(calls[:3],old[:3]);self.assertEqual(len(calls),5)
            for c in calls[3:]:
                self.assertEqual(c['cuda_sources'],n)
                self.assertIn('-DOKC_NW=16',c['extra_cuda_cflags'])
                self.assertIn('-std=c++20',c['extra_cuda_cflags'])
                self.assertEqual(c['extra_cflags'],['-std=c++20'])
                self.assertEqual('-DOKC_SC6W=1' in c['extra_cuda_cflags'],sc6)
            with tempfile.TemporaryDirectory() as d:s,_,_=g.generate(Path(d))
            for name in ('_kperm_src','_kperm_codes','_kperm_rows','_wgs_copy','_ma_make_scratch','_release_expert_self_test'):
                self.assertEqual(function_source(s,name),function_source(BASE,name))
            for sig in ('__device__ __forceinline__ void dq4(',
                '__device__ __forceinline__ uint32_t bmul2l(', '__device__ __forceinline__ uint32_t bmul2h(',
                '__device__ __forceinline__ int wgs_split(', '__device__ __forceinline__ void wgs_flush(',
                '__device__ __forceinline__ void wgs_phase(', '__device__ __forceinline__ void wgs_prefix(',
                '__device__ __forceinline__ int wgs_stage_at(', '__device__ __forceinline__ void wgs_range(',
                '__device__ __noinline__ void wgs_ranges(', '__device__ __forceinline__ int wgs_cta_of(',
                '__device__ void wgs_gather(', '__device__ __forceinline__ void wgs_epi_tile('):
                self.assertEqual(body(n,sig),body(original,sig),sig)
            begin=original.index('template <int NTT> struct WGSMMA;');end=original.index('// two producer warps',begin)
            self.assertIn(original[begin:end],n)
            begin=original.index('#if OKC_WGS && OKC_B4FIN\n  // bet4_claude: item');end=original.index('#elif OKC_FINCOL',begin)
            self.assertIn(original[begin:end],n)

    def test_global_physical_address_both_phases_formats_all_units(self):
        n=g.transform(g.literal(BASE,'MOE_CUDA_SRC'))
        f=body(n,'__device__ __forceinline__ size_t dw_weight_offset(')
        formula=re.search(r'return (.*?);',f)[1]
        for sc6,piece in ((False,9216),(True,8960)):
          stage=piece*4
          for pb in (False,True):
            ubl=4 if pb else 5;units=1<<ubl
            bank=32*units*16*stage
            for expert in range(32):
              for ub in range(units):
                for st in range(16):
                  for wg in range(4):
                    actual=expression(formula,expert=expert,ub=ub,physical_st=st,wg=wg,ubl=ubl,WGS_STG=stage,WGS_PIECE=piece)
                    expected=(((expert*units+ub)*16+st)*4+wg)*piece
                    self.assertEqual(actual,expected)
                    self.assertEqual(actual%16,0)
                    self.assertGreaterEqual(actual,0);self.assertLessEqual(actual+piece,bank)
        seg=body(n,'__device__ __forceinline__ void wgs_seg(')
        self.assertIn('dw_weight_offset(S.jE[u >> UBL],u & ((1 << UBL)-1),0,wg,UBL)',seg)
        self.assertIn('pc0 + (size_t)physical_st * WGS_STG',seg)
        self.assertIn('stage(std::true_type{},s0)',seg)
        self.assertIn('stage(std::false_type{},st)',seg)
        self.assertNotIn('seq % 16',seg)

    def test_skewed_reordered_split_sequence_negative(self):
        cases=((False,3,13),(False,0,20),(False,17,40),(False,29,69),(True,31,65))
        for pb,g0,g1 in cases:
            order=physical_order(pb,g0,g1)
            self.assertEqual(sorted(order),list(range(g0,g1)))
            ubl=4 if pb else 5
            experts=[(j*11+7)%32 for j in range(88)]
            mismatches=0
            for initial_seq in (0,1,7,19,133):
                correct=[];wrong=[]
                for seq,gs in enumerate(order,initial_seq):
                    u,st=gs>>4,gs&15;j,ub=u>>ubl,u&((1<<ubl)-1)
                    correct.append(((experts[j]*(1<<ubl)+ub)*16+st)*35840)
                    wrong.append(((experts[j]*(1<<ubl)+ub)*16+seq%16)*35840)
                mismatches+=correct!=wrong
            self.assertGreater(mismatches,0)
        # Empty and exact-unit tails do not invent a stage.
        for pb in (False,True):
            self.assertEqual(physical_order(pb,0,0),[])
            self.assertEqual(physical_order(pb,0,16),list(range(16)))

    def test_unchanged_lane_packet_coverage_and_register_math(self):
        original=g.literal(BASE,'MOE_CUDA_SRC');n=g.transform(original)
        old=body(original,'__device__ __forceinline__ void wgs_seg(')
        new=body(n,'__device__ __forceinline__ void wgs_seg(')
        a='    // codes and scales one 4-step group at a time (fewer live registers)'
        b='    // All reads of this activation slot are complete'
        raw='''      } else {
        asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
        mbar_arrive1(S.bars + BW_E + ws);  // every code and scale of this stage is in registers: weights slot free
      }'''
        expected=old[old.index(a):old.index(b)].replace(raw,'      }')
        moved='    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);\n'
        actual=new[new.index(a):new.index(b)].replace(moved,'')
        self.assertEqual(actual,expected)
        self.assertLess(new.index('uint4 cwn ='),new.index(moved.strip()))
        self.assertLess(new.index('const uint2 scs ='),new.index(moved.strip()))
        code=set();raw_scales=set();sc6_scales=set()
        for wp in range(4):
          for lane in range(32):
            tt=wp*32+lane
            for lg in range(4):
                off=((lg*4+wp)*32+lane)*16
                code.update(range(off,off+16))
            raw_scales.update(range(8192+tt*8,8192+tt*8+8))
            sc6_scales.update(range(8192+tt*4,8192+tt*4+4))
            sc6_scales.update(range(8192+512+tt*2,8192+512+tt*2+2))
        self.assertEqual(code,set(range(8192)))
        self.assertEqual(raw_scales,set(range(8192,9216)))
        self.assertEqual(sc6_scales,set(range(8192,8960)))

    def test_sole_X_producer_no_raw_path_and_retirement(self):
        n=g.transform(g.literal(BASE,'MOE_CUDA_SRC'))
        begin=n.index('#define OKC_WGSNX')
        wgs=n[begin:n.index('template <bool PB, bool BIG>',begin)]
        self.assertGreater(len(wgs),1000)
        code=re.sub(r'//[^\n]*','',wgs)
        self.assertNotIn('wgs_wproducer',n)
        for dead in ('BW_F','BW_E','WGS_NS','UNIT_BYTES'):self.assertNotIn(dead,code)
        self.assertEqual(code.count('cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes'),1)
        x=body(n,'__device__ void wgs_xproducer(')
        self.assertIn('unsigned char* xring = ring;',x)
        self.assertIn('mf_da(p, S) + j * 16 + st) < 8',x)
        self.assertIn('bar.sync 7',x);self.assertIn('NT + 32',x)
        seg=body(n,'__device__ __forceinline__ void wgs_seg(')
        tail=seg[seg.index('// All reads of this activation slot'):]
        self.assertLess(tail.index('wg_wait<0>()'),tail.index('mbar_arrive1(S.bars + BX_E + xs)'))
        self.assertLess(tail.index('fence.proxy.async.shared::cta'),tail.index('mbar_arrive1(S.bars + BX_E + xs)'))
        self.assertIn('mf_da(p, S) + j * 16 + (ub >> 1)',n)
        self.assertIn('old + pd.nst == 16',n)
        self.assertIn('resident>=1',n);self.assertIn('cudaLaunchCooperativeKernel',n)
        self.assertIn('NTL = NT + (OKC_WGS ? 32',n)
        self.assertIn('return base + (size_t)okc::WGS_NX * okc::WGS_XS;',n)
        self.assertNotIn('grid *= 2',n)
        # Unchanged inherited bounds cover the largest384-token/tail jobs.
        for count in (1,7,8,9,32,33,63,64,257,384):
            jobs=[min(64,count-i) for i in range(0,count,64)]
            self.assertEqual(sum(jobs),count)
            self.assertTrue(all(1<=(c+7)//8<=8 for c in jobs))


if __name__=='__main__':unittest.main()
