"""Actual selected source, exhaustive producer layout and CPU bit-domain checks."""
from pathlib import Path
import random
import re
import tempfile
import unittest
import torch
import wgs_shared_a_pipeline as g
from g_expert_g224_prefix import function_source
from test_g_expert_g224_prefix import capture,BASE,body,expression


def selected(sc6=True):
    calls,e=capture(BASE,False)
    with tempfile.TemporaryDirectory() as d:s,n,m=g.generate(Path(d))
    exec(s[len(BASE):],e);e['_MA_EXTS']={};e['_SSA_EXTS']={};calls.clear()
    state=dict(ma_sfx=('-DOKC_PRMTSC=1',),ma_sc6=sc6,ma_ext={})
    for t in (1,8,16,32,128,256,384):e['_fork_native_moe'](t,state)
    return calls,e,n


def pack_env():
    e=dict(torch=torch,DQ7=True)
    for name in ('_stored_pos','_pack_codes_nat'):exec(function_source(BASE,name),e)
    e['_POS']=e['_stored_pos']();exec(g.packers(),e)
    return e


class SharedA(unittest.TestCase):
    def test_actual_factories_and_unchanged_numerical_boundaries(self):
        old,_=capture(BASE,False)
        for sc6 in (False,True):
            calls,e,n=selected(sc6)
            self.assertEqual(len(calls),5)
            # Compare same raw/SC6 flavor's short factories (unaffected).
            self.assertEqual(calls[:3],old[:3])
            for c in calls[3:]:
                self.assertEqual(c['cuda_sources'],n)
                self.assertIn('-DOKC_NW=16',c['extra_cuda_cflags'])
                self.assertEqual(c['extra_cflags'],['-std=c++20'])
                self.assertIn('-std=c++20',c['extra_cuda_cflags'])
                self.assertEqual('-DOKC_SC6W=1' in c['extra_cuda_cflags'],sc6)
            original=g.literal(BASE,'MOE_CUDA_SRC')
            for sig in ('__device__ __forceinline__ void dq4(',
                        '__device__ __forceinline__ uint32_t bmul2l(',
                        '__device__ __forceinline__ uint32_t bmul2h('):
                self.assertEqual(body(n,sig),body(original,sig))
            # Full finalization is identical, not a remapped two-WG finalizer.
            a=original.index('#if OKC_WGS && OKC_B4FIN\n  // bet4_claude: item')
            b=n.index('#if OKC_WGS && OKC_B4FIN\n  // bet4_claude: item')
            self.assertEqual(original[a:original.index('#elif OKC_FINCOL',a)],n[b:n.index('#elif OKC_FINCOL',b)])
            self.assertIn('NTL = NT + (OKC_WGS ? 160',n)
            self.assertIn('if (tid < NT + 128) wgs_wproducer',n)
            self.assertIn('resident>=1',n);self.assertIn('cudaLaunchCooperativeKernel',n)
            self.assertNotIn('grid *= 2',n)
            self.assertIn('SSA_ADMISSION',n)

    def test_exhaustive_producer_store_and_ss_descriptor(self):
        n=g.transform(g.literal(BASE,'MOE_CUDA_SRC'))
        fn=body(n,'__device__ __forceinline__ int ssa_f(')
        formula=re.search(r'return (.*?);',fn)[1]
        seen=set()
        for target in range(4):
          for dt in range(128):
            wp,lane=dt>>5,dt&31
            for step in range(8):
              row=target*64+16*wp+(lane>>2);k=step*16+2*(lane&3)
              for rr,kk in ((0,0),(8,0),(0,8),(8,8)):
                for half in (0,1):
                    r,c=row+rr,k+kk+half
                    actual=expression(formula,row=r,k=c)
                    # Independently decoded no-swizzle SS descriptor:
                    # WG base + row8*SBO + K8*LBO + inner row/column.
                    expected=target*8192+(r%64//8)*1024+(c//8)*64+(r%8)*8+c%8
                    self.assertEqual(actual,expected)
                    self.assertNotIn(actual,seen);seen.add(actual)
        self.assertEqual(seen,set(range(256*128)))
        producer=body(n,'__device__ void wgs_wproducer(')
        expected=[('row,k','bmul2l(r0,sgp)'),('row+8,k','bmul2h(r1,sgp)'),('row,k+8','bmul2l(r2,sgp)'),('row+8,k+8','bmul2h(r3,sgp)')]
        stores=re.findall(r'a\+ssa_f\(([^)]+)\)\)=(bmul2[lh]\(r\d,sgp\))',producer)
        self.assertEqual(stores,expected)
        seg=body(n,'__device__ __forceinline__ void wgs_seg(')
        self.assertIn('wg_desc(aring_s,128,2048)',seg)
        self.assertIn('wg_desc(xring_s,128,2048)',seg)
        self.assertIn('(ci*2+jj)*16',seg)
        # Using the old pitch is a discriminating wrong descriptor.
        wrong=lambda r,c:(r//8)*2048+(c//8)*64+(r%8)*8+c%8
        self.assertNotEqual(wrong(8,0),expression(formula,row=8,k=0))
        for ntt in range(1,9):
            addresses={(r//8)*1024+(c//8)*64+(r%8)*8+c%8 for r in range(ntt*8) for c in range(128)}
            self.assertEqual(addresses,set(range(ntt*8*128)))

    def test_full_k_raw_sc6_decode_bits_and_logical_matrix(self):
        e=pack_env();R,K=256,4096
        rng=torch.Generator().manual_seed(202611022)
        logical=torch.randint(0,16,(R,K),dtype=torch.uint8,generator=rng)
        codes=logical[:,::2]|(logical[:,1::2]<<4)
        scales=torch.randint(64,128,(R,K//16),dtype=torch.uint8,generator=rng)
        rows=torch.randperm(R,generator=rng)
        perm=e['_kperm_src']()
        # Physical [row,K] reference from original logical FP4 nibble bit table
        # and original raw scale construction, independent of packed bit slots.
        ref_codes=logical.index_select(0,rows).view(R,32,128).index_select(2,perm)
        ref_bits=(((ref_codes.long()&8)<<12)|((ref_codes.long()&7)<<6)).to(torch.int16)
        # The legal PRMT raw path also covers positive scale bytes8..126,
        # beyond SC6's compressed domain. Same bits, not truncated scales.
        raw_scales=torch.randint(8,127,(R,K//16),dtype=torch.uint8,generator=rng)
        self.assertEqual(set(raw_scales.flatten().tolist()),set(range(8,127)))
        for sc6,scale_input in ((False,raw_scales),(False,scales),(True,scales)):
            ref_scale=scale_input.index_select(0,rows).repeat_interleave(16,1).view(R,32,128).index_select(2,perm)
            scale_bits=(ref_scale.long()*16+0x5c00).to(torch.int16)
            expected=(ref_bits.view(torch.bfloat16)*scale_bits.view(torch.bfloat16)).view(torch.int16)
            packed=e['_wgs_copy'](codes,scale_input,rows,sc6).view(32,4,4480 if sc6 else 4608)
            raw=packed[:,:,:4096].reshape(32,4,4,128,2,4).long()
            q=sum(raw[...,i]<<(8*i) for i in range(4))
            mask=0x81c081c0
            r0=q&mask;r1=(q*16)&mask;r2=(((q<<10)|(q>>22))&0xffffffff)&mask
            r3=((q*64)&0x80c080c0)|((((q<<14)|(q>>18))&0xffffffff)&0x01000100)
            regs=torch.stack((r0,r1,r2,r3),-1)
            if sc6:
                sb=packed[:,:,4096:].reshape(32,4,3,128).long()
                z=sb[:,:,0]|(sb[:,:,1]<<8)|(sb[:,:,2]<<16)
                sb=torch.stack([((z>>(6*i))&63)+64 for i in range(4)],-1)
            else:sb=packed[:,:,4096:].reshape(32,4,128,4).long()
            values=torch.empty((32,256,128),dtype=torch.int16)
            for target in range(4):
              for lg in range(4):
                for js in range(2):
                  dt=torch.arange(128);wp=dt>>5;lane=dt&31
                  row=target*64+wp*16+(lane>>2)
                  k=(lg*2+js)*16+2*(lane&3)
                  for af,(dr,dk) in enumerate(((0,0),(8,0),(0,8),(8,8))):
                    scale=(sb[:,target,:,2*js+(af%2)]*16+0x5c00).to(torch.int16).view(torch.bfloat16)
                    for half in (0,1):
                        bits=((regs[:,target,lg,:,js,af]>>(16*half))&65535).to(torch.int16)
                        result=(bits.view(torch.bfloat16)*scale).view(torch.int16)
                        values[:,row+dr,k+dk+half]=result
            self.assertTrue(torch.equal(values.permute(1,0,2),expected))
        self.assertIn('const uint32_t k16c=16u+p.zero',g.SSA_PRODUCER)

    def test_gather_epi_row_permutations(self):
        e=pack_env();n=g.transform(g.literal(BASE,'MOE_CUDA_SRC'))
        gather=body(n,'__device__ void wgs_gather(')
        formula=re.search(r'\+ (\(8 \* \(\(G.*?);',gather)[1]
        for tid in range(512):
            group=tid&15;pair=expression(formula,G=group)
            for i in range(8):
                actual=(tid>>4)*128+2*(pair+16*(i//2))+(i&1)
                expected=(tid>>4)*128+int(e['_kperm_src']()[group*8+i])
                self.assertEqual(actual,expected)
        rows=torch.arange(4096).view(-1,8)
        original=torch.stack((rows,rows+4096),1).reshape(-1)
        got=e['_kperm_rows'](original);p=e['_kperm_src']()
        for col in range(4096):
            logical=(col//128)*128+int(p[col%128])
            self.assertEqual(int(got[(col//8)*16+col%8]),logical)
            self.assertEqual(int(got[(col//8)*16+8+col%8]),logical+4096)
        epi=body(n,'__device__ __forceinline__ void wgs_epi_tile(')
        self.assertIn('col >> 7',epi);self.assertIn('col & 127',epi)
        for cnt in (1,7,8,9,16,32,33,63,64):
            ntt=(cnt+7)//8
            for col in range(4096):
                tk=cnt-1
                pos=(col&7)+(((col//128)*ntt+tk//8)*16+(col%128)//8)*64+(tk%8)*8
                self.assertLess(pos,ntt*8*4096)

    def test_publication_and_raw_A_X_retirement(self):
        n=g.transform(g.literal(BASE,'MOE_CUDA_SRC'))
        self.assertEqual(n.count('j * 32 + ub'),2)
        self.assertIn('j * 32 + st) < 4',n)
        self.assertNotIn('(ub >> 1)',n[n.index('#define OKC_WGSNX'):n.index('template <bool PB, bool BIG>')])
        producer=body(n,'__device__ void wgs_wproducer(')
        for x in ('fence.proxy.async.shared::cta','mbar_arrive1(S.bars+BA_F+slot)','mbar_arrive1(S.bars+BW_E+slot)','bar.sync 8, 128'):
            self.assertIn(x,producer)
        self.assertLess(producer.index('fence.proxy.async.shared::cta'),producer.index('mbar_arrive1(S.bars+BA_F+slot)'))
        seg=body(n,'__device__ __forceinline__ void wgs_seg(')
        end=seg.index('// All512 consumers')
        self.assertLess(seg.index('wg_wait<0>();',end),seg.index('mbar_arrive1(S.bars+BA_E+slot)',end))
        self.assertLess(seg.index('wg_wait<0>();',end),seg.index('mbar_arrive1(S.bars+BX_E+slot)',end))
        self.assertIn('ci==3 && pd.kind',seg)
        for stage in range(32):self.assertEqual(len([(u,wg) for u in range(32) for wg in range(4) if u==stage]),4)
        # CPU state model: skewed retirement of four WGs never reuses an A slot
        # before all4 groups retired. Not a GPU memory-ordering proof.
        for total in (0,1,2,3,31,32,33,65):
          for seed in range(8):
            rng=random.Random(seed);produced=0;consumed=[0]*4;slots=[-1,-1]
            for tick in range(100000):
                if produced==total and consumed==[total]*4:break
                choices=list(range(5));rng.shuffle(choices)
                progress=False
                for who in choices:
                    if who==4:
                        if produced<total and (produced<2 or min(consumed)>produced-2):
                            slots[produced&1]=produced;produced+=1;progress=True
                    else:
                        seq=consumed[who]
                        if seq<produced:
                            self.assertEqual(slots[seq&1],seq)
                            consumed[who]+=1;progress=True
                self.assertTrue(progress)
            else:self.fail('CPU slot model failed progress')

    def test_stage_order_ranges_scratch_and_bounds(self):
        n=g.transform(g.literal(BASE,'MOE_CUDA_SRC'))
        self.assertIn('constexpr int UBL = PB ? 4 : 5',n)
        self.assertIn('s1 - s0 == 32',n);self.assertIn('s1 == 32',n)
        self.assertIn('old + pd.nst == 32',n);self.assertIn('wgs_cta_of(S, PB, u << 5',n)
        self.assertNotIn('JMAX * 16',n)
        for a,b in ((0,0),(0,1),(0,31),(0,32),(0,33),(17,65),(31,97)):
            sp=(((b-1)//32)*32) if b%32 and ((b-1)//32)*32>a else a
            visits=list(range(sp,b))+list(range(a,sp))
            self.assertEqual(sorted(visits),list(range(a,b)))
        # Actual flattened producer gs ordering must equal per-phase consumer.
        f=body(n,'__device__ __forceinline__ int ssa_gs(')
        self.assertIn('pb=n>=na',f);self.assertIn('if(gs>=g1)gs=g0+gs-g1',f)
        class Shapes:
            int32='i';float32='f';bfloat16='b'
            def zeros(self,shape,**kwargs):return dict(shape=shape,**kwargs)
        e=dict(torch=Shapes(),_MA_EMAX=3072,WGS_XROWS=4160,_df_inter=lambda t:t)
        exec(function_source(g.APPENDIX,'_ma_make_scratch'),e)
        a=e['_ma_make_scratch']('cpu');b=e['_ma_make_scratch']('cpu')
        self.assertEqual(a[1]['shape'],3072+3072*4096+5246976)
        self.assertGreaterEqual(8192,512+2*88*32)
        self.assertEqual(a[2]['shape'],(3072+2*4160)*4096)
        for x,y in zip(a,b):self.assertIsNot(x,y)
        for count in (33,64,257,384):
            jobs=[min(64,count-i) for i in range(0,count,64)]
            self.assertTrue(all(1<=(c+7)//8<=8 for c in jobs))
            self.assertEqual(sum(jobs),count)


if __name__=='__main__':unittest.main()
