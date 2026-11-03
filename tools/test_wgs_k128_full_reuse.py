"""Actual selected-factory and independent semantic packing checks, CPU only."""
import ast
from pathlib import Path
import tempfile
import unittest
import re
import torch
import wgs_k128_full_reuse as g
from g_expert_g224_prefix import function_source
from test_g_expert_g224_prefix import capture, BASE
from test_g_expert_g224_prefix import body, expression


def selected():
    calls,env=capture(BASE,False)
    with tempfile.TemporaryDirectory() as d:candidate,native,_=g.generate(Path(d))
    exec(candidate[len(BASE):],env)
    env['_MA_EXTS']={};env['_K128_EXTS']={}; calls.clear()
    st=dict(ma_sfx=('-DOKC_PRMTSC=1',),ma_sc6=True,ma_ext={})
    for t in (1,8,16,32,128,256,384):env['_fork_native_moe'](t,st)
    return calls,env,native


def pack_env():
    env=dict(torch=torch,DQ7=True)
    for name in ('_stored_pos','_pack_codes_nat'):
        exec(function_source(BASE,name),env)
    env['_POS']=env['_stored_pos']()
    exec(g.PACKERS,env)
    return env


class K128(unittest.TestCase):
    def test_actual_source_factory(self):
        old,_=capture(BASE,False);calls,env,native=selected()
        self.assertEqual(len(calls),5)
        self.assertEqual(calls[:3],old[:3])
        self.assertEqual(env['_K128_CUDA'],native)
        for call in calls[3:]:
            self.assertEqual(call['cuda_sources'],native)
            self.assertEqual(call['extra_cflags'],['-std=c++20'])
            self.assertIn('-DOKC_NW=8',call['extra_cuda_cflags'])
            self.assertIn('-std=c++20',call['extra_cuda_cflags'])
        self.assertIn('resident>=2',native)
        self.assertIn('cudaLaunchCooperativeKernel',native)
        self.assertIn('K128_ADMISSION',native)
        self.assertIn('cudaFuncGetAttributes(&actual,kernel)',native)
        self.assertNotIn('union { int wc',native)
        self.assertIn('FIRST && ci == 3 && pd.kind',native)
        self.assertIn('mf_da(p, S) + j * 32 + st) < 4',native)
        self.assertIn('wg_desc(xring_s, 128, 2048)',native)
        for n in range(1,9):self.assertIn('struct WGSMMA<'+str(n)+'>',native)

    def test_packing_semantic_sentinels(self):
        env=pack_env()
        # Independent logical codes and scales; every row/K/group varies.
        R,K=128,4096
        rng=torch.Generator().manual_seed(20261102)
        logical=torch.randint(0,16,(R,K),dtype=torch.uint8,generator=rng)
        codes=logical[:,::2]|(logical[:,1::2]<<4)
        sf=torch.randint(64,128,(R,K//16),dtype=torch.uint8,generator=rng)
        rows=torch.arange(R)
        for sc6 in (False,True):
            packed=env['_wgs_copy'](codes,sf,rows,sc6).view(1,32,2,4480 if sc6 else 4608)
            # Decode raw bit positions independently from original FP4 mapping.
            pos=env['_POS']
            for st in (0,1,15,16,31):
              for wg in range(2):
                pc=packed[0,st,wg]
                for wp in range(4):
                  for lane in range(32):
                    lane_g,t=lane>>2,lane&3
                    tid=wp*32+lane
                    if sc6:
                        q=int(pc[4096+tid])|(int(pc[4096+128+tid])<<8)|(int(pc[4096+256+tid])<<16)
                        scales=[((q>>(6*i))&63)+64 for i in range(4)]
                    else:scales=[int(x) for x in pc[4096+tid*4:4100+tid*4]]
                    for lg in range(4):
                      for j in range(2):
                        off=((lg*4+wp)*32+lane)*8+j*4
                        word=int.from_bytes(bytes(pc[off:off+4].tolist()),'little')
                        for rr,(hh,kk) in enumerate(((0,0),(1,0),(0,1),(1,1))):
                          for up in range(2):
                            raw=sum(((word>>pos[(rr,up,b)])&1)<<b for b in range(4))
                            c=2*(kk*4+t)+up
                            # Original packed row fragment: g/h row and t/up K.
                            source_row=wg*64+wp*16+hh*8+lane_g
                            semantic_k=st*128+16*(2*t+j)+4*lg+kk*2+up
                            self.assertEqual(raw,int(logical[source_row,semantic_k]))
                            self.assertEqual(scales[2*j+hh],int(sf[source_row,semantic_k//16]))

    def test_permutation_bijection_and_scale_identity(self):
        e=pack_env();p=e['_kperm_src']().tolist()
        self.assertEqual(sorted(p),list(range(128)))
        for v in range(64):
            old=((v<<2)*4)+0x6000
            new=((v+64)*16)+0x5c00
            self.assertEqual(old,new)
        for n in (1,2,3,8):
            for cnt in (1,7,8,9,16,24,32,33,63,64):
                padded=((cnt+7)//8)*8
                self.assertLessEqual(padded,64)
                self.assertTrue(1<=padded//8<=8)

    def test_actual_gather_semantic_k_and_descriptor_negatives(self):
        native=g.transform(g.literal(BASE,'MOE_CUDA_SRC'))
        gather=body(native,'__device__ void wgs_gather(')
        formula=re.search(r'\+ (\(8 \* \(\(group.*?);',gather)[1]
        dstline=next(line for line in gather.splitlines() if '+ ((((column' in line)
        dst=dstline.strip()[2:].rstrip(';')
        perm=pack_env()['_kperm_src']().tolist()
        def check_k(formula):
            for column in range(512):
                group=column&15;stage=column>>4
                pair=expression(formula,group=group)
                for i in range(8):
                    actual=stage*128+2*(pair+16*(i//2))+(i&1)
                    expected=stage*128+perm[group*8+i]
                    assert actual==expected,(column,i,actual,expected)
        check_k(formula)
        with self.assertRaises(AssertionError):check_k('8 * ((group >> 1) & 3) + 2 * (group >> 3) + (group & 1)')
        for ntt in range(1,9):
            for row in (0,ntt*8-1):
              seen=set()
              for column in range(512):
                actual=8*expression(dst,column=column,ntt=ntt,r=row)
                expected=(column//16)*(ntt*8)*128+(row//8)*1024+(column%16)*64+(row%8)*8
                self.assertEqual(actual,expected)
                self.assertNotIn(actual,seen);seen.add(actual)
                self.assertLess(actual+7,ntt*8*4096)
        desc=body(native,'__device__ __forceinline__ void wgs_seg(')
        self.assertIn('wg_desc(xring_s, 128, 2048)',desc)
        def validate_desc(s):assert 'wg_desc(xring_s, 128, 2048)' in s
        with self.assertRaises(AssertionError):validate_desc(desc.replace('128, 2048','128, 4096'))

    def test_gateup_rows_and_inter_layout(self):
        e=pack_env();r=torch.arange(4096).view(-1,8)
        rows=torch.stack((r,r+4096),1).reshape(-1)
        got=e['_kperm_rows'](rows).tolist();perm=e['_kperm_src']().tolist()
        for r16 in range(512):
            for lane_g in range(8):
                col=r16*8+lane_g
                logical=(col//128)*128+perm[col%128]
                self.assertEqual(got[r16*16+lane_g],logical)
                self.assertEqual(got[r16*16+8+lane_g],logical+4096)
        self.assertEqual(sorted(got),list(range(8192)))
        native=g.transform(g.literal(BASE,'MOE_CUDA_SRC'))
        epi=body(native,'__device__ __forceinline__ void wgs_epi_tile(')
        self.assertIn('col >> 7',epi);self.assertIn('col & 127',epi)
        # Both sides use the same physical permutation index, and code operands
        # pair it with the above logical channel, not a changed activation.
        for cnt in (1,7,8,9,16,24,32,33,63,64):
            ntt=(cnt+7)//8
            for col in range(4096):
                for tk in (0,cnt-1):
                    address=(col&7)+(((col//128)*ntt+tk//8)*16+(col%128)//8)*64+(tk%8)*8
                    self.assertLess(address,ntt*8*4096)

    def test_stage_split_and_publication_contract(self):
        native=g.transform(g.literal(BASE,'MOE_CUDA_SRC'))
        split=body(native,'__device__ __forceinline__ int wgs_split(')
        self.assertIn('((g1 - 1) >> 5) << 5',split)
        for begin,end in ((0,1),(0,31),(0,32),(0,33),(17,65),(31,97),(512,1025)):
            last=((end-1)//32)*32
            sp=last if end%32 and last>begin else begin
            visits=list(range(sp,end))+list(range(begin,sp))
            self.assertEqual(sorted(visits),list(range(begin,end)))
        for original,mutated in [('>> 5) << 5','>> 4) << 4'),('j * 32 + st) < 4','j * 16 + st) < 8')]:
            self.assertIn(original,native)
            with self.assertRaises(AssertionError):
                changed=native.replace(original,mutated)
                assert original in changed
        self.assertIn('s1 - s0 == 32',native)
        self.assertIn('wgs_cta_of(S, PB, u << 5',native)
        self.assertIn('old + pd.nst == 32',native)
        self.assertNotIn('JMAX * 16',native)

    def test_actual_scratch_and_odd_job_capacity(self):
        # Execute the actual allocation entry, without allocating giant buffers.
        class Shapes:
            int32='i32';float32='f32';bfloat16='bf16'
            def zeros(self,shape,**kw):return dict(shape=shape,**kw)
        env=dict(torch=Shapes(),_MA_EMAX=384*8,_df_inter=lambda t:t)
        exec(function_source(g.APPENDIX,'_ma_make_scratch'),env)
        a=env['_ma_make_scratch']('cpu');b=env['_ma_make_scratch']('cpu')
        emax=384*8;jmax=32+emax//64+8
        self.assertEqual(jmax,88)
        self.assertEqual(a[0]['shape'],8+1024+emax+8192)
        self.assertGreaterEqual(8192,512+2*jmax*32)
        self.assertEqual(a[1]['shape'],emax+emax*4096+4345636)
        self.assertEqual(a[2]['shape'],(emax+2*(emax+8*jmax))*4096)
        for x,y in zip(a,b):self.assertIsNot(x,y)
        for count in (33,64,257,384):
            jobs=[min(64,count-i) for i in range(0,count,64)]
            self.assertEqual(sum(jobs),count)
            self.assertTrue(all(1<=n<=64 for n in jobs))
            self.assertTrue(all(1<=(n+7)//8<=8 for n in jobs))
        # An extreme all-eight-hot-experts assignment still fits jmax.
        self.assertLessEqual(8*((384+63)//64),jmax)
        for stage in range(32):
            publications=[(unit,wg) for unit in range(64) for wg in range(2) if unit//2==stage]
            self.assertEqual(len(publications),4)
        for grid in (2,132,264):
            self.assertLess(grid,265)
            self.assertLess((grid*2+1)*4096,265*2*4096)


if __name__=='__main__':unittest.main()
