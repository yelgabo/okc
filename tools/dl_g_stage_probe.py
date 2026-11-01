"""Sparse observer-only stamps on exact #303. Compile modes split ready/retire."""
import hashlib
DEVICE_SHA="c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71"
ROWS,WIDTH,WORDS=24,7,475
HELPERS=r'''
#if G_STAGE_PROBE
static_assert(OKC_WGS && NW==16 && !OKC_MFIN && WGS_NS==3 && WGS_NX==3 && JMAX==88,"exact stage probe ABI");
__device__ __forceinline__ unsigned long long gsp_tick(){unsigned long long t;asm volatile("mov.u64 %0, %globaltimer;":"=l"(t)::"memory");return t;}
__device__ __forceinline__ int gsp_cta(){int c=blockIdx.x;return c==0?0:c==gridDim.x/2?1:c==gridDim.x-1?2:-1;}
template<bool PB,int FIELD>
__device__ __forceinline__ void gsp_mark(const Params&,const Smem&,int seq){
 asm volatile(
  "{ .reg .u32 tid,org,ord,k,plo,phi,b,ph,bits; .reg .u64 base,off,addr,t; .reg .pred skip,p0,p1,p2,p3,sel; "
  "mov.u32 tid, %%tid.x; setp.ne.u32 skip,tid,0; @skip bra GSP_DONE_%=; "
  "ld.volatile.shared.u32 plo,[%1]; ld.volatile.shared.u32 phi,[%1+4]; mov.b64 base,{plo,phi}; setp.eq.u64 skip,base,0; @skip bra GSP_DONE_%=; "
  "ld.volatile.shared.u32 org,[%2]; sub.u32 ord,%0,org; "
  "setp.eq.u32 p0,ord,0; setp.eq.u32 p1,ord,7; setp.eq.u32 p2,ord,15; setp.eq.u32 p3,ord,31; "
  "or.pred sel,p0,p1; or.pred sel,sel,p2; or.pred sel,sel,p3; @!sel bra GSP_DONE_%=; "
  "mov.u32 k,0; @p1 mov.u32 k,1; @p2 mov.u32 k,2; @p3 mov.u32 k,3; "
  "mul.wide.u32 off,k,56; add.u64 addr,base,off; add.u64 addr,addr,%3; "
  "setp.ne.u32 skip,%3,0; @skip bra GSP_TIME_%=; div.u32 ph,%0,3; and.b32 ph,ph,1; rem.u32 b,%0,3; shl.b32 b,b,3; add.u32 b,b,%4; "
  "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 p0,[b],ph; add.u32 b,b,48; "
  "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 p1,[b],ph; "
  "selp.u32 bits,1,0,p0; selp.u32 b,2,0,p1; or.b32 bits,bits,b; st.global.u32 [addr+48],bits; GSP_TIME_%=: "
  "mov.u64 t, %%globaltimer; st.global.u64 [addr],t; GSP_DONE_%=: }"
  ::"r"(seq),"r"(smem_u32(&reinterpret_cast<const Smem*>(smem_raw)->fcnt[0])),
    "r"(smem_u32(&reinterpret_cast<const Smem*>(smem_raw)->fcnt[2])),"n"(FIELD*8),"r"(smem_u32(&reinterpret_cast<const Smem*>(smem_raw)->bars[0])):"memory");
}
template<bool PB>
__device__ __forceinline__ void gsp_phase(const Params&p,Smem&S,int g0,int g1,int seq,bool end){
 if(threadIdx.x!=0)return;const int c=gsp_cta();
 if(!end){
  const unsigned long long ptr=c<0?0:reinterpret_cast<unsigned long long>(p.stage_trace+(c*2+(PB?1:0))*4*7);
  S.fcnt[0]=(uint32_t)ptr;S.fcnt[1]=(uint32_t)(ptr>>32);
  S.fcnt[2]=seq;
 }
 if(c<0)return;
 auto*r=p.stage_trace+168+(c*2+(PB?1:0))*6;
 if(end){r[3]=seq;r[5]=gsp_tick();}else{r[0]=g0;r[1]=g1;r[2]=seq;r[4]=gsp_tick();}
}
__device__ __forceinline__ void gsp_jobs(const Params&p,const Smem&S){
 if(threadIdx.x==0 && blockIdx.x==0){const int n=S.njF+S.njS;p.stage_trace[207]=n;
  for(int j=0;j<n;++j){p.stage_trace[208+3*j]=S.jE[j];p.stage_trace[209+3*j]=S.jS[j];p.stage_trace[210+3*j]=S.jC[j];}}
}
#endif
'''
def patch_device(cuda):
 if hashlib.sha256(cuda.encode()).hexdigest()!=DEVICE_SHA:raise ValueError("unreviewed native stage source")
 def ins(anchor,extra,before=False):
  nonlocal cuda
  if cuda.count(anchor)!=1:raise ValueError(("ambiguous stage probe anchor",anchor,cuda.count(anchor)))
  cuda=cuda.replace(anchor,extra+anchor if before else anchor+extra)
 cuda="#ifndef G_STAGE_PROBE\n#define G_STAGE_PROBE 0\n#endif\n"+cuda
 ins("struct Params {\n","#if G_STAGE_PROBE\n  unsigned long long* stage_trace;\n#endif\n")
 ins("  DBG(0);\n","#if G_STAGE_PROBE\n  if(tid==0 && gsp_cta()>=0)p.stage_trace[472+gsp_cta()]=gsp_tick();\n#endif\n")
 ins("template <bool PB, int NTT>\n__device__ __forceinline__ void wgs_seg(",HELPERS,before=True)
 ins("    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);\n","#if G_STAGE_PROBE == 1\n    gsp_mark<PB,0>(p,S,seq);\n#endif\n",before=True)
 ins("    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);\n","#if G_STAGE_PROBE == 1\n    gsp_mark<PB,1>(p,S,seq);\n#endif\n")
 ins("    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);\n","#if G_STAGE_PROBE == 1\n    gsp_mark<PB,2>(p,S,seq);\n#endif\n")
 ins("    // All reads of this activation slot are complete before the ring can reuse it.\n","#if G_STAGE_PROBE == 2\n    gsp_mark<PB,3>(p,S,seq);\n#endif\n")
 anchor="    wg_wait<0>();\n    asm volatile(\"fence.proxy.async.shared::cta;\\n\" ::: \"memory\");"
 if cuda.count(anchor)!=1:raise ValueError("expected stage retirement")
 cuda=cuda.replace(anchor,"    wg_wait<0>();\n#if G_STAGE_PROBE == 2\n    gsp_mark<PB,4>(p,S,seq);\n#endif\n    asm volatile(\"fence.proxy.async.shared::cta;\\n\" ::: \"memory\");")
 ins("    mbar_arrive1(S.bars + BX_E + xs);\n","#if G_STAGE_PROBE == 2\n    gsp_mark<PB,5>(p,S,seq);\n#endif\n")
 ins("  WgsPend pd{0, 0, 0, 0};\n","#if G_STAGE_PROBE\n  gsp_phase<PB>(p,S,g0,g1,seq,false);\n#endif\n")
 anchor="  wgs_flush<PB>(p, S, pd);\n}\n// cost-weighted"
 if cuda.count(anchor)!=1:raise ValueError("expected phase final flush")
 cuda=cuda.replace(anchor,"  wgs_flush<PB>(p, S, pd);\n#if G_STAGE_PROBE\n  gsp_phase<PB>(p,S,g0,g1,seq,true);\n#endif\n}\n// cost-weighted")
 ins("  route(p, S);\n","#if G_STAGE_PROBE\n  gsp_jobs(p,S);\n#endif\n")
 ins("    int* bc = p.ctr + 4 + (S.gen & 1);\n","#if G_STAGE_PROBE\n    if(gsp_cta()>=0)p.stage_trace[204+gsp_cta()]=gsp_tick();\n#endif\n",before=True)
 ins("  p.out = (__nv_bfloat16*)out.data_ptr();\n","#if G_STAGE_PROBE\n  TORCH_CHECK(scratch.size()==5 && scratch[4].device()==x.device() && scratch[4].scalar_type()==at::kLong && scratch[4].is_contiguous() && scratch[4].numel()==475, \"G_STAGE_PROBE trace ABI\");\n  p.stage_trace=reinterpret_cast<unsigned long long*>(scratch[4].data_ptr<int64_t>());\n#endif\n")
 return cuda
