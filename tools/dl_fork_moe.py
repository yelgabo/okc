"""Conservative native-MoE repairs for the pinned attributed layer fork.

No GPU execution. Unsupported native scale/fallback/resource paths fail closed;
the coordinator must select the separately validated owned/Marlin adapter there.
The imported external source/notices remain in the caller's private fork.
"""
import ast
import hashlib
from pathlib import Path
import re
import sys

PIN='82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19'  # LF-normalized


def once(s, old, new, n=1):
    assert s.count(old)==n, (old[:100],s.count(old),n)
    return s.replace(old,new)


def wrap_balanced(s, prefix):
    """Wrap complete fixed-prefix C++ function expressions, including nested ldgs."""
    at=0; count=0
    while True:
        a=s.find(prefix,at)
        if a<0: break
        z=a+len(prefix); depth=1
        while depth:
            depth += (s[z]=='(')-(s[z]==')'); z+=1
        s=s[:a]+'fork_bf16('+s[a:z]+')'+s[z:]
        at=z+len('fork_bf16()'); count+=1
    return s,count


COOPERATIVE=r'''
  // Ordinary same-stream cooperative launch: prior PDL producers finish first.
  // No native launch may depend on an oversubscribed software grid barrier.
  int device=0; cudaDeviceProp prop{}; int resident=0;
  TORCH_CHECK(cudaGetDevice(&device)==cudaSuccess && cudaGetDeviceProperties(&prop,device)==cudaSuccess,
              "FORK_MOE_RESOURCE device properties");
  TORCH_CHECK(prop.major==9 && prop.minor==0 && prop.cooperativeLaunch,
              "FORK_MOE_RESOURCE requires SM90 cooperative launch; use owned/Marlin adapter");
  TORCH_CHECK(grid>0 && grid<=prop.multiProcessorCount && grid<=136 && grid<160,
              "FORK_MOE_RESOURCE grid/fixup bound; use owned/Marlin adapter");
  TORCH_CHECK(p.M>0 && p.M<=okc::MMAX && p.nloc==32 && p.lo==0,
              "FORK_MOE_RESOURCE shape/local-ID ABI");
  TORCH_CHECK(x.scalar_type()==torch::kBFloat16 && out.scalar_type()==torch::kBFloat16
              && ids.scalar_type()==torch::kInt32 && tw.scalar_type()==torch::kFloat32,
              "FORK_MOE_RESOURCE activation/routing dtype ABI");
  auto kernel = p.M>=OKC_BIGM ? okc::moe_fused<true> : okc::moe_fused<false>;
  TORCH_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm)==cudaSuccess,
              "FORK_MOE_RESOURCE dynamic shared-memory attribute");
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,okc::NTL,sm)==cudaSuccess
              && resident>=1 && grid<=resident*prop.multiProcessorCount,
              "FORK_MOE_RESOURCE cooperative residency");
  void* args[]={&p};
  const cudaError_t launch=cudaLaunchCooperativeKernel((const void*)kernel,dim3((unsigned)grid),
      dim3(okc::NTL),args,sm,st);
  TORCH_CHECK(launch==cudaSuccess,"FORK_MOE_LAUNCH ",cudaGetErrorString(launch));
'''

HOST_BRIDGE=r'''
# The attributed native code is optional. A proven owned adapter is mandatory,
# including when scales do not admit the native PRMT path or resources fail.
# Its generator is tools/dl_fork_owned_moe.py (composed by the coordinator).
_moe_tiny = lambda: None
_moe_tiny2 = lambda: None
_moe_mid = lambda: None

class _ForkMarlinAdapter:
    def __init__(self,state): self.state=state
    def okc_set_pdl(self,on): pass  # ordinary same-stream library calls
    def okc_run(self,w,s,x,ids,tw,out,lo,nloc,grid):
        if lo!=0 or nloc!=32 or not 1<=x.shape[0]<=384:
            raise RuntimeError('FORK_MOE_FALLBACK local-ID/chunk ABI')
        from sgl_kernel.scalar_type import scalar_types
        from sglang.kernels.ops.activation.activation import silu_and_mul
        from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
        from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
        from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum
        bank=self.state['_fork_marlin_bank']; m,h=x.shape
        tw=tw.float().contiguous(); ids=ids.contiguous()
        for bm in (8,16,32,48,64):
            if m*8/32/bm<.9: break
        sorted_ids,experts,padded=moe_align_block_size(ids,bm,256)
        c2=torch.empty((m*8,h),device=x.device,dtype=x.dtype)
        c13=torch.empty((m*8*2*h,),device=x.device,dtype=x.dtype)
        c1=c13.view(-1,2*h); c3=c13[:m*8*h].view(-1,h)
        common=dict(moe_block_size=bm,mul_topk_weights=False,is_ep=True,
            b_q_type=scalar_types.float4_e2m1f,is_k_full=True,use_atomic_add=False,
            use_fp32_reduce=True,is_zp_float=False)
        c1=moe_wna16_marlin_gemm(x,c1,bank['w13'],None,bank['w13_s'],bank['w13_gs'],None,None,None,
            bank['workspace'],sorted_ids,experts,padded,tw,top_k=8,size_m=m,size_n=2*h,size_k=h,**common)
        silu_and_mul(c1,c2,expert_ids=ids.view(-1),expert_step=1)
        y=moe_wna16_marlin_gemm(c2,c3,bank['w2'],None,bank['w2_s'],bank['w2_gs'],None,None,None,
            bank['workspace'],sorted_ids,experts,padded,tw,top_k=1,size_m=m*8,size_n=h,size_k=h,**common).view(m,8,h)
        out.copy_(fused_post_expert_norm_weighted_sum(y,tw,norm_weight=self.state['_fork_marlin_gamma'],
            eps=1e-8,weight_first=True,slot_expert_ids=ids))

def _fork_prepare_marlin(weights,st):
    st['_fork_marlin_bank']=_marlin_bank(weights)  # clones before SGLang loader mutates layouts
    st['_fork_marlin_gamma']=(weights['post_expert_norm'].float()+1).to(torch.bfloat16)
    st['_fork_marlin_adapter']=_ForkMarlinAdapter(st)

def _fork_moe_fallback(M,st):
    if st['_fork_fallback_kind']=='owned': return _fork_owned_moe_for(M,st)
    return st['_fork_marlin_adapter']

def _ma_prepare(weights, st):
    installed=_fork_install_owned_moe(weights, st)
    st['_fork_fallback_kind']='owned' if installed else 'Marlin'
    if not installed: _fork_prepare_marlin(weights,st)
    torch.cuda.synchronize()  # packing/earlier execution faults never become optional
    dev=weights['w13'].device
    prop=torch.cuda.get_device_properties(dev)
    eligible=bool(st['ma_sfx']) and prop.major==9 and prop.minor==0
    st['_fork_native_eligible']=eligible
    st['_fork_moe_status']='native pending' if eligible else st['_fork_fallback_kind']+': scale/architecture fallback'
    if not eligible:
        st['ma_w']=()
        st['ma_scratch']=()
        if not installed:
            # Warm the exact official fallback before any graph capture.
            for m in (1,8,16,32,33,128,256):
                x=torch.zeros((m,4096),device=dev,dtype=torch.bfloat16)
                ids=torch.full((m,8),-1,device=dev,dtype=torch.int32);ids[:,0]=0
                tw=torch.full((m,8),.125,device=dev,dtype=torch.float32)
                st['_fork_marlin_adapter'].okc_run((),(),x,ids,tw,torch.empty_like(x),0,32,prop.multi_processor_count)
                torch.cuda.synchronize()
        print('FORK_MOE_ROUTE',st['_fork_moe_status'],flush=True)
        return
    _fork_native_ma_prepare(weights, st)
    torch.cuda.synchronize()  # separate GPU repack from CPU-only compilation fallback
    try:
        for m in (1,8,16,32,33,128,256):
            _fork_native_moe(m,st)  # compile outside capture, only supported paths
    except RuntimeError as exc:
        # No kernels are executed by module compilation. CUDA execution failures
        # from the probes below are deliberately NOT optional.
        st['_fork_native_eligible']=False
        st['_fork_moe_status']=st['_fork_fallback_kind']+': native build failed '+str(exc)[-300:]
        print('FORK_MOE_ROUTE',st['_fork_moe_status'],flush=True)
        return
    for m in (1,8,16,32,33,128,256):
        x=torch.zeros((m,4096),device=dev,dtype=torch.bfloat16)
        ids=torch.full((m,8),-1,device=dev,dtype=torch.int32)
        ids[:,0]=0  # one valid local route; no all-empty scheduler assumption
        tw=torch.full((m,8),.125,device=dev,dtype=torch.float32)
        out=torch.empty_like(x)
        torch.cuda.synchronize()  # input setup faults cannot look like a resource miss
        try:
            _fork_native_moe(m,st).okc_run(st['ma_w'],st['ma_scratch'],x,ids,tw,out,0,32,
                                         torch.cuda.get_device_properties(dev).multi_processor_count)
        except RuntimeError as exc:
            if 'FORK_MOE_RESOURCE' not in str(exc):
                raise
            st['_fork_native_eligible']=False
            st['_fork_moe_status']=st['_fork_fallback_kind']+': '+str(exc)[-300:]
            print('FORK_MOE_ROUTE',st['_fork_moe_status'],flush=True)
            return
        torch.cuda.synchronize()  # launch/execution faults remain fatal
    st['_fork_moe_status']='native repaired: cooperative+BF16, MFIN/DF/YBF/PDL off'
    print('FORK_MOE_ROUTE',st['_fork_moe_status'],flush=True)

def _moe(M=None, state=None):
    if state is None:
        return None  # suppress all old unconditional native background builds
    if not state.get('_fork_native_eligible',False):
        return _fork_moe_fallback(M,state)
    return _fork_native_moe(M,state)
'''


def patch_device(s, main, *, wgs_down_sites=None):
    # Public scope is intentionally restricted: the unreviewed non-WS/non-WGS
    # MAIN_FALLBACK_DEFINES must never become a silently selected alternative.
    guard='''
#if !OKC_WGS && !OKC_WS
#error FORK_MOE_UNPROVEN_FALLBACK_select_owned_or_Marlin_adapter
#endif
#if OKC_WGS
#undef OKC_MFIN
#define OKC_MFIN 0
#endif
'''
    s=once(s,'namespace okc {','namespace okc {\n'+guard)
    s=once(s,'__device__ __forceinline__ float probe_q(float v) {',
        '__device__ __forceinline__ float fork_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }\n'
        '__device__ __forceinline__ float probe_q(float v) {')
    # Gamma equals official baseline's BF16 effective weight, not FP32(1+raw).
    s,n1=wrap_balanced(s,'1.f + __bfloat162float(')
    s,n2=wrap_balanced(s,'1.f + __uint_as_float(')
    assert (n1,n2)==(14,2),(n1,n2)
    pattern=r'((?:const )?float gt = )([^;\n]*?), up = ([^;\n]*);'
    s,n=re.subn(pattern,lambda m:m[1]+'fork_bf16('+m[2]+'), up = fork_bf16('+m[3]+');',s)
    assert n==5,n
    pattern=r'(float v0f? = )([^;\n]*?) \* gsc \* (\w+), (v1f? = )([^;\n]*?) \* gsc \* (\w+);'
    s,n=re.subn(pattern,lambda m:m[1]+'fork_bf16('+m[2]+' * gsc) * '+m[3]+', '+m[4]+
                'fork_bf16('+m[5]+' * gsc) * '+m[6]+';',s)
    assert n==5,n
    # WGS gamma and global weight scale must be separated at the BF16 GEMM2 boundary.
    s=once(s,'(fork_bf16(1.f + __bfloat162float(__ldg(p.normw + n0)))) * gsc',
             'fork_bf16(1.f + __bfloat162float(__ldg(p.normw + n0)))')
    s=once(s,'(fork_bf16(1.f + __bfloat162float(__ldg(p.normw + n0 + 8)))) * gsc',
             'fork_bf16(1.f + __bfloat162float(__ldg(p.normw + n0 + 8)))')
    s,n=re.subn(r'd4\[([0-3h +]+)\] \* (w[01])',r'fork_bf16(d4[\1] * gsc) * \2',s)
    assert n == (wgs_down_sites if wgs_down_sites is not None else (8 if main else 2)), n
    if main:
        s=once(s,'#define OKC_YBF 1   // ybf: WGS ybuf in bf16',
                 '#define OKC_YBF 0   // fork: preserve FP32 gamma product after BF16 GEMM2 result')
    # Skinny slot reads are generic; every reader bridges generic->async before
    # the compute barrier permits tid0's TMA overwrite. No producer-only shortcut.
    old='''    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    CTA_SYNC();'''
    s=once(s,old,old.replace('    CTA_SYNC();',
        '    asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n    CTA_SYNC();'))
    if main:
        # WGS ring: each reader fences and arrives; the producer waits for NT,
        # never NW lane0 arrivals after un-ordered generic reads.
        s=once(s,'k < BW_E ? 1 : k < BX_F ? NW : k < BX_E ? 1 : NW',
                 'k < BW_E ? 1 : k < BX_F ? NT : k < BX_E ? 1 : NT')
        s=once(s,'} else if (lane == 0) {\n        mbar_arrive1(S.bars + BW_E + ws);',
                 '} else {\n        asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n        mbar_arrive1(S.bars + BW_E + ws);')
        s=once(s,'if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);',
                 'asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n            mbar_arrive1(S.bars + BX_E + prev);',n=2)
        # generic xA/xB stores -> global TMA reads, via release/acquire flags.
        s=once(s,'  CTA_SYNC();\n  if (tid == 0) {  // count this CTA\'s rows per job',
                 '  asm volatile("fence.proxy.async.global;\\n" ::: "memory");\n  CTA_SYNC();\n  if (tid == 0) {  // count this CTA\'s rows per job')
        s=once(s,'  if (pd.kind == 0) return;\n  const int tid = threadIdx.x,',
                 '  if (pd.kind == 0) return;\n  if constexpr (!PB) { asm volatile("fence.proxy.async.global;\\n" ::: "memory"); }\n  const int tid = threadIdx.x,')
        # Split-unit finishers create xB inside this flush, after its entry
        # fence. Their fresh stores need their OWN writer-side bridge too.
        s=once(s,'  if (!PB) {\n    asm volatile("bar.sync %0, 128;\\n"',
                 '  if (!PB) {\n    asm volatile("fence.proxy.async.global;\\n" ::: "memory");\n    asm volatile("bar.sync %0, 128;\\n"')
    # Counter publication accumulates all CTA writes through the preceding
    # CTA barrier, then acq_rel RMW chain, then the reader acquire + CTA barrier.
    s=once(s,'    const int old = atomicAdd(bc, 1);',
        '    int old;\n    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(bc), "r"(1) : "memory");')
    # No programmatic secondary launch: ordinary same-stream launch waits for
    # previous primaries to complete before the cooperative grid is admitted.
    a=s.index('  // large batches run the variant with the fat path inlined')
    z=s.index('\n}',a)
    s=s[:a]+COOPERATIVE+s[z:]
    return s


def patch_moe(source):
    """Patch exact pinned full-layer source. Fail closed on any source drift."""
    source=source.replace('\r\n','\n')
    assert hashlib.sha256(source.encode()).hexdigest()==PIN,'unrecognized layer source; apply this patch before other fork edits'
    tree=ast.parse(source); lines=source.splitlines(keepends=True); edits=[]; found=[]
    last_moe=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_moe'][-1]
    ma_prep=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_ma_prepare')
    for node,newname in ((last_moe,'_fork_native_moe'),(ma_prep,'_fork_native_ma_prepare')):
        text=ast.get_source_segment(source,node)
        text=text.replace('def '+node.name+'(', 'def '+newname+'(',1)
        if node is ma_prep:
            text=once(text,'_MA_SC6W','st["ma_sc6"]',n=2)
        else:
            text=once(text,'    ext = state["ma_ext"].get(key)',
                '    if state["ma_sc6"] and "-DOKC_WGS=1" in defines:\n'
                '        defines = tuple(defines) + ("-DOKC_SC6W=1",)\n'
                '    ext = state["ma_ext"].get(key)')
        edits.append((node.lineno-1,node.end_lineno,text+'\n'))
    builder=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_ma_ext')
    text=ast.get_source_segment(source,builder)
    a=text.index('    if _MA_SC6W and');z=text.index('    if DQ7',a)
    text=text[:a]+text[z:]
    edits.append((builder.lineno-1,builder.end_lineno,text+'\n'))
    prep=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare'
        and any(isinstance(g,ast.Global) and '_MA_SFX' in g.names for g in n.body))
    text=ast.get_source_segment(source,prep)
    a=text.index('    global _MA_SFX, _MA_SC6W');z=text.index('    st["ma_ext"] = {}',a)
    text=text[:a]+'''    st["ma_sfx"] = _MA_SFX_IF_PRMT if (_MA_PRMT_SCALES and _ma_scales_prmt_exact(weights["w13_sf"], weights["w2_sf"])) else ()
    st["ma_sc6"] = bool(SC6_WGS and st["ma_sfx"] and _wgs_on() and _sc6_ok(weights["w13_sf"], weights["w2_sf"]))
    print("FORK_MOE_SCALE",st["ma_sfx"],st["ma_sc6"],flush=True)
'''+text[z:]
    a=text.index('    if _MA_SC6W:');z=text.index('    _bfp_prepare(weights, st)',a)
    text=text[:a]+text[z:]  # no diagnostics can rebuild native after fallback
    for key in ('w13q','w13s','w2q','w2s'):
        # Exact pin has these four stores and no consumers, including forced picks:
        # _after_attention rejects any backend except agent/Marlin.
        nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Subscript)
               and isinstance(n.slice,ast.Constant) and n.slice.value==key]
        assert len(nodes)==1 and isinstance(nodes[0].ctx,ast.Store),key
        line=next(line for line in text.splitlines(keepends=True) if f'st["{key}"] =' in line)
        text=once(text,line,'')
    edits.append((prep.lineno-1,prep.end_lineno,text+'\n'))
    for n in tree.body:
        if not isinstance(n,ast.Assign): continue
        target=ast.unparse(n.targets[0])
        if target in ('MOE_CUDA_SRC',"_MA_ALT_SRCS['moe_helper0']"):
            value=ast.literal_eval(n.value)
            patched=patch_device(value,target=='MOE_CUDA_SRC')
            assert '"""' not in patched
            # Keep visible multiline CUDA so the coordinator's safe whole-line
            # comment stripper can meet the single-file size limit.
            edits.append((n.lineno-1,n.end_lineno,target+' = r"""'+patched+'"""\n')); found.append(target)
        elif target=="_MA_ALT_SRCS['moe_helper0df']":
            edits.append((n.lineno-1,n.end_lineno,"_MA_ALT_SRCS['moe_helper0df'] = _MA_ALT_SRCS['moe_helper0']  # fork: disable sentinel DF\n"))
    assert set(found)=={'MOE_CUDA_SRC',"_MA_ALT_SRCS['moe_helper0']"}
    for a,z,text in sorted(edits,reverse=True): lines[a:z]=[text]
    out=''.join(lines)
    out=once(out,'MOE_HAS_PDL = True','MOE_HAS_PDL = False')
    out=once(out,'MFAB_LAYER = True','MFAB_LAYER = False')
    out+='\n'+HOST_BRIDGE
    ast.parse(out)
    return out


if __name__=='__main__':
    Path(sys.argv[2]).write_text(patch_moe(Path(sys.argv[1]).read_text(encoding='utf-8')),encoding='utf-8')
