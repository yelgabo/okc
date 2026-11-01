"""Owned complete-output q8/B1/sliding clusters; no donor CUDA and no production edits."""
from pathlib import Path
import ast
from attn_cluster_merge_patch import once

FINALIZE=r'''
        if constexpr (HC) {
            cp_wait<0>(); csync();
            // All raw-stage readers and copies are retired; this CTA owns one KV head.
            // Fixed-frame partials: local row = token*8 + query-head-within-KV-head.
            // Generic DSM pushes require every destination to have retired raw-stage readers.
            cooperative_groups::this_cluster().sync();
        }
'''
MERGE=r'''
        if constexpr (HC) {
            __syncthreads();
            auto cluster=cooperative_groups::this_cluster();
            cluster.sync(); // all sixteen complete-context partitions published/alive
            const int lr=static_cast<int>(cluster.block_rank())*4+(tid>>3);
            const bool active=(tid<32);
            const int t8=tid&7;
            float acc[8]={};float sum=0.f;
            if(active) {
#pragma unroll 1
                for(int peer=0;peer<16;++peer) {
                    const float* row=reinterpret_cast<float*>(smem)+(peer*4+(lr&3))*68;
                    const float4 a=*reinterpret_cast<const float4*>(row+t8*8);
                    const float4 b4=*reinterpret_cast<const float4*>(row+t8*8+4);
                    acc[0]+=a.x;acc[1]+=a.y;acc[2]+=a.z;acc[3]+=a.w;
                    acc[4]+=b4.x;acc[5]+=b4.y;acc[6]+=b4.z;acc[7]+=b4.w;
                    if(t8==0)sum+=row[64];
                }
            }
            sum=__shfl_sync(0xffffffffu,sum,(tid&31)&~7);
            const float inv=sum>0.f?__fdividef(v_scale,sum):0.f;
            float ss=0.f;
#pragma unroll
            for(int e=0;e<8;++e){acc[e]=__bfloat162float(__float2bfloat16_rn(acc[e]*inv));ss+=acc[e]*acc[e];}
            const float rn=rsqrtf(oct_sum(ss)*(1.f/64.f)+EPS);
            if(active) {
                const int token=lr/8, head=hc_head*8+(lr&7);
                const uint4 raw=__ldg(reinterpret_cast<const uint4*>(p.qkv+token*QKV_STRIDE+GATE_OFF+head*64+t8*8));
                const uint32_t gw[4]={raw.x,raw.y,raw.z,raw.w};uint32_t ow[4];
#pragma unroll
                for(int e=0;e<4;++e){
                    const float2 gate=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gw+e));
                    const __nv_bfloat162 value=__float22bfloat162_rn(make_float2(acc[2*e]*rn*sigmoid(gate.x),acc[2*e+1]*rn*sigmoid(gate.y)));
                    ow[e]=*reinterpret_cast<const uint32_t*>(&value);
                }
                *reinterpret_cast<uint4*>(p.out+token*HQ*64+head*64+t8*8)=make_uint4(ow[0],ow[1],ow[2],ow[3]);
            }
            // All DSM writes retired at the publication barrier. Only local reads/output stores
            // follow; no surviving CTA dereferences a peer's shared storage after this point.
        }
'''
LAUNCH=r'''
        if constexpr (ONE_HEADCLUSTER && L==8 && MODE==3) {
            TORCH_CHECK(gx==32 && gy==1 && p.window==2048 && p.page_size==1 && p.nc==1,"unsupported head-cluster geometry");
            cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(32);cfg.blockDim=dim3(NT);cfg.dynamicSmemBytes=BYTES;cfg.stream=stream;
            cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;
            attr.val.clusterDim.x=16;attr.val.clusterDim.y=1;attr.val.clusterDim.z=1;cfg.attrs=&attr;cfg.numAttrs=1;
            static bool checked=false;
            if(!checked){int active=0;TORCH_CHECK(cudaOccupancyMaxActiveClusters(&active,attn_one_kernel<L,MODE>,&cfg)==cudaSuccess && active>=2,"two head clusters must fit");checked=true;}
            const cudaError_t err=cudaLaunchKernelEx(&cfg,attn_one_kernel<L,MODE>,p);
            return err!=cudaSuccess?err:cudaGetLastError();
        }
'''


def patch_headcluster(s):
    s=once(s,'#include <torch/extension.h>','#include <torch/extension.h>\n#include <cooperative_groups.h>\n#ifndef ONE_HEADCLUSTER\n#define ONE_HEADCLUSTER 0\n#endif')
    s=once(s,'    return (L == 8 ? 256 : 128) + ((mode & M_PROD) ? 128 : 0);','    return (ONE_HEADCLUSTER && L==8 && mode==3) ? 128 : (L == 8 ? 256 : 128) + ((mode & M_PROD) ? 128 : 0);')
    s=once(s,'    constexpr int NC = L == 8 ? 256 : 128;','    constexpr bool HC=ONE_HEADCLUSTER && L==8 && MODE==3;\n    constexpr int NC = HC ? 128 : (L == 8 ? 256 : 128);')
    s=once(s,'    const int split = blockIdx.x;','    const int split = blockIdx.x;\n    const int hc_head=split>>4,hc_rank=split&15;\n    static_assert(L!=8 || S::Q>=64*68*4,"dead raw-stage alias capacity");')
    anchor='        nact = COOP ? min(p.n_splits, (ntl + chunk - 1) / chunk) : 0;'
    s=once(s,anchor,anchor+'\n        if constexpr(HC){lo=start+(ntl*hc_rank/16)*BN;hi=min(end,start+(ntl*(hc_rank+1)/16)*BN);ntiles=max(0,(hi-lo+BN-1)/BN);nact=16;}')
    s=once(s,'        if (!SPEC && lo >= end) {','        if (!SPEC && lo >= end && !HC) {')
    s=once(s,'    auto issue_kv = [&](int tile, int st, bool patch) {','    auto issue_kv = [&](int tile, int st, bool patch) {\n        if constexpr(HC){if((ch>>2)!=hc_head)return;}')
    raw='''                    cp_async16(smem_base + S::QST + row * 128 + cc * 16,
                               qbase + (row >> 4) * QKV_STRIDE + (row & 15) * D + cc * 8, 16);'''
    assert s.count(raw)==2
    s=s.replace(raw,'                    if(!HC || ((row&15)>>3)==hc_head) '+raw.strip(),2)
    old='''            if (has_new && qtid < L * 32) {
                const int rr = qtid >> 3;
                const int cc = qtid & 7;'''
    new='''            if (has_new) for(int ni=qtid;ni<L*32;ni+=NC) {
                const int rr = ni >> 3;
                const int cc = ni & 7;'''
    s=once(s,old,new)
    s=once(s,'''            if (tid < L * 32) {
                const float2 sc =
                    fast_sincos(static_cast<float>(seq + (tid >> 5)) * invf[tid & 31]);
                rope_sin[tid] = sc.x;
                rope_cos[tid] = sc.y;''','''            for(int ri=tid;ri<L*32;ri+=NC) {
                const float2 sc = fast_sincos(static_cast<float>(seq+(ri>>5))*invf[ri&31]);
                rope_sin[ri]=sc.x;rope_cos[ri]=sc.y;''')
    s=once(s,'            const int head = row & 15;\n            float x[8];','            const int head = row & 15;\n            if constexpr(HC){if((head>>3)!=hc_head)return;}\n            float x[8];')
    s=once(s,'''        if (hi > seq && tid < L * 32) {
            const int rr = tid >> 3;
            const int t8 = tid & 7;''','''        if(hi>seq) for(int ni=tid;ni<L*32;ni+=NC) {
            const int rr=ni>>3;
            const int t8=ni&7;''')
    s=once(s,'            if (pos >= lo && pos < hi) {\n                const int64_t loc =','            if (pos >= lo && pos < hi && (!HC || hh==hc_head)) {\n                const int64_t loc =')
    s=once(s,'            const int hk = warp >> 2;','            const int hk = HC ? hc_head : warp >> 2;')
    old='''#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const float f = ex2(fminf(off[hh], 126.f));'''
    s=once(s,old,FINALIZE+old)
    old='''                float* dst = p.ws_acc + static_cast<size_t>(row) * D + c * 4;
#pragma unroll
                for (int pp = 0; pp < 4; ++pp) {
                    red_add_v4(dst + 16 * pp, o[8 * pp + 2 * hh] * f, o[8 * pp + 4 + 2 * hh] * f,
                               o[8 * pp + 1 + 2 * hh] * f, o[8 * pp + 5 + 2 * hh] * f);
                }
                if (c == 0) {
                    red_add_f32(p.ws_l + row, o[32 + 2 * hh] * f);
                }'''
    new='''                float* dst=p.ws_acc+static_cast<size_t>(row)*D+c*4;
                float* local=reinterpret_cast<float*>(smem)+((2*wq+hh)*8+g)*68;
                if constexpr(HC){
                    const int local_row=(2*wq+hh)*8+g;
                    local=cooperative_groups::this_cluster().map_shared_rank(reinterpret_cast<float*>(smem)+(hc_rank*4+(local_row&3))*68,local_row>>2);
                }
#pragma unroll
                for(int pp=0;pp<4;++pp){
                    const float4 v=make_float4(o[8*pp+2*hh]*f,o[8*pp+4+2*hh]*f,o[8*pp+1+2*hh]*f,o[8*pp+5+2*hh]*f);
                    if constexpr(HC)*reinterpret_cast<float4*>(local+c*4+16*pp)=v;
                    else red_add_v4(dst+16*pp,v.x,v.y,v.z,v.w);
                }
                if(c==0){if constexpr(HC)local[64]=o[32+2*hh]*f;else red_add_f32(p.ws_l+row,o[32+2*hh]*f);}'''
    s=once(s,old,new)
    s=once(s,'        stamp(17);\n        return true;',MERGE+'        stamp(17);\n        return true;')
    s=once(s,'''        // Diagnostic exits after the merge reds (11) and after the ticket or grid barrier (12).''','''        if constexpr(HC)return; // complete disjoint outputs; no global partials/counters/finalize
        // Diagnostic exits after the merge reds (11) and after the ticket or grid barrier (12).''')
    s=once(s,'            cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES);','            cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES);\n            if(ONE_HEADCLUSTER && L==8 && MODE==3)TORCH_CHECK(cudaFuncSetAttribute(kern,cudaFuncAttributeNonPortableClusterSizeAllowed,1)==cudaSuccess,"cluster opt-in");')
    s=once(s,'        prepare();\n        if (!coop) {','        prepare();\n'+LAUNCH+'        if (!coop) {')
    return s


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('control',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    tree=ast.parse(a.control.read_text(encoding='utf-8'))
    src=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_ONE_SRC' for t in n.targets))
    a.output.write_text(patch_headcluster(src),encoding='utf-8')
