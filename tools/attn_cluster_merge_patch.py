"""Owned CHECK-ONLY mode3 hierarchical DSM merge; production remains untouched."""
import ast
from pathlib import Path


def once(s,a,b):
    assert s.count(a)==1,a[:90]
    return s.replace(a,b,1)


def patch_cuda(s):
    s=once(s,'#include <torch/extension.h>','#include <torch/extension.h>\n#include <cooperative_groups.h>\n#ifndef ONE_CLUSTER_MERGE\n#define ONE_CLUSTER_MERGE 0\n#endif')
    s=once(s,'    const int split = blockIdx.x;','    const int split = blockIdx.x;\n    const bool clmerge = ONE_CLUSTER_MERGE && L == 8 && MODE == 3 && gridDim.x == 32 && gridDim.y == 1 && p.window == 2048 && p.nc == 1;\n    static_assert(L != 8 || S::Q >= 128 * 68 * 4, "dead raw KV alias capacity");')
    anchor='        nact = COOP ? min(p.n_splits, (ntl + chunk - 1) / chunk) : 0;'
    s=once(s,anchor,anchor+'''
        if (clmerge) {
            lo = start + ((ntl * split) / 32) * BN;
            hi = min(end, start + ((ntl * (split + 1)) / 32) * BN);
            ntiles = max(0, (hi - lo + BN - 1) / BN);
            nact = 32; // empty partitions still join cluster and grid collectives
        }''')
    s=once(s,'        if (!SPEC && lo >= end) {','        if (!SPEC && lo >= end && !clmerge) {')
    anchor='''            for (int hh = 0; hh < 2; ++hh) {
                const float f = ex2(fminf(off[hh], 126.f));
                const int row = ws_row(b, p.nc, copy, L, 2 * wq + hh, hk * 8 + g);'''
    # The two shared raw matrices are dead only after all CTA readers and pending copies finish.
    prefix='''            if (clmerge) { cp_wait<0>(); csync(); }
#pragma unroll
'''
    s=once(s,anchor,prefix+anchor)
    old='''                float* dst = p.ws_acc + static_cast<size_t>(row) * D + c * 4;
#pragma unroll
                for (int pp = 0; pp < 4; ++pp) {
                    red_add_v4(dst + 16 * pp, o[8 * pp + 2 * hh] * f, o[8 * pp + 4 + 2 * hh] * f,
                               o[8 * pp + 1 + 2 * hh] * f, o[8 * pp + 5 + 2 * hh] * f);
                }
                if (c == 0) {
                    red_add_f32(p.ws_l + row, o[32 + 2 * hh] * f);
                }'''
    new='''                float* local = reinterpret_cast<float*>(smem) + (row % ROWS) * 68;
                float* dst = p.ws_acc + static_cast<size_t>(row) * D + c * 4;
#pragma unroll
                for (int pp = 0; pp < 4; ++pp) {
                    const float4 v = make_float4(o[8*pp+2*hh]*f,o[8*pp+4+2*hh]*f,o[8*pp+1+2*hh]*f,o[8*pp+5+2*hh]*f);
                    if (clmerge) *reinterpret_cast<float4*>(local+c*4+16*pp)=v;
                    else red_add_v4(dst+16*pp,v.x,v.y,v.z,v.w);
                }
                if (c == 0) {
                    if (clmerge) local[64]=o[32+2*hh]*f;
                    else red_add_f32(p.ws_l+row,o[32+2*hh]*f);
                }'''
    s=once(s,old,new)
    anchor='''        stamp(17);
        return true;'''
    s=once(s,anchor,'''        if (clmerge) {
            __syncthreads();
            auto cluster = cooperative_groups::this_cluster();
            cluster.sync(); // all peer shared stores visible and every peer alive
            if (cluster.block_rank() == 0) {
                float* base = reinterpret_cast<float*>(smem);
                for (int e=tid; e<ROWS*16; e+=NC) {
                    const int row=e/16, d=(e%16)*4;
                    float4 sum=make_float4(0,0,0,0);
#pragma unroll 1
                    for (int peer=0;peer<16;++peer) {
                        const float4 v=*reinterpret_cast<float4*>(cluster.map_shared_rank(base+row*68+d,peer));
                        sum.x+=v.x;sum.y+=v.y;sum.z+=v.z;sum.w+=v.w;
                    }
                    red_add_v4(p.ws_acc+row*64+d,sum.x,sum.y,sum.z,sum.w);
                }
                if (tid<ROWS) {
                    float sum=0;
#pragma unroll 1
                    for(int peer=0;peer<16;++peer) sum+=*cluster.map_shared_rank(base+tid*68+64,peer);
                    red_add_f32(p.ws_l+tid,sum);
                }
            }
            cluster.sync(); // all remote reads complete before any block exits/reuses storage
        }
'''+anchor)
    s=once(s,'        cfg.stream = stream;\n        cudaLaunchAttribute attr[1];', '        cfg.stream = stream;\n        cudaLaunchAttribute attr[2];')
    anchor='''        cfg.numAttrs = 1;
        const cudaError_t err = cudaLaunchKernelEx(&cfg, attn_one_kernel<L, MODE>, p);'''
    s=once(s,anchor,'''        cfg.numAttrs = 1;
        if (ONE_CLUSTER_MERGE && L==8 && MODE==3 && gx==32 && gy==1 && p.window==2048 && p.nc==1) {
            attr[1].id=cudaLaunchAttributeClusterDimension;
            attr[1].val.clusterDim.x=16; attr[1].val.clusterDim.y=1; attr[1].val.clusterDim.z=1;
            cfg.numAttrs=2;
            static bool cluster_checked=false;
            if (!cluster_checked) {
                int active_clusters=0;
                TORCH_CHECK(cudaOccupancyMaxActiveClusters(&active_clusters,attn_one_kernel<L,MODE>,&cfg)==cudaSuccess && active_clusters>=2,
                            "owned cluster cooperative residency requires both clusters");
                cluster_checked=true;
            }
        }
        const cudaError_t err = cudaLaunchKernelEx(&cfg, attn_one_kernel<L, MODE>, p);''')
    s=once(s, '#pragma unroll\n            if (clmerge) { cp_wait<0>(); csync(); }', '            if (clmerge) { cp_wait<0>(); csync(); }')
    s=once(s, '            cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES);', '            cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES);\n            if (ONE_CLUSTER_MERGE && L==8 && MODE==3) TORCH_CHECK(cudaFuncSetAttribute(kern,cudaFuncAttributeNonPortableClusterSizeAllowed,1)==cudaSuccess, "cluster opt-in failed");')
    s=once(s, 'ep, hi - lo,\n                             n_keys, tid, p.dbg);', 'ep, clmerge ? 1 : hi - lo,\n                             clmerge ? 32 : n_keys, tid, p.dbg);')
    return s


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('control',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    tree=ast.parse(a.control.read_text(encoding='utf-8'))
    src=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_ONE_SRC' for t in n.targets))
    a.output.write_text(patch_cuda(src),encoding='utf-8')
