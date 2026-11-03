"""One mechanism: single-warp stable position map for skinny M<=16."""
import ast
from pathlib import Path
import sys
import moe_skinny_testwait_check as prior

HOOK = r'''
// Only warp0 owns wc[0] here. Every lane participates in every full-mask collective,
// including nonlocal/invalid entries. Two warp barriers separate old-count reads,
// leader-only updates, and next-chunk reads. The final CTA barrier publishes pos/ent.
// CUDA12.8 Programming Guide 7.6: syncwarp orders participating-lane memory;
// syncthreads publishes shared writes to all CTA threads. No async-proxy path changes.
__device__ __forceinline__ void position_singlewarp(Smem& S, int E) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    if ((tid >> 5) == 0) {
        S.wc[0][lane] = 0;
        __syncwarp();
        for (int base = 0; base < E; base += 32) {
            const int i = base + lane;
            const int eid = i < E ? S.eid[i] : -1;
            const unsigned peers = __match_any_sync(0xffffffffu, eid);
            const int run = eid >= 0 ? S.wc[0][eid] : 0;
            if (eid >= 0) {
                const int q = S.off[eid] + run + __popc(peers & ((1u << lane) - 1u));
                S.pos[i] = static_cast<short>(q);
                S.ent[q] = static_cast<short>(i);
            } else if (i < E) {
                S.pos[i] = -1;
            }
            __syncwarp();
            if (eid >= 0 && lane == __ffs(peers) - 1) {
                S.wc[0][eid] = run + __popc(peers);
            }
            __syncwarp();
        }
    }
    __syncthreads();
}
'''
OLD = '    position_pass<kNWarps>(S.eid, S.pos, S.ent, S.wc, S.off, E);'
NEW = '''    if constexpr ((kMode & 512) != 0) {
        if (p.M <= 16) {
            position_singlewarp(S, E);
        } else {
            position_pass<kNWarps>(S.eid, S.pos, S.ent, S.wc, S.off, E);
        }
    } else {
        position_pass<kNWarps>(S.eid, S.pos, S.ent, S.wc, S.off, E);
    }'''


def build(src):
    # Reuse the already executed four-arm formatter/gate, not the prior CUDA mechanism.
    out = prior.build(src)
    out = out.replace(prior.HOOK+'\n', '').replace(
        'full_slot_ready(&S.full[slot], (q / kNS) & 1)',
        'mbar_try(&S.full[slot], (q / kNS) & 1)')
    out = out.replace('sk_8_3_128', 'sk_8_3_512').replace('sk_8_3_384', 'sk_8_3_768')
    out = out.replace('(8, 3, 128)', '(8, 3, 512)').replace('(8, 3, 384)', '(8, 3, 768)')
    out = out.replace('{128,a.numRegs', '{512,a.numRegs').replace('{384,a.numRegs', '{768,a.numRegs')
    out = out.replace('[0,256,128,384]', '[0,256,512,768]')
    out = out.replace('("test",128),("test2",384)', '("test",512),("test2",768)')
    out = out.replace(', (8, 3, 2), (8, 3, 130)', '')
    # No optional wait-counter experiment: finish with the complete core evidence.
    a = out.index('    # Fully counted corrected wait windows')
    out = out[:a] + '    return state\n'
    # Remove unused decoder definitions; no snapshot/error-scope machinery is needed.
    a = out.index('def _tw_snapshot_wait_counters(')
    z = out.index('_tw_original_prepare=prepare', a)
    out = out[:a]+out[z:]
    out = out.replace('TWCORE', 'PWCORE').replace('TWCASE', 'PWCASE').replace('TWRESOURCE','PWRESOURCE')
    out = out.replace('TWCHECK', 'PWCHECK').replace('TWSCORE', 'PWSCORE')
    out = out.replace('Testwait comparison', 'Position comparison').replace('testwait ', 'position ')
    out = out.replace('(321,322)', '(441,442)')  # fresh fixed heldout inputs, no seed search
    anchor='__device__ __forceinline__ void compute_loop('
    assert out.count(anchor)==1 and out.count(OLD)==1
    out = out.replace(anchor,HOOK+'\n'+anchor).replace(OLD,NEW)
    ast.parse(out)
    return out


def model(eids):
    """CPU literal chunk/peer/count model, independent stable-sort oracle in tests."""
    counts=[eids.count(e) for e in range(32)]
    off=[sum(counts[:e]) for e in range(32)]
    prior_counts=[0]*32
    pos=[-1]*len(eids); ent=[None]*sum(counts)
    for base in range(0,len(eids),32):
        chunk=(eids[base:base+32]+[-1]*32)[:32]
        for lane,e in enumerate(chunk):
            if e >= 0:
                q=off[e]+prior_counts[e]+chunk[:lane].count(e)
                pos[base+lane]=q; ent[q]=base+lane
        for e in range(32):
            prior_counts[e]+=chunk.count(e)
    return pos,ent


if __name__ == '__main__':
    Path(sys.argv[2]).write_text(build(Path(sys.argv[1]).read_text()))
