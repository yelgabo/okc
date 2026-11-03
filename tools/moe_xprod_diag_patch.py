"""Add producer-only wait attribution to a mode56 diagnostic bench, never production.

Generate with moe_make_bench_ab.py --nopub=1 --base=51 --sizes=128,256
--routings=layer,std --seeds=211,212 xp=56; then patch IN.py OUT.py.
Uses the unused fat-path inter scratch gap below kSkOff; stage stamps start at kSkOff.
"""
from pathlib import Path
import sys
import re


def patch(text):
    def change(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError(f"expected unique anchor: {old[:90]!r}")
        text = text.replace(old, new)

    # Repurpose timing-only56 as51 plus producer counters, removing its old
    # consumer/light/gather stamps. No production mode's predicates change.
    assert text.count('|| mode == 56') == 3
    text = text.replace('|| mode == 56', '')
    text = re.sub(r'^_AB_LIGHT = .*$', '_AB_LIGHT = {}', text, flags=re.M)

    change('    Ring rg = {0, 0, 0, 0};\n    long long txe = 0;',
           '    Ring rg = {0, 0, 0, 0};\n'
           '    // Diagnostic only: phase A/B x small/large x ready/empty/stages/events.\n'
           '    long long pd[16] = {};\n'
           '    constexpr bool XP = MODE == 56;\n'
           '    long long txe = 0;')
    change('            const unsigned bytes = ntt * 4096;\n            const unsigned char* base',
           '            const unsigned bytes = ntt * 4096;\n'
           '            const int pi = ph * 8 + (ntt > 4 ? 4 : 0);\n'
           '            const unsigned char* base')
    change('                spin_ge(p.xready + j, S.job_n[j], p.ctr, 32);\n                if (pa_diag(MODE))',
           '                const long long pr = XP ? clock64() : 0;\n'
           '                spin_ge(p.xready + j, S.job_n[j], p.ctr, 32);\n'
           '                if constexpr (XP) { pd[pi] += clock64() - pr; pd[pi+3]++; }\n'
           '                if (pa_diag(MODE))')
    change('                if (rg.xlap > 0) {\n                    pxe += mbar_wait_m<MODE, 0>',
           '                const long long pe = XP ? clock64() : 0;\n'
           '                if (rg.xlap > 0) {\n                    pxe += mbar_wait_m<MODE, 0>')
    change('                if (kDiag) {\n                    const long long c1 = clock64();\n                    txe',
           '                if constexpr (XP) { pd[pi+1] += clock64() - pe; pd[pi+2]++; }\n'
           '                if (kDiag) {\n                    const long long c1 = clock64();\n                    txe')
    # Mode56's static work is phase A only; phase B uses the dynamic claimed path.
    change('            const unsigned bytes = S.job_ntt[j] * 4096;\n            const unsigned char* base',
           '            const unsigned bytes = S.job_ntt[j] * 4096;\n'
           '            const int pi = (ca ? 0 : 8) + (S.job_ntt[j] > 4 ? 4 : 0);\n'
           '            const unsigned char* base')
    change('                long long c0 = is_diag(MODE) ? clock64() : 0;\n                if (rg.xlap > 0)',
           '                long long c0 = is_diag(MODE) ? clock64() : 0;\n'
           '                const long long pe = XP ? clock64() : 0;\n'
           '                if (rg.xlap > 0)')
    change('                if (is_diag(MODE)) {\n                    const long long c1 = clock64();\n                    txe',
           '                if constexpr (XP) { pd[pi+1] += clock64() - pe; pd[pi+2]++; }\n'
           '                const long long pr = XP ? clock64() : 0;\n'
           '                if (is_diag(MODE)) {\n                    const long long c1 = clock64();\n                    txe')
    change('                if (is_diag(MODE)) {\n                    tda += clock64() - c0;\n                }',
           '                if constexpr (XP) { pd[pi] += clock64() - pr; pd[pi+3]++; }\n'
           '                if (is_diag(MODE)) {\n                    tda += clock64() - c0;\n                }')
    change('    if (is_diag(MODE)) {\n        S.dg[17][0] = txe;',
           '    if constexpr (XP) {\n'
           '        long long* out = reinterpret_cast<long long*>(p.inter) + (1ll << 20) - 16384 + cta * 64;\n'
           '#pragma unroll\n'
           '        for (int k = 0; k < 16; ++k) out[k] = pd[k];\n'
           '    }\n'
           '    if (is_diag(MODE)) {\n        S.dg[17][0] = txe;')
    change('                if _AB_WORK:\n                    works.append',
           '                _run(d, state, timed["xp"])\n'
           '                torch.cuda.synchronize()\n'
           '                raw = state["scratch"]["inter"].reshape(-1).view(torch.int64)\n'
           '                assert G * 64 <= 16384 and raw.numel() > (1 << 20) + G * 64\n'
           '                rows = raw[(1 << 20)-16384:(1 << 20)-16384+G*64].view(G,64)[:,:16].double().cpu().reshape(G,2,2,4)\n'
           '                for ph in range(2):\n'
           '                    for size in range(2):\n'
           '                        col = rows[:,ph,size,:]\n'
           '                        print(f"ABXP {tag} seed={s} phase={ph} large={size} sums={col.sum(0).tolist()} max={col.max(0).values.tolist()}", flush=True)\n'
           '                if _AB_WORK:\n                    works.append')
    return text


if __name__ == '__main__':
    Path(sys.argv[2]).write_text(patch(Path(sys.argv[1]).read_text()))
