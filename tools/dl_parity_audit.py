"""Per-kernel parity audit: decode-layer profile rows against the best judged component kernels.

Usage: python3 tools/dl_parity_audit.py [table|rank|glue] LAYER_TXT [PDL_TXT]
Component receipts: BF16 9eba96894ee1402e, attention 8914751e3b9f468b, MoE public draw 2821fdb184d740b1.
"""
import json
import re
import sys

BF16 = 'muse_bf16_proj/profiles/scored.9eba96894ee1402e.json'
ATTN = 'muse_attention_decode/profiles/scored.8914751e3b9f468b.json'
MOE = 'muse_moe_w4a16/profiles/scored-draw2.2821fdb184d740b1.txt'
LOADS = ['decode 1', 'verify 1', 'decode 16', 'decode 32', 'verify 16', 'verify 32']
W = {'sliding': 46, 'full': 16}
TOK = {'decode 1': 1, 'verify 1': 8, 'decode 16': 16, 'decode 32': 32, 'verify 16': 128, 'verify 32': 256}
# sk2 template prefixes <mt, tokr, NT, NWR, WSK, UPW, nstage> per token count, read off the 607eb52 rows
SK2 = {1: {(1, 8, 3, 1, 4, 2, 6): 'qkv_gate', (1, 8, 4, 1, 4, 1, 4): 'o', (1, 8, 1, 1, 8, 2, 4): 'unpooler'},
       8: {(1, 8, 3, 1, 4, 2, 5): 'qkv_gate', (1, 8, 4, 1, 4, 1, 4): 'o', (1, 8, 1, 1, 8, 2, 4): 'unpooler'},
       16: {(1, 16, 4, 1, 8, 1, 5): 'pooler+router', (1, 16, 4, 1, 4, 1, 5): 'unpooler',
            (1, 16, 3, 1, 4, 2, 5): 'qkv_gate', (1, 16, 4, 1, 4, 1, 4): 'o'},
       32: {(2, 32, 4, 1, 8, 1, 3): 'pooler+router', (2, 32, 4, 2, 2, 2, 3): 'unpooler',
            (2, 32, 3, 1, 4, 2, 4): 'qkv_gate', (2, 32, 2, 2, 2, 2, 3): 'o'}}
# cuBLASLt tiles at 128 / 256 tokens, mapped by the component board's picks for the same shapes
NVJET = {128: {'tst_128x64_64x8_1x2_h_bz_NNT': 'o+unpooler', 'tss_80x64': 'pooler+router', 'tst_48x64': 'qkv_gate'},
         256: {'tss_144x64': 'pooler+router', 'coopA_NNT': 'unpooler', 'tst_72x64': 'qkv_gate', 'coopA_NNN': 'o'}}


def parse_layer(path):
    txt = open(path).read()
    ends, prof = {}, {}
    for m in re.finditer(r'layer=(\w+) load=(\w+ \d+)[^\n]*?\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x', txt):
        ends[(m.group(2), m.group(1))] = float(m.group(4))
    for blk in txt.split('\nprofile ')[1:]:
        m = re.search(r'layer=(\w+) load=(\w+ \d+)', blk.splitlines()[0])
        rows = []
        for line in blk.splitlines()[2:]:
            mm = re.match(r'(.*?)\s+([\d.]+) µs\s+[\d.]+%\s+(\d+)', line)
            if mm:
                rows.append((mm.group(1).strip(), float(mm.group(2))))
        prof[(m.group(2), m.group(1))] = rows
    return ends, prof


def profile_totals(path):
    out = {}
    for r in json.load(open(path))['result']['records']:
        if r['event'] == 'profile':
            out[(r['case']['load'], r['case']['layer'])] = float(re.search(r'total\s+([\d.]+)', r['report']).group(1))
    return out


def moe_totals(path):
    out = {}
    for blk in open(path).read().split('\nprofile ')[1:]:
        out[re.search(r'load=(\w+ \d+)', blk.splitlines()[0]).group(1)] = float(re.search(r'total\s+([\d.]+)', blk).group(1))
    return out


def role(name, load):
    T = TOK[load]
    if 'moe_fused' in name:
        return 'moe'
    if 'attn_kernel' in name:
        return 'attention'
    if 'sk2_kernel_gle' in name:
        return 'pooler+router'
    if 'glue_cluster_boundary<4, 32, true' in name:
        return 'glue: finish'
    if 'glue_cluster_boundary' in name:
        return 'glue: boundary 1'
    if name.startswith(('_sl_tile_route', '_graft_route')):
        return 'glue: route'
    if name.startswith('_sl_cluster_finish'):
        return 'glue: finish'
    if name.startswith('_graft_row_boundary'):
        return 'glue: row boundaries'
    if T <= 32:
        geo = tuple(map(int, re.search(r'<(\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+)', name).groups()))
        return SK2[T][geo]
    for k, v in NVJET[T].items():
        if k in name:
            return v
    return name


def main():
    mode, lpath = sys.argv[1], sys.argv[2]
    ppath = sys.argv[3] if len(sys.argv) > 3 else None
    E, PR = parse_layer(lpath)
    PR2 = parse_layer(ppath)[1] if ppath else {}
    B, A, M = profile_totals(BF16), profile_totals(ATTN), moe_totals(MOE)

    def comp(rl, load, layer):
        if rl == 'moe':
            return M[load]
        if rl == 'attention':
            return A[(load, layer)]
        if rl == 'pooler+router':
            return B[(load, 'pooler')] + B[(load, 'router')]
        if rl == 'o+unpooler':
            return B[(load, 'o')] + B[(load, 'unpooler')]
        if rl in ('qkv_gate', 'o', 'unpooler'):
            return B[(load, rl)]
        return None

    rows = []
    for load in LOADS:
        for layer in ('sliding', 'full'):
            key = (load, layer)
            val = W[layer] / 372 / E[key] * 100
            agg, agg2 = {}, {}
            for n, t in PR[key]:
                agg[role(n, load)] = agg.get(role(n, load), 0) + t
            for n, t in PR2.get(key, []):
                agg2[role(n, load)] = agg2.get(role(n, load), 0) + t
            for rl, t in agg.items():
                cu = comp(rl, load, layer)
                gap = None if cu is None else t - cu
                rows.append((load, layer, rl, t, agg2.get(rl), cu, gap, val, None if gap is None else gap * val))
    f = lambda v, d=1: '-' if v is None else f'{v:.{d}f}'
    if mode == 'table':
        print('| cell | kernel | layer us (607eb52) | layer us (pdl8 8f6acdd) | component us | gap us | % score per us | weighted gap, % score |')
        print('|---|---|---:|---:|---:|---:|---:|---:|')
        for r in rows:
            if r[2].startswith('glue'):
                continue
            print(f'| {r[0]} {r[1]} | {r[2]} | {f(r[3])} | {f(r[4])} | {f(r[5])} | {f(r[6])} | {f(r[7], 3)} | {f(r[8], 3)} |')
    elif mode == 'rank':
        agg = {}
        for r in rows:
            if r[8] is None or r[2] == 'moe':
                continue
            a = agg.setdefault((r[2], r[0]), [0.0, 0.0])
            a[0] += r[6]
            a[1] += r[8]
        for k, v in sorted(agg.items(), key=lambda kv: -kv[1][1])[:12]:
            print(f'{k[0]} at {k[1]}: gap {v[0]:.1f} us summed over both cells, {v[1]:.3f}% of score')
    elif mode == 'glue':
        print('| cell | boundary 1 | route | finish | row boundaries | glue sum (607eb52) | glue sum (pdl8) | % score per us | glue, % score |')
        print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
        for load in LOADS:
            for layer in ('sliding', 'full'):
                key = (load, layer)
                val = W[layer] / 372 / E[key] * 100
                g = {}
                for n, t in PR[key]:
                    if role(n, load).startswith('glue'):
                        g[role(n, load)] = g.get(role(n, load), 0) + t
                g2 = sum(t for n, t in PR2.get(key, []) if role(n, load).startswith('glue'))
                s = sum(g.values())
                print(f"| {load} {layer} | {f(g.get('glue: boundary 1'))} | {f(g.get('glue: route'))} | "
                      f"{f(g.get('glue: finish'))} | {f(g.get('glue: row boundaries'))} | {f(s)} | {f(g2 or None)} | "
                      f"{val:.3f} | {s * val:.3f} |")


if __name__ == '__main__':
    main()
