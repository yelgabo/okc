"""CHECK ONLY: strict fresh-output error distribution for four owned modules.

0a/0b and2a/2b are independent CUDA builds and workspace dictionaries. Source
default and judged path stay0. The unchanged1e-4 global relativeL2 comparator
records all six null/cross pairs on BOTH shapes before its final assertion.
"""
from pathlib import Path
import sys
import attn_make_bench_one as base

ROOT = Path(__file__).resolve().parent.parent

BODY = r'''
import types as _dn_types
import triton as _dn_triton
import triton.language as _dt
_dn_mods = {"0a": None}
for _tag, _src in _DN_SOURCES.items():
    _mod = _dn_types.ModuleType("fresh_diag_" + _tag)
    exec(compile(_src, "fresh_diag_" + _tag + ".py", "exec"), _mod.__dict__)
    _dn_mods[_tag] = _mod

@_dn_triton.jit
def _dn_compare(A, B, S, N: _dt.constexpr, BLOCK: _dt.constexpr):
    i = _dt.arange(0, BLOCK)
    a = _dt.load(A+i, i<N, other=0).to(_dt.float32)
    b = _dt.load(B+i, i<N, other=0).to(_dt.float32)
    sq = _dt.sum((a-b)*(a-b), 0) / _dt.maximum(_dt.sum(b*b, 0), 1e-20)
    finite = _dt.sum(((i<N) & ((_dt.abs(a)==float("inf")) | (_dt.abs(b)==float("inf")) |
                     (a!=a) | (b!=b))).to(_dt.int32), 0) == 0
    _dt.atomic_add(S, ((~finite) | (sq>1e-8)).to(_dt.float32))
    _dt.atomic_max(S+1, sq)
    ar = _dt.reshape(a, (16,64))
    br = _dt.reshape(b, (16,64))
    rsq = _dt.sum((ar-br)*(ar-br), 1) / _dt.maximum(_dt.sum(br*br, 1), 1e-20)
    _dt.atomic_max(S+2, _dt.max(rsq, 0))
    _dt.atomic_add(S+3, _dt.sum((rsq>1e-8).to(_dt.float32), 0))

def _dn_run(replays=2000):
    pairs = (("0b","0a"),("2a","0a"),("2a","0b"),
             ("2b","0a"),("2b","0b"),("2b","2a"))
    tags = ("0a","0b","2a","2b")
    failures = []
    for kind in ("sliding","full"):
        data = _ob_inputs(1,80000,"tp8",kind,1,"fixed",
                          "soft" if kind=="sliding" else "coherent",1,1.,seed=311104)
        fns = {t: kernel_fn if _dn_mods[t] is None else _dn_mods[t].kernel_fn for t in tags}
        for fn in fns.values():
            for _ in range(3): fn(data)
        graphs, outputs = {}, {}
        for tag in tags:
            graphs[tag] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[tag]): outputs[tag] = fns[tag](data)
            graphs[tag].replay()
        stats = {p: torch.zeros(4,device="cuda") for p in pairs}
        for p in pairs:
            _dn_compare[(1,)](outputs[p[0]],outputs[p[1]],stats[p],1024,1024)
            stats[p].zero_()
        gen=torch.Generator(device="cuda").manual_seed(311105)
        bank=torch.randn((replays,*data["qkv"].shape),device="cuda",dtype=torch.bfloat16,generator=gen)
        for i in range(replays):
            data["qkv"].copy_(bank[i])
            order = tags[i%4:]+tags[:i%4]
            for tag in order: graphs[tag].replay()
            for p in pairs:
                _dn_compare[(1,)](outputs[p[0]],outputs[p[1]],stats[p],1024,1024)
        torch.cuda.synchronize()
        for a,b in pairs:
            bad, sq, row, rows = stats[a,b].tolist()
            print(f"LATENULL {kind} {a}/{b} inputs={replays} strict_bad={bad:g} "
                  f"max_rel={sq**.5:.8g} max_row_rel={row**.5:.8g} row_bad={rows:g}",flush=True)
            if bad: failures.append((kind,a,b,bad,sq))
        # Independent FP32 samples, all four arms, after the queue.
        for i in (0,replays//2,replays-1):
            data["qkv"].copy_(bank[i])
            for tag in tags:
                graphs[tag].replay()
                torch.cuda.synchronize()
                ok,msg=_ob_check(data,outputs[tag])
                print(f"LATENULL REF {kind} {tag} sample={i} ok={ok} {msg}",flush=True)
                assert ok, (kind,tag,i,msg)
        del data,graphs,outputs,stats,bank
        torch.cuda.empty_cache()
    print(f"LATENULL FINAL strict_pass={not failures} failures={failures}",flush=True)
    assert not failures, "strict1e-4 failed; classification diagnostic, no promotion"

_ob_selftest()
_dn_run()
'''


def main():
    src = (ROOT / "muse_attention_decode/submission.py").read_text()
    assert src.count("#define ONE_LATE_TID 0") == 1
    sources = {}
    for tag, mode in (("0b",0),("2a",2),("2b",2)):
        other = src.replace("#define ONE_LATE_TID 0", f"#define ONE_LATE_TID {mode}")
        other = other.replace('name="okc_attn_one_v1"', f'name="okc_attn_null_{tag}"')
        sources[tag] = other
    runner = base.RUNNER.format(variants=["_one"],coop_probe=True)
    head, sep, _ = runner.rpartition("\nimport os as _os\n")
    assert sep
    Path(sys.argv[1]).write_text(src + head + "\n_DN_SOURCES = " + repr(sources) + "\n" + BODY)


if __name__ == "__main__": main()
