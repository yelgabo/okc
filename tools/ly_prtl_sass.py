#!/usr/bin/env python3
"""SASS of the fused pooler + router sk2 kernels (gle extension at T <= 8, bf16_proj extension at T = 16 / 32) for a
control file and a candidate, compiled on the box CPU with each file's own load_inline arguments, and compared kernel
by kernel: registers, stack, local (spill) bytes, total instructions, and every HMMA loop's instruction count and
opcode mix. Kernel names are normalized (the gle extension's md5 tag removed) before matching.

    CUDA_VISIBLE_DEVICES= python tools/ly_prtl_sass.py CONTROL.py CANDIDATE.py OUTDIR [gle|bfp ...]
"""
import concurrent.futures as cf
import importlib.util
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ly_micro_sass as ms  # noqa: E402  (sets CUDA_VISIBLE_DEVICES and the sm_90 capability)


def capture(path, which):
    spec = importlib.util.spec_from_file_location("cand_" + os.path.basename(path).replace(".", "_"), path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    got = {}

    class Ext:
        def __getattr__(self, n):
            return lambda *a, **k: None

    def rec(*a, **kw):
        if a:
            kw = {**dict(zip(("name", "cpp_sources"), a)), **kw}
        got.update(kw)
        return Ext()

    m.load_inline = rec
    out = {}
    for k in which:
        got.clear()
        if k == "gle":
            m._GLE_EXT = None
            m._gle_ext()
        else:
            m.bfp__ext = None
            m.bfp__get_ext()
        out[k] = (got["name"], got["cuda_sources"], list(got["extra_cuda_cflags"]))
    return out


def norm(name):
    return re.sub(r"(sk2_kernel_gle)[0-9a-f]{12}", r"\1", name)


def main():
    ctl, cand, outdir = sys.argv[1:4]
    which = sys.argv[4:] or ["gle", "bfp"]
    os.makedirs(outdir, exist_ok=True)
    caps = {"ctl": capture(ctl, which), "cand": capture(cand, which)}
    jobs = {}
    with cf.ThreadPoolExecutor(2 * len(which)) as ex:
        for arm, cap in caps.items():
            for k in which:
                name, src, flags = cap[k]
                jobs[(arm, k)] = ex.submit(ms.compile_one, outdir, f"{arm}_{k}", name, src, flags)
        rep = {}
        for (arm, k), f in jobs.items():
            _, sass, res = f.result()
            r, a = ms.resources(res), ms.analyze(sass)
            rep[(arm, k)] = {norm(n): {**a.get(n, {}), **r.get(n, {})} for n in a if "sk2_kernel" in n}
    json.dump({f"{a}_{k}": v for (a, k), v in rep.items()}, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    for k in which:
        c, w = rep[("ctl", k)], rep[("cand", k)]
        for n in sorted(set(c) | set(w)):
            if n not in c or n not in w:
                print(f"{k} ONLY-IN-{'cand' if n in w else 'ctl'} {n[:90]}")
                continue
            a, b = c[n], w[n]
            same_loops = [(l["n"], l["mix"]) for l in a["loops"]] == [(l["n"], l["mix"]) for l in b["loops"]]
            # template tail after the output type: PK, FUSE (Lb1ELb1EEv = packed x, fused pooler + router)
            tail = n.split("13__nv_bfloat16")[-1][:10] if "13__nv_bfloat16" in n else n.split("EfLb")[-1][:10]
            tag = "FUSE" if tail.endswith("Lb1EEv") else "    "
            print(f"{k} {tag} {n[18:44]} {tail:10s} reg {a.get('reg')}->{b.get('reg')} stack {a.get('stack')}->{b.get('stack')} "
                  f"local {a.get('local')}->{b.get('local')} ins {a['n']}->{b['n']} loops "
                  f"{[l['n'] for l in a['loops']]}->{[l['n'] for l in b['loops']]} {'SAME' if same_loops else 'DIFF'}")
            if not same_loops:
                for la, lb in zip(a["loops"], b["loops"]):
                    if (la["n"], la["mix"]) != (lb["n"], lb["mix"]):
                        print("   ctl ", la["n"], la["mix"])
                        print("   cand", lb["n"], lb["mix"])


if __name__ == "__main__":
    main()
