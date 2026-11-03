"""Write a private check that certifies main's attn_one.cu against the public attention build.

    python3 tools/attn_make_main_cert.py PUBLIC.py OUT.py [--inputs K] [--rounds R]

The judged path of OUT.py is PUBLIC.py byte for byte (the runner is appended after it). At import
the runner execs two more copies into their own module objects, each with its own one-launch
extension name and its own workspace state: `ctrl2`, the public file again (the null), and
`main`, muse_attention_decode/submission.py with attn_one.cu inlined (tools/attn_inline_one.py)
at every switch default. The other three extensions keep their names, so they are shared, which
is valid only while their sources match; the generator checks that.

  CERTSASS  each arm's one-launch .so: the machine code (.text sections, with their sh_info and
            flags) of every kernel in its embedded sm_90a cubin, names normalized (the
            anonymous-namespace hash), compared per function against ctrl. The judge rejects files that start
            processes, so cuobjdump is not run there.
  CERTBIT   per scored shape and held-out input (seeds 1000 + k): each arm called eagerly on the
            same inputs; output and the written cache rows compared bitwise against ctrl.
  CERTCHK   the judge-style FP32 check on seed-1 inputs for ctrl and main, batch-1 shapes.
  CERTCOLD  per scored shape: cold-L2 graph replays (256 MB flush), arms in a seeded random
            order per round; medians, the per-round paired median main - ctrl with a bootstrap
            95% interval, and the null ctrl2 - ctrl.
  CERTSUM   repeated at the end for the log tail.
"""

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import attn_make_bench_one as base  # noqa: E402

ONE_NAME = 'name="okc_attn_one_v1"'

CERT = r'''

import hashlib
import os as _ct_os
import random as _ct_random
import re as _ct_re
import types as _ct_types

@@CONFIG@@


def _ct_load():
    mods = {}
    for name, src in _CT_SRCS.items():
        try:
            m = _ct_types.ModuleType(f"okc_cert_{name}")
            exec(compile(src, f"{name}_attention.py", "exec"), m.__dict__)
            if m._ONE is None:
                raise RuntimeError("one-launch extension unavailable")
            mods[name] = m
            print(f"CERTBUILD {name} loaded so={m._ONE.__file__} {_ob_elapsed()}", flush=True)
        except Exception as exc:
            print(f"CERTBUILD {name} FAIL {type(exc).__name__}: {str(exc)[:600]}", flush=True)
    return mods


def _ct_anon(text):
    """Replaces each mangled anonymous-namespace name (per-file hash) by ANON, reading its
    length from the _ZN<len> prefix, since the next identifier also starts with digits."""
    out, at = [], 0
    for m in _ct_re.finditer(r"(\d+)(_GLOBAL__N_)", text):
        if m.start() < at:
            continue
        n = int(m.group(1))
        out.append(text[at:m.start()] + "ANON")
        at = m.start(2) + n
    out.append(text[at:])
    return "".join(out)


def _ct_cubin_text(blob):
    """Per kernel, the machine code of every sm_90 cubin embedded uncompressed in an extension
    .so: ELF64 images with e_machine EM_CUDA (190), their .text.<name> sections keyed by the
    normalized name, with the section's sh_info and flags."""
    import struct
    out = {}
    at = blob.find(b"\x7fELF", 1)
    while at >= 0:
        try:
            if blob[at + 4] == 2 and struct.unpack_from("<H", blob, at + 18)[0] == 190:
                shoff, = struct.unpack_from("<Q", blob, at + 40)
                shentsize, shnum, shstrndx = struct.unpack_from("<HHH", blob, at + 58)
                secs = [struct.unpack_from("<IIQQQQIIQQ", blob, at + shoff + i * shentsize)
                        for i in range(shnum)]
                stro = at + secs[shstrndx][4]
                for name_off, _, flags, _, off, size, _, info, _, _ in secs:
                    end = blob.index(b"\0", stro + name_off)
                    name = blob[stro + name_off:end].decode()
                    if name.startswith(".text."):
                        key = _ct_anon(name[6:])
                        out[key] = (info >> 24, flags, bytes(blob[at + off:at + off + size]))
        except Exception:
            pass
        at = blob.find(b"\x7fELF", at + 1)
    return out


def _ct_sass(path):
    with open(path, "rb") as f:
        return _ct_cubin_text(f.read())


_CT_RESULT = {}


def _ct_sass_compare(mods):
    try:
        sass = {"ctrl": _ct_sass(_ONE.__file__)}
        for name, m in mods.items():
            sass[name] = _ct_sass(m._ONE.__file__)
        ref = sass["ctrl"]
        for name, s in sass.items():
            if name == "ctrl":
                continue
            same = [k for k in ref if s.get(k) == ref[k]]
            diff = sorted(set(ref) ^ set(s) | {k for k in ref if k in s and s[k] != ref[k]})
            for k in diff:
                a, b = ref.get(k, (0, 0, b"")), s.get(k, (0, 0, b""))
                nd = sum(x != y for x, y in zip(a[2], b[2])) + abs(len(a[2]) - len(b[2]))
                print(f"CERTSASS DIFF {name} {k[:100]} info={a[0]}/{b[0]} bytes={len(a[2])}/"
                      f"{len(b[2])} differing_bytes={nd}", flush=True)
            kern = [k for k in ref if "attn_one_kernel" in k]
            ksame = sum(s.get(k) == ref[k] for k in kern)
            digest = hashlib.sha256(b"".join(ref[k][2] for k in sorted(kern))).hexdigest()[:16]
            line = (f"{name}_vs_ctrl functions={len(ref)}/{len(s)} identical={len(same)} "
                    f"differing={len(diff)} attn_one_kernel_identical={ksame}/{len(kern)} "
                    f"ctrl_kernel_digest={digest}")
            _CT_RESULT[f"sass_{name}"] = line
            print("CERTSASS " + line, flush=True)
    except Exception as exc:
        _CT_RESULT["sass"] = f"FAIL {type(exc).__name__}: {str(exc)[:300]}"
        print("CERTSASS " + _CT_RESULT["sass"], flush=True)


def _ct_fns(mods):
    fns = {"ctrl": kernel_fn}
    for name, m in mods.items():
        fns[name] = m.kernel_fn
    return fns


def _ct_shapes():
    for batch, layer, q_len in _OB_SCORED:
        yield batch, layer, q_len, f"b={batch:2d} {layer:7s} L={q_len}"


def _ct_inputs(batch, layer, q_len, seed):
    return _ob_inputs(batch, 80000, "tp8", layer, q_len, "fixed" if batch == 1 else "agentic",
                      "soft" if layer == "sliding" else "coherent", 1, 1.0, seed=seed)


def _ct_slots(data):
    cfg = data["config"]
    pos = data["seq_lens"].long()[:, None] + torch.arange(cfg["q_len"], device=data["qkv"].device)
    size = cfg["page_size"]
    slots = (torch.gather(data["page_table"].long(), 1, pos // size) * size + pos % size).reshape(-1)
    hkv = cfg["num_kv_heads"]
    return (data["k_cache"].view(-1, hkv, 64)[slots].view(torch.uint8).clone(),
            data["v_cache"].view(-1, hkv, 64)[slots].view(torch.uint8).clone())


_CT_BIT = {}


def _ct_bitwise(fns):
    for batch, layer, q_len, tag in _ct_shapes():
        for k in range(_CT_INPUTS):
            try:
                data = _ct_inputs(batch, layer, q_len, 1000 + k)
                outs = {}
                for name, fn in fns.items():
                    out = _ob_run(lambda: fn(data), f"{name} {tag}")
                    torch.cuda.synchronize()
                    outs[name] = (out.clone(),) + _ct_slots(data)
                ref = outs["ctrl"]
                parts = []
                for name, o in outs.items():
                    if name == "ctrl":
                        continue
                    eq = torch.equal(o[0], ref[0]) and torch.equal(o[1], ref[1]) \
                        and torch.equal(o[2], ref[2])
                    _CT_BIT.setdefault(name, [0, 0])[0 if eq else 1] += 1
                    parts.append(f"{name}={'EQUAL' if eq else 'DIFF'} "
                                 f"out_{_ob_diffstat(o[0], ref[0])} "
                                 f"krows_eq={torch.equal(o[1], ref[1])} vrows_eq={torch.equal(o[2], ref[2])}")
                print(f"CERTBIT {tag} in={k} " + " ; ".join(parts), flush=True)
                outs = data = None
            except Exception as exc:
                _CT_BIT.setdefault("error", [0, 0])[1] += 1
                print(f"CERTBIT FAIL {tag} in={k} {type(exc).__name__}: {str(exc)[:300]}", flush=True)
            torch.cuda.empty_cache()
    _CT_RESULT["bit"] = " ".join(f"{n}=equal{v[0]}/diff{v[1]}" for n, v in _CT_BIT.items())
    print("CERTBIT summary " + _CT_RESULT["bit"], flush=True)


def _ct_check(fns):
    fails = 0
    for batch, layer, q_len, tag in _ct_shapes():
        if batch != 1:
            continue
        try:
            data = _ct_inputs(batch, layer, q_len, 1)
            for name in ("ctrl", "main"):
                if name not in fns:
                    continue
                out = _ob_run(lambda: fns[name](data), f"{name} {tag}")
                torch.cuda.synchronize()
                ok, msg = _ob_check(data, out)
                fails += not ok
                print(f"CERTCHK {'PASS' if ok else 'FAIL'} {name} {tag} {msg}", flush=True)
            data = None
        except Exception as exc:
            fails += 1
            print(f"CERTCHK FAIL {tag} {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        torch.cuda.empty_cache()
    _CT_RESULT["chk"] = f"failing={fails}"


def _ct_graph(fn, data):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(data)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(data)
    torch.cuda.synchronize()
    return graph


def _ct_med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def _ct_boot(xs, seed):
    rng = _ct_random.Random(seed)
    meds = sorted(_ct_med([rng.choice(xs) for _ in xs]) for _ in range(400))
    return meds[10], meds[389]


_CT_SUM = []


def _ct_cold(fns):
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    names = list(fns)
    print(f"CERTORDER seed={_CT_SEED} rounds={_CT_ROUNDS} inputs={_CT_INPUTS} arms={names}",
          flush=True)
    for batch, layer, q_len, tag in _ct_shapes():
        samples = {n: [] for n in names}
        paired = {n: [] for n in names if n != "ctrl"}
        for k in range(_CT_INPUTS):
            try:
                data = _ct_inputs(batch, layer, q_len, 1000 + k)
                cap = list(names)
                _ct_random.Random(_CT_SEED + 31 * k + batch + q_len).shuffle(cap)
                graphs = {n: _ct_graph(fns[n], data) for n in cap}
                for r in range(_CT_ROUNDS):
                    order = list(names)
                    _ct_random.Random(_CT_SEED + 7919 * k + r).shuffle(order)
                    evs = []
                    for n in order:
                        flush.zero_()
                        a = torch.cuda.Event(enable_timing=True)
                        z = torch.cuda.Event(enable_timing=True)
                        a.record()
                        graphs[n].replay()
                        z.record()
                        evs.append((n, a, z))
                    torch.cuda.synchronize()
                    t = {n: 1000.0 * a.elapsed_time(z) for n, a, z in evs}
                    for n in names:
                        samples[n].append(t[n])
                        if n != "ctrl":
                            paired[n].append(t[n] - t["ctrl"])
                graphs = data = None
            except Exception as exc:
                print(f"CERTCOLD FAIL {tag} in={k} {type(exc).__name__}: {str(exc)[:300]}",
                      flush=True)
            torch.cuda.empty_cache()
        meds = " ".join(f"{n}={_ct_med(samples[n]):.2f}" for n in names)
        deltas = []
        for n, xs in paired.items():
            if xs:
                lo, hi = _ct_boot(xs, _CT_SEED + batch)
                deltas.append(f"{n}-ctrl={_ct_med(xs):+.3f} [{lo:+.3f},{hi:+.3f}]")
        line = f"{tag} {meds} " + " ".join(deltas)
        _CT_SUM.append(line)
        print(f"CERTCOLD {line} {_ob_elapsed()}", flush=True)


def _ct_main():
    mods = _ct_load()
    _ct_sass_compare(mods)
    fns = _ct_fns(mods)
    for step in (_ct_bitwise, _ct_check, _ct_cold):
        try:
            step(fns)
        except Exception as exc:
            print(f"CERT {step.__name__} failed: {type(exc).__name__}: {exc}", flush=True)
    print(f"CERTDONE {_ob_elapsed()}", flush=True)
    for line in _CT_SUM:
        print("CERTSUM " + line, flush=True)
    for key, val in _CT_RESULT.items():
        print(f"CERTSUM {key} {val}", flush=True)


if not _ct_os.environ.get("ATTN_CERT_DONE"):
    _ct_os.environ["ATTN_CERT_DONE"] = "1"
    try:
        _ct_main()
    except Exception as _exc:
        print(f"CERT failed: {type(_exc).__name__}: {_exc}", flush=True)
'''


def _rename_one(src, prefix):
    if src.count(ONE_NAME) != 1:
        raise SystemExit(f"expected one {ONE_NAME} in the source")
    return src.replace(ONE_NAME, f'name="{prefix}_okc_attn_one_v1"')


def _other_sources(src):
    return re.findall(r'(_(?:PREP|ATTN|WS)_SRC = r""".*?""")', src, re.S)


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        raise SystemExit(__doc__)
    pub_path, out = Path(args.pop(0)), Path(args.pop(0))
    inputs, rounds, seed = 2, 100, 20261103
    while args:
        a = args.pop(0)
        if a == "--inputs":
            inputs = int(args.pop(0))
        elif a == "--rounds":
            rounds = int(args.pop(0))
        else:
            raise SystemExit(f"unknown option {a}")
    pub = pub_path.read_text()
    sub = (ROOT / "muse_attention_decode/submission.py").read_text()
    cu = (ROOT / "muse_attention_decode/attn_one.cu").read_text()
    sub, n = re.subn(r'_ONE_SRC = r""".*?"""', lambda _: f'_ONE_SRC = r"""\n{cu}"""', sub,
                     flags=re.S)
    if n != 1:
        raise SystemExit("no _ONE_SRC block in submission.py")
    if _other_sources(pub) != _other_sources(sub) or len(_other_sources(pub)) != 3:
        raise SystemExit("prep / attn / ws sources differ between the public file and main")
    runner = base.RUNNER.format(variants=["_one"], coop_probe=True)
    head, sep, _ = runner.rpartition("\nimport os as _os\n")
    if not sep:
        raise SystemExit("runner layout changed: no 'import os as _os' block")
    config = "\n".join([
        f"_CT_SRCS = {{'ctrl2': {_rename_one(pub, 'null')!r}, 'main': {_rename_one(sub, 'cert')!r}}}",
        f"_CT_INPUTS = {inputs!r}",
        f"_CT_ROUNDS = {rounds!r}",
        f"_CT_SEED = {seed!r}",
    ])
    text = pub + head + CERT.replace("@@CONFIG@@", config, 1)
    out.write_text(text)
    print(f"wrote {out} sha256={hashlib.sha256(text.encode()).hexdigest()} "
          f"public sha256={hashlib.sha256(pub.encode()).hexdigest()} "
          f"attn_one.cu sha256={hashlib.sha256(cu.encode()).hexdigest()} "
          f"inputs={inputs} rounds={rounds}")


if __name__ == "__main__":
    main()
