#!/usr/bin/env python3
"""Run muse_attention_decode's 16 TASK.md tests on one submission file and print the one-launch
kernel's residency limits (length, mode, resident, per_sm). Box only; takes the GPU lock in-process
and lets it go while load_inline builds (tools/bx_box_run.py's GpuLock and extension cache).

    /root/venv/bin/python tools/attp_tests.py <submission.py> [--lock /root/okc/.build/box/gpu.lock]
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bx_box_run as bx  # noqa: E402

PDIR = os.path.join(ROOT, "muse_attention_decode")
KEYS = {"batch": int, "ctx": int, "heads": str, "layer": str, "q_len": int, "spread": str,
        "page": int, "dist": str, "kv_scale": float}


def tests():
    text = open(os.path.join(PDIR, "TASK.md")).read()
    block = text.split("must pass these 16 tests", 1)[1].split("```")[1]
    out = []
    for line in block.strip().splitlines():
        case = {}
        for part in line.split(","):
            k, v = part.strip().split(" ", 1)
            case[k] = KEYS[k](v)
        out.append(case)
    assert len(out) == 16, len(out)
    return out


def main():
    sub = os.path.abspath(sys.argv[1])
    lock = sys.argv[sys.argv.index("--lock") + 1] if "--lock" in sys.argv else ""
    if lock:
        bx.LOCK = bx.GpuLock(lock)
        bx.LOCK.acquire("start")
    bx.content_addressed_extensions("/root/.cache/bx_ext")
    import torch
    sys.path.insert(0, PDIR)
    ref = bx.load_module("reference", os.path.join(PDIR, "reference.py"))
    mod = bx.load_module("submission", sub)
    if lock:
        bx.LOCK.acquire("tests")
    fails = 0
    for i, case in enumerate(tests()):
        data = ref.generate_input(**case, seed=1000 + i)
        out = mod.kernel_fn(data)
        torch.cuda.synchronize()
        ok, msg = ref.check_implementation(data, ref.collect(data, out))
        fails += not ok
        plan = ""
        try:
            cfg = data["config"]
            if mod._one_ok(cfg):
                st = next(iter(mod._ONE_STATE.values()))
                plan = f" plan {mod._one_plan(cfg, data['page_table'].shape[1], st['resident'])}"
        except Exception as exc:
            plan = f" plan ? ({type(exc).__name__})"
        msg = re.sub(r"\s+", " ", str(msg))[:160]
        print(f"TEST {i:2d} {'PASS' if ok else 'FAIL'} {case}{plan} | {msg}", flush=True)
        del data, out
    lim = [int(v) for v in mod._ONE.attn_one_prepare()]
    for j in range(0, len(lim), 5):
        print(f"LIMIT L={lim[j]} mode={lim[j + 1]} resident={lim[j + 2]} per_sm={lim[j + 4]}", flush=True)
    print(f"TESTS {16 - fails}/16 PASS {os.path.basename(sub)}", flush=True)
    if lock:
        bx.LOCK.release("done")


if __name__ == "__main__":
    main()
