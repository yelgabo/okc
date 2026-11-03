"""Box sensitivity test for the pdl8f bound: how far does a stale qkv read move the layer metric?

Under alternating inputs, the attention is fed qkv rows from the previous replay (the other input) for: all rows,
one token's rows, or one (token, KV pair) block of q / k / v. The metric is pdl8's max |diff| / max |ref| of the
layer outputs against the clean output of the same input.

    python3 box_fault.py LAYER.py REPO_DIR
"""
import importlib.util
import os
import sys

layer, repo = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(repo, "muse_decode_layer"))
import torch  # noqa: E402
import reference  # noqa: E402

spec = importlib.util.spec_from_file_location("lut", layer)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
st = m.prepare(reference.generate_weights(1, device="cuda"))
dev = torch.device("cuda")
gen = torch.Generator(device=dev).manual_seed(7)
inner = m._decode_attention7
stale = {}


def faulty(data, state, qkv):
    mode = stale.get("mode")
    if mode is None:
        return inner(data, state, qkv)
    prev = stale["buf"].get(qkv.shape)
    if prev is None:
        prev = qkv.clone()
        stale["buf"][qkv.shape] = prev
    mixed = qkv.clone()
    if mode == "all":
        mixed.copy_(prev)
    elif mode == "token":
        mixed[0].copy_(prev[0])
    elif mode == "qhead":
        mixed[0, 0:64].copy_(prev[0, 0:64])
    elif mode == "kv":
        mixed[0, 1024:1152].copy_(prev[0, 1024:1152])
    out = inner(data, state, mixed)
    prev.copy_(qkv)
    return out


m._decode_attention7 = faulty
for B, L, layer_kind in ((1, 8, "full"), (1, 8, "sliding"), (1, 1, "sliding"), (16, 1, "sliding"), (32, 1, "full")):
    T = B * L
    data = m._pdl8_data(B, L, layer_kind, gen, dev)
    x0 = data["attn_input"].clone()
    s = torch.randn(x0.shape, generator=gen, device=dev)
    x1 = (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x0.dtype)
    off = dict(st, pdl8_on={})
    refs = []
    for x in (x0, x1):
        data["attn_input"].copy_(x)
        refs.append([t.clone() for t in m._pdl8_flat(m.kernel_fn(data, off))])
    torch.cuda.synchronize()
    line = []
    for mode in ("all", "token", "qhead", "kv"):
        stale["mode"], stale["buf"] = mode, {}
        worst = [0.0, 0.0]
        bad = 0
        for k in range(6):
            data["attn_input"].copy_((x0, x1)[k & 1])
            outs = m._pdl8_flat(m.kernel_fn(data, off))
            if k == 0:
                continue
            ref = refs[k & 1]
            for j, (a, b) in enumerate(zip(outs, ref)):
                if a.is_floating_point():
                    d = float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6))
                    worst[j] = max(worst[j], d)
                else:
                    bad += int((a != b).sum())
        line.append(f"{mode}: out {worst[0]:.2e} s2 {worst[1]:.2e} ids_mismatch {bad}")
        stale["mode"] = None
    print(f"FAULT T={T} {layer_kind}: " + "; ".join(line), flush=True)
    data["attn_input"].copy_(x0)
