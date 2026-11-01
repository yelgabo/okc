"""Production shape dispatcher; constants and modules are supplied by the builder.

The mask is fixed in the emitted file, never selected by timings or input values.
Each bit chooses one complete owned verify pipeline, including its BF16 boundaries.
"""


def _verify_owned_metadata(data, batch):
    # Raw native ABIs assume these layouts. Inspect metadata only; tensor values
    # (including scales, routing and cache lengths) remain runtime inputs.
    device = data["attn_input"].device
    specs = {"attn_input": ("torch.bfloat16", (batch * 8, 8192)),
             "stream": ("torch.float32", (batch * 8, 8192)),
             "page_table": ("torch.int32", (batch, 86008)),
             "seq_lens": ("torch.int32", (batch,)),
             "cache_loc": ("torch.int64", (batch * 8,)),
             "local_start": ("torch.int32", (1,)),
             "k_scale": ("torch.float32", ()),
             "v_scale": ("torch.float32", ())}
    for name, (dtype, shape) in specs.items():
        value = data[name]
        if (str(value.dtype) != dtype or value.shape != shape or
                value.device != device or not value.is_contiguous()):
            return False
    k, v = data["k_cache"], data["v_cache"]
    return (k.shape == v.shape and len(k.shape) == 4 and k.shape[0] > 0 and
            k.shape[1:] == (1, 2, 64) and
            all(str(value.dtype) == "torch.float8_e4m3fn" and
                value.device == device and value.is_contiguous() for value in (k, v)))


def _verify_override_bit(data):
    config = data["config"]
    batch = config.get("batch")
    window = config.get("window")
    if batch not in (16, 32) or window not in (0, 2048):
        return 0
    expected = {"batch": batch, "q_len": 8, "tokens": batch * 8,
                "window": window, "page_size": 1, "hidden": 8192,
                "num_q_heads": 16, "num_kv_heads": 2, "head_dim": 64,
                "n_experts": 256, "local_experts": 32, "top_k": 8,
                "kv_scale": 1.0}
    if any(config.get(key) != value for key, value in expected.items()):
        return 0
    if data["page_table"].shape != (batch, 86008):
        return 0
    if config.get("rope_theta") != (500000.0 if window else 0.0):
        return 0
    index = (0 if batch == 16 else 2) + (0 if window == 2048 else 1)
    bit = (1 << index) & _VERIFY_OVERRIDE_MASK
    return bit if bit and _verify_owned_metadata(data, batch) else 0


def _verify_owned_weights_eligible(weights):
    # Exactly the rejection predicate in the frozen owned _moe1_prepare.
    # This reads immutable weights during prepare, never timed input values.
    owned = _VERIFY_HYBRID_OWNED
    return all(owned._w4_scales_ok(weights[name + "_sf"].view(owned.torch.uint8))
               for name in ("w13", "w2"))


def prepare(weights):
    import time
    deadline = time.monotonic() + 295
    # These states share only immutable input weights/code. Each backend creates
    # its own scratch, module buffers, repacked weights and Lt workspace closures.
    state = {"G": _VERIFY_HYBRID_G.prepare(weights)}
    if time.monotonic() >= deadline:
        raise RuntimeError("verify hybrid incumbent preparation exceeded budget")
    if _VERIFY_OVERRIDE_MASK and _verify_owned_weights_eligible(weights):
        state["owned"] = _VERIFY_HYBRID_OWNED.prepare(weights)
    if time.monotonic() >= deadline:
        raise RuntimeError("verify hybrid preparation exceeded budget")
    return state


def kernel_fn(data, state):
    if "owned" in state and _verify_override_bit(data):
        return _VERIFY_HYBRID_OWNED.kernel_fn(data, state["owned"])
    return _VERIFY_HYBRID_G.kernel_fn(data, state["G"])
