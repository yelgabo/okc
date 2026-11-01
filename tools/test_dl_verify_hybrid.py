"""CPU contracts for fixed production dispatch, not device correctness tests."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dl_verify_hybrid import build, CELLS, LOADER, selected_cells

ROOT = Path(__file__).resolve().parents[1]
BODY = (ROOT / "tools/dl_verify_hybrid_body.py").read_text(encoding="utf8")


class Metadata:
    def __init__(self, dtype, shape):
        self.dtype, self.shape, self.device = "torch." + dtype, shape, "cuda:0"
        self.contiguous = True
    def is_contiguous(self):
        return self.contiguous


class Scale:
    def __init__(self, supported=True): self.supported = supported
    def view(self, dtype):
        assert dtype == "uint8"
        return self


def weights():
    return {"w13_sf": Scale(), "w2_sf": Scale(), "immutable": object()}


def data(batch, window, q_len=8):
    result = {"config": dict(batch=batch, q_len=q_len, tokens=batch*q_len,
        window=window, page_size=1, hidden=8192, num_q_heads=16, num_kv_heads=2,
        head_dim=64, n_experts=256, local_experts=32, top_k=8, kv_scale=1.,
        rope_theta=500000. if window else 0.)}
    for name, dtype, shape in (("attn_input","bfloat16",(batch*q_len,8192)),
            ("stream","float32",(batch*q_len,8192)),("page_table","int32",(batch,86008)),
            ("seq_lens","int32",(batch,)),("cache_loc","int64",(batch*q_len,)),
            ("local_start","int32",(1,)),("k_scale","float32",()),("v_scale","float32",()),
            ("k_cache","float8_e4m3fn",(86008,1,2,64)),("v_cache","float8_e4m3fn",(86008,1,2,64))):
        result[name] = Metadata(dtype, shape)
    return result


class VerifyHybrid(unittest.TestCase):
    def env(self, mask):
        events = []
        def backend(name):
            def prepare(weights):
                events.append(("prepare", name, weights))
                return {"backend": name, "scratch": object(), "weights": weights}
            def kernel_fn(d, state):
                events.append(("run", name, state))
                return name, state
            return SimpleNamespace(prepare=prepare, kernel_fn=kernel_fn,
                torch=SimpleNamespace(uint8="uint8"), _w4_scales_ok=lambda sf:sf.supported)
        env = {"_VERIFY_OVERRIDE_MASK": mask, "_VERIFY_HYBRID_G": backend("G"),
               "_VERIFY_HYBRID_OWNED": backend("owned")}
        exec(compile(BODY, "verify-production-dispatch", "exec"), env)
        return env, events

    def test_zero_mask_keeps_exact_scored_file(self):
        source, manifest = build(ROOT)
        # Git may check this file out as CRLF on Windows; the scored payload is
        # the pinned LF source returned by the builder on either platform.
        original = (ROOT / "muse_decode_layer/cand/g1b4_stage_release.py").read_text(encoding="utf8").encode()
        self.assertEqual(source.encode(), original)
        self.assertEqual(manifest["bytes"], 869076)
        self.assertTrue(manifest["default_byte_identical"])
        self.assertTrue(manifest["fits_one_megabyte"])

    def test_all_masks_route_all_twelve_cells(self):
        for mask in range(16):
            env, events = self.env(mask)
            bank = weights()
            state = env["prepare"](bank)
            expected_backends = ["G", "owned"] if mask else ["G"]
            self.assertEqual([e[1] for e in events], expected_backends)
            self.assertTrue(all(e[2] is bank for e in events))
            self.assertEqual(set(state), set(expected_backends))
            if mask:
                self.assertIsNot(state["G"]["scratch"], state["owned"]["scratch"])
            for q_len in (1, 8):
                for batch in (1, 16, 32):
                    for window in (2048, 0):
                        d = data(batch, window, q_len)
                        expected = "owned" if q_len == 8 and (batch,
                            "sliding" if window else "full") in selected_cells(mask) else "G"
                        name, selected = env["kernel_fn"](d, state)
                        self.assertEqual(name, expected, (mask, batch, window, q_len))
                        self.assertIs(selected, state[expected])

    def test_outside_owned_domain_uses_incumbent(self):
        env, _ = self.env(15)
        state = env["prepare"](weights())
        changes = {"batch":8,"q_len":1,"tokens":129,"window":1024,"page_size":16,
            "hidden":4096,"num_q_heads":8,"num_kv_heads":4,"head_dim":128,
            "n_experts":128,"local_experts":16,"top_k":4,"kv_scale":.5,"rope_theta":1.}
        for key, value in changes.items():
            d = data(16, 2048)
            d["config"][key] = value
            self.assertEqual(env["kernel_fn"](d, state)[0], "G", key)
        d = data(16, 2048)
        d["page_table"].shape = (16, 86007)
        self.assertEqual(env["kernel_fn"](d, state)[0], "G")

    def test_raw_abi_metadata_mismatch_uses_incumbent(self):
        env, _ = self.env(15)
        state = env["prepare"](weights())
        for name in data(16, 2048):
            if name == "config": continue
            for field, value in (("dtype","torch.float64"),("shape",(7,)),
                                 ("contiguous",False),("device","cuda:1")):
                d = data(16, 2048)
                setattr(d[name], field, value)
                self.assertEqual(env["kernel_fn"](d, state)[0], "G", (name, field))

    def test_preparation_is_per_call_and_default_skips_owned(self):
        for mask in (0, 15):
            env, _ = self.env(mask)
            first, second = env["prepare"](weights()), env["prepare"](weights())
            for key in first:
                self.assertIsNot(first[key], second[key])
                self.assertIsNot(first[key]["scratch"], second[key]["scratch"])

    def test_readable_source_transport_and_module_isolation(self):
        # Real wrappers exercise generated module construction. No CUDA library
        # is imported; native/backend semantics have their own source reviews.
        stub = 'from types import SimpleNamespace\ntorch=SimpleNamespace(uint8="uint8")\ndef _w4_scales_ok(sf):\n    return sf.supported\ndef prepare(weights):\n    return {"scratch": object()}\ndef kernel_fn(data, state):\n    return state\n'
        for mask in range(1,16):
            source, manifest = build(ROOT, mask, g_backend=stub, owned_backend=stub)
            self.assertFalse(manifest["H200_qualified"])
            self.assertFalse(manifest["new_performance_measurement"])
            self.assertEqual(manifest["selected_cells"], selected_cells(mask))
            tree = ast.parse(source)
            values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)}
            self.assertEqual(values["_VERIFY_HYBRID_SOURCE_G"], stub)
            self.assertEqual(values["_VERIFY_HYBRID_SOURCE_OWNED"], stub)
            env = {}
            exec(compile(source, "hybrid-host-fixture", "exec"), env)
            self.assertIsNot(env["_VERIFY_HYBRID_G"].__dict__, env["_VERIFY_HYBRID_OWNED"].__dict__)
            state = env["prepare"](weights())
            self.assertIsNot(state["G"]["scratch"], state["owned"]["scratch"])

    def test_unsupported_weights_preserve_incumbent_fallback(self):
        for name in ("w13_sf", "w2_sf"):
            env, events = self.env(15)
            bank = weights()
            bank[name].supported = False
            state = env["prepare"](bank)
            self.assertEqual(set(state), {"G"})
            self.assertEqual([e[1] for e in events], ["G"])
            for batch, layer in CELLS:
                self.assertEqual(env["kernel_fn"](data(batch, 2048 if layer=="sliding" else 0), state)[0], "G")

    def test_deadline_also_covers_unsupported_weight_preflight(self):
        env, _ = self.env(15)
        bank = weights()
        bank["w13_sf"].supported = False
        with patch("time.monotonic", side_effect=[0., 1., 296.]):
            with self.assertRaisesRegex(RuntimeError, "hybrid preparation exceeded budget"):
                env["prepare"](bank)

    def test_bad_mask_and_missing_backends_are_rejected(self):
        for mask in (-1,16,True,1.5,"1"):
            with self.assertRaises(ValueError): selected_cells(mask)
        with self.assertRaisesRegex(ValueError,"reviewed production backends"):
            build(ROOT, 1)


if __name__ == "__main__":
    unittest.main()
