"""Actual generated Python entries plus source and sampled event-model negatives."""
import ast
import collections
import json
import random
import sys
import types
import unittest
from unittest.mock import patch

from attention_layer_continuity import BASE, OUT, build, literal, native_edits, replace_once, sha
from attention_top100_screen import function, literal as configuration_literal


class Tensor:
    def __init__(self, name, shape=()):
        self.name, self.shape, self.device = name, shape, "cpu-standin"

    def view(self, *shape):
        return self

    def t(self):
        return self

    def __eq__(self, other):
        return isinstance(other, Tensor) and (self.name, self.shape) == (other.name, other.shape)


def config(batch, length, window, page=1):
    return dict(batch=batch, q_len=length, tokens=batch * length, num_q_heads=16,
                num_kv_heads=2, head_dim=64, page_size=page, window=window,
                sm_scale=.125, qk_scale_factor=1., rope_theta=0., kv_scale=1.)


def environment(text, captures, ownership):
    tree = ast.parse(text)
    env = dict(_num_sms7=lambda device: 132, LOG2E=1.4426950408889634,
               Q_DIM=1024, HIDDEN=8192, BFP_MAX_T=256, ATT_HAS_PDL=True,
               torch=types.SimpleNamespace(bfloat16="bf16", empty=lambda *args, **kwargs: Tensor("output"),
                                           mm=lambda *args: Tensor("qkv")),
               _inv_freq=lambda *args: "freq", _att_fits7=lambda *args: True,
               _glue_bind_context=lambda *args: None,
               _after_attention=lambda data, state, output, gate, tokens: (output, tokens),
               _TUNED7=configuration_literal(tree, "_TUNED7"))
    for name in ("OKC_FLATCOOP", "OKC_PDL", "OKC_FLAT1B", "OKC_PLAIN8", "OKC_F2", "OKC_S2",
                 "OKC_W0_L1", "OKC_W0_L8", "OKC_W0_PR", "PDL", "PDL_MAX_T"):
        env[name] = literal(text, name)
    def workspace(rows, device, owner):
        ownership.append(owner)
        owner.setdefault("observed", []).append(rows)
        return "acc", "sum", "count"
    env["_workspace7"] = workspace
    env["_cluster_bufs7"] = lambda *args: ("cluster", "barrier")
    control = types.SimpleNamespace(run=lambda *args: captures.append(("control", args)))
    treatment = types.SimpleNamespace(run=lambda *args: captures.append(("continuity", args)))
    env["_att7"] = lambda: control
    env["_att7_continuity"] = lambda: treatment
    wanted = {"_plan7", "_launch7", "_pdl", "_supported7", "_decode_attention7",
              "_attention_module7", "_prepare_attention_modules7", "kernel_fn"}
    aliases = {"_attention_fa3_base", "_vj_kernel_fn", "_glue_context_parent_kernel"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
             or isinstance(node, ast.Assign) and any(getattr(target, "id", "") in aliases for target in node.targets)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "actual-proposal-entries", "exec"), env)
    return env, control, treatment


def invoke(env, cfg, modules=None):
    state = dict(att_ok=True, qkv=Tensor("weight", (1536, 8192)), attention_scratch={})
    if "_prepare_attention_modules7" in env:
        env["_prepare_attention_modules7"](state)
    if modules is not None:
        state["attention_modules7"] = modules
    data = dict(config=cfg, attn_input=Tensor("input"), qkv=Tensor("qkv"),
                page_table=Tensor("pages", (cfg["batch"], 86008)))
    data.update({name: name for name in ("seq_lens", "cache_loc", "k_cache", "v_cache", "k_scale", "v_scale")})
    result = env["kernel_fn"](data, state)
    return state, data, result


def source_contract(control, treatment):
    restored = treatment
    for old, new in reversed(native_edits()):
        restored = replace_once(restored, new, old)
    assert restored == control
    for token in ("#define OKC_PGCONT 0", "#define OKC_PTR 0", "#define OKC_F2 0", "#define OKC_S2 0",
                  "#define OKC_XP_NOFENCE 0", "if (q == 0 && x < ntiles)",
                  "if constexpr (R == 1 && OKC_CONT) cp_wait<0>();"):
        assert token in treatment, token
    init = treatment.index("if constexpr (Cfg<L, FLAT>::WS && OKC_CONT)")
    split = treatment.index("auto run_segments", init)
    assert "mbar_fence_init();\n    }\n    __syncthreads();" in treatment[init:split]
    assert "alignas(16) float continuity_merge[64 * 18]" in treatment
    assert "if constexpr (L == 1 && CONT) xs = S.continuity_merge" in treatment
    merge = "const float mm = fmaxf(m_r[0], m1);"
    assert treatment[treatment.index(merge):treatment.index("  if (M_CL)", treatment.index(merge))] == control[control.index(merge):control.index("  if (M_CL)", control.index(merge))]


def ring_events(lengths, stages, warps, seed, published=True, early=False):
    if not published:
        raise ValueError("role observes unpublished full/empty initialization")
    tiles = [(segment, local) for segment, count in enumerate(lengths) for local in range(count)]
    slots = [None] * stages
    produced, consumed = 0, 0
    rng = random.Random(seed)
    while consumed < len(tiles):
        can_produce = produced < len(tiles) and (slots[produced % stages] is None or early)
        can_consume = slots[consumed % stages] is not None
        if can_produce and (not can_consume or rng.randrange(3) != 0):
            slot = produced % stages
            if slots[slot] is not None:
                raise ValueError("raw slot overwritten before all last readers retire")
            slots[slot] = [produced, produced // stages & 1, set(range(warps))]
            produced += 1
        else:
            current = slots[consumed % stages]
            assert current[:2] == [consumed, consumed // stages & 1]
            reader = rng.choice(sorted(current[2]))
            current[2].remove(reader)
            if not current[2]:
                slots[consumed % stages] = None
                consumed += 1
    assert produced == consumed == sum(lengths)
    return produced


def page_events(counts, readers=128, omit_last_join=False):
    fills, reads = 0, 0
    for count in counts:
        slots = {}
        def fill(tile):
            nonlocal fills
            if tile >= count:
                return
            if tile % 4 in slots:
                raise ValueError("page slot reused before last reader")
            slots[tile % 4] = (tile, set(range(readers)))
            fills += 1
        for tile in range(3):
            fill(tile)
        for tile in range(count):
            fill(tile + 3)
            owner, pending = slots[tile % 4]
            assert owner == tile
            for reader in range(readers - int(omit_last_join and tile == count - 1)):
                pending.remove(reader)
                reads += 1
            if not pending:
                del slots[tile % 4]
        if slots:
            raise ValueError("page barrier reinitialized with last reader outstanding")
    assert fills == sum(counts) and reads == fills * readers
    return fills, reads


class ContinuityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = BASE.read_text(encoding="utf-8")
        cls.variant, cls.control, cls.treatment = build(cls.base)

    def test_plain_deterministic_and_wrong_source(self):
        self.assertEqual(build(self.base)[0], self.variant)
        self.assertLessEqual(len(self.variant.encode()), 1_000_000)
        self.assertNotIn("\r", self.variant)
        with self.assertRaises(ValueError):
            build(self.base + "\n")

    def test_native_exact_inverse_and_math(self):
        source_contract(self.control, self.treatment)
        for old, new in native_edits()[1:]:
            with self.assertRaises((ValueError, AssertionError)):
                source_contract(self.control, replace_once(self.treatment, new, old))

    def test_all_original_python_ast_except_two_edits(self):
        original = ast.parse(self.base)
        changed = ast.parse(self.variant)
        originals = [node for node in original.body if isinstance(node, ast.FunctionDef)]
        added = {"_attention_continuity_cuda7", "_att7_continuity", "_prepare_attention_modules7", "_attention_module7"}
        variants = [node for node in changed.body if isinstance(node, ast.FunctionDef) and node.name not in added]
        self.assertEqual(len(originals), len(variants))
        differences = [(old.name, old.lineno) for old, new in zip(originals, variants)
                       if ast.dump(old) != ast.dump(new)]
        self.assertEqual([name for name, line in differences], ["prepare", "_launch7"])

    def test_actual_final_binding_all12_and_fallbacks(self):
        records = []
        cases = [(batch, length, window, 1) for length in (1, 8) for batch in (1, 16, 32) for window in (2048, 0)]
        cases += [(33, 1, 0, 1), (4, 1, 2048, 64), (4, 8, 0, 64), (16, 1, 0, 64)]
        for batch, length, window, page in cases:
            captures, controls, owners, original_owners = [], [], [], []
            actual, _, _ = environment(self.variant, captures, owners)
            original, _, _ = environment(self.base, controls, original_owners)
            cfg = config(batch, length, window, page)
            state, data, output = invoke(actual, cfg)
            invoke(original, cfg)
            target = batch in (16, 32) and window == 0 and page == 1
            self.assertEqual(captures[-1][0], "continuity" if target else "control")
            self.assertEqual(captures[-1][1], controls[-1][1])
            self.assertIs(owners[-1], state["attention_scratch"])
            args = captures[-1][1]
            records.append(dict(batch=batch, length=length, window=window, page=page,
                                module=captures[-1][0], launch_arguments=[value.name if isinstance(value, Tensor) else value for value in args]))
        (OUT / "actual-entry-cases.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    def test_state_module_ownership_and_scope_negatives(self):
        captures, owners = [], []
        env, control, treatment = environment(self.variant, captures, owners)
        state, data, _ = invoke(env, config(16, 1, 0))
        other, _, _ = invoke(env, config(16, 1, 0))
        self.assertIsNot(state["attention_scratch"], other["attention_scratch"])
        self.assertIsNot(state["attention_modules7"], other["attention_modules7"])
        state["attention_scratch"]["private"] = 1
        self.assertNotIn("private", other["attention_scratch"])
        for kwargs in ({"pdl": True}, {"n_splits": 132}, {"w0": 1023}, {"_mod": control}):
            with self.assertRaises(ValueError):
                env["_launch7"](data, attention_state=state, **kwargs)
        data["config"] = config(1, 1, 0)
        with self.assertRaises(ValueError):
            env["_launch7"](data, attention_state=state, _mod=treatment)

    def test_actual_builders_sources_flags_and_cache_independence(self):
        tree = ast.parse(self.variant)
        captures = []
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(get_device_capability=lambda: (9, 0))
        utility = types.ModuleType("torch.utils.cpp_extension")
        utility.load_inline = lambda **kwargs: captures.append(kwargs) or types.SimpleNamespace(name=kwargs["name"])
        env = dict(ATT7_CUDA_SRC=self.control, ATT7_CPP_SRC=literal(self.base, "ATT7_CPP_SRC"),
                   _ATT7=None, _ATT7_CONTINUITY=None, torch=torch)
        for name in ("_attention_continuity_cuda7", "_att7", "_att7_continuity"):
            function(tree, name, env)
        with patch.dict(sys.modules, {"torch": torch, "torch.utils.cpp_extension": utility}), patch.dict("os.environ"):
            original = env["_att7"]()
            treatment = env["_att7_continuity"]()
            self.assertIsNot(original, treatment)
            self.assertIs(env["_att7"](), original)
            self.assertIs(env["_att7_continuity"](), treatment)
        self.assertEqual(len(captures), 2)
        self.assertEqual(captures[0]["extra_cuda_cflags"], captures[1]["extra_cuda_cflags"])
        self.assertEqual(captures[0]["cuda_sources"], self.control)
        self.assertEqual(captures[1]["cuda_sources"], self.treatment)
        env["ATT7_CUDA_SRC"] += "\n"
        with self.assertRaises(ValueError):
            env["_attention_continuity_cuda7"]()
        (OUT / "actual-builder-cases.json").write_text(json.dumps([
            {key: value for key, value in call.items() if key not in ("cpp_sources", "cuda_sources")}
            for call in captures], indent=2) + "\n", encoding="utf-8")

    def test_ring_phase_and_early_reuse_negative(self):
        for length, stages, warps in ((1, 3, 4), (8, 4, 8)):
            for counts in ([], [0], [1], [0, 1, 2, 0, 11, 1, 257], [1, 1, 1, 1, 1, 1, 1]):
                for seed in range(40):
                    self.assertEqual(ring_events(counts, stages, warps, seed), sum(counts))
        with self.assertRaises(ValueError):
            ring_events([1], 3, 4, 0, published=False)
        with self.assertRaises(ValueError):
            ring_events([9, 1], 3, 4, 0, early=True)

    def test_pages_short_long_and_lifetime_negative(self):
        for counts in ([], [0], [1], [2], [3], [4], [0, 1, 3, 9, 1, 257]):
            self.assertEqual(page_events(counts), (sum(counts), sum(counts) * 128))
        with self.assertRaises(ValueError):
            page_events([1, 1], omit_last_join=True)

    def test_key_newrow_and_causal_coverage(self):
        for length in (1, 8):
            for sequence in (0, 1, 17, 63, 64, 65, 129, 8193):
                total = sequence + length
                cuts = sorted(set([0, min(1, total), min(63, total), min(67, total), total]))
                seen = collections.Counter()
                new = collections.Counter()
                causal = collections.Counter()
                for start, end in zip(cuts, cuts[1:]):
                    for tile in range((end - start + 63) // 64):
                        for key in range(start + tile * 64, min(end, start + (tile + 1) * 64)):
                            seen[key] += 1
                            if key >= sequence:
                                new[key] += 1
                            for query in range(length):
                                if key <= sequence + query:
                                    causal[query, key] += 1
                self.assertEqual(seen, collections.Counter(range(total)))
                self.assertEqual(new, collections.Counter(range(sequence, total)))
                self.assertTrue(all(count == 1 for count in causal.values()))
                self.assertEqual(len(causal), sum(sequence + query + 1 for query in range(length)))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    unittest.main(verbosity=2)
