"""CPU-only, fail-closed screen of the exact layer attention continuity proposal.

Does not import the submission, torch, a donor, or a GPU/dispatch library.
Executes only the AST-extracted Python shape/launch functions with CPU stand-ins.
This is a source-path witness, not a compiler, GPU ordering, or numerical test.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
LAYER = "muse_decode_layer/cand/g1b4_stage_release.py"
EXPECTED = "ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def literal(tree, name):
    value = next(n.value for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    class DictCalls(ast.NodeTransformer):
        def visit_Call(self, node):
            if not (isinstance(node.func, ast.Name) and node.func.id == "dict"
                    and not node.args and all(k.arg is not None for k in node.keywords)):
                raise ValueError("nonliteral call in configuration")
            return ast.Dict(keys=[ast.Constant(k.arg) for k in node.keywords],
                            values=[self.visit(k.value) for k in node.keywords])
    return ast.literal_eval(DictCalls().visit(value))


def function(tree, name, env):
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise ValueError(f"ambiguous/missing function {name}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), LAYER, "exec"), env)
    return env[name]


def screen(text):
    if digest(text) != EXPECTED:
        raise ValueError("incumbent source identity changed; re-audit, do not silently repin")
    tree = ast.parse(text)
    cuda = literal(tree, "ATT7_CUDA_SRC")
    flags_node = next(n for n in ast.walk(next(n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_att7"))
        if isinstance(n, ast.keyword) and n.arg == "extra_cuda_cflags")
    flags = eval(compile(ast.Expression(flags_node.value), "flags", "eval"), {"major": 9})
    macros = {}
    for line in cuda.splitlines():
        words = line.split()
        if len(words) >= 3 and words[0] == "#define" and words[2].isdigit():
            macros[words[1]] = int(words[2])
    for flag in flags:
        if flag.startswith("-D"):
            k, v = flag[2:].split("=")
            macros[k] = int(v)
    captures = []
    fake_mod = SimpleNamespace(run=lambda *args: captures.append(args))
    env = {"_num_sms7": lambda _: 132, "LOG2E": 1.4426950408889634,
           "torch": SimpleNamespace(bfloat16="bf16", empty=lambda *a, **k: "output"),
           "_workspace7": lambda *a: ("acc", "sum", "count"),
           "_cluster_bufs7": lambda *a: ("cluster", "barrier"),
           "_inv_freq": lambda *a: "freq"}
    for name in ("_TUNED7", "OKC_FLATCOOP", "OKC_PDL", "OKC_FLAT1B", "OKC_PLAIN8",
                 "OKC_F2", "OKC_S2", "OKC_W0_L1", "OKC_W0_L8", "OKC_W0_PR", "PDL", "PDL_MAX_T"):
        env[name] = literal(tree, name)
    function(tree, "_plan7", env)
    launch = function(tree, "_launch7", env)
    pdl = function(tree, "_pdl", env)
    rows = []
    for length in (1, 8):
        for batch in (1, 16, 32):
            for window in (2048, 0):
                cfg = dict(batch=batch, q_len=length, num_q_heads=16, num_kv_heads=2,
                    head_dim=64, page_size=1, window=window, sm_scale=.125,
                    qk_scale_factor=1., rope_theta=0.)
                data = dict(config=cfg, qkv=SimpleNamespace(device="cpu-standin"),
                    page_table=SimpleNamespace(shape=(batch, 86008)))
                data.update({k: k for k in ("seq_lens", "cache_loc", "k_cache", "v_cache", "k_scale", "v_scale")})
                launch(data, _mod=fake_mod, attention_state={"attention_scratch": {}}, pdl=pdl(batch * length))
                args = captures[-1]
                row = dict(batch=batch, q_len=length, window=window, splits=args[20],
                    flat=args[21], copies=args[22], cluster_size=args[25], clusters=args[26],
                    coop=(args[29] >> 1) & 1, w0=(args[29] >> 2) & 4095,
                    pdl=(args[29] >> 16) & 1)
                if batch > 1 and not window:
                    assert (row["splits"], row["flat"], row["copies"], row["cluster_size"]) == (264, 1, 1, 0)
                    assert row["w0"] == (1024 if length == 1 else 1135)
                    assert row["coop"] == (length == 8)
                rows.append(row)
    required = {
        "decode_half_merge_disabled_by_cont": "if (L == 1 && !CONT && !(OKC_NOXCH1 && !M_CL))",
        "half_merge_uses_raw_ring": "float* xs = reinterpret_cast<float*>(S.k) + ((warp & 1) * 32 + lane) * 18;",
        "producer_segment_drain": "if constexpr (!CONT) {\n        cp_wait<0>();\n        __syncthreads();",
        "consumer_join_changes_scope": "if constexpr (CONT) csync<WS, NT>();",
        "ring_ordinal_carried": "OKC_CONT ? tb : 0",
        "q8_phase_from_ordinal": "uint32_t ph_c = (uint32_t)(tb / STAGES) & 1u;",
        "terminal_producer_drain": "if constexpr (R == 1 && OKC_CONT) cp_wait<0>();",
    }
    for name, token in required.items():
        assert token in cuda, name
    assert macros["OKC_CONT"] == 0 and macros["OKC_XP_NOFENCE"] == 0
    return dict(source_lf_sha256=digest(text), source_lf_bytes=len(text.encode()),
                cuda_sha256=digest(cuda), flags=flags, macros=macros,
                launch_rows=rows, source_witnesses=required,
                status="STOP_EXISTING_MECHANISM_AND_NONISOLATED_SWITCH")


def run():
    text = (ROOT / LAYER).read_text(encoding="utf-8")
    result = screen(text)
    # Identity negative must reject even a benign source drift; no automatic rebase.
    try:
        screen(text + "\n# identity negative\n")
    except ValueError:
        pass
    else:
        raise AssertionError("identity negative accepted")
    result["cpu_checks"] = {"actual_launcher_cases": 12, "source_identity_negative": "PASS"}
    other_text = (ROOT / "muse_attention_decode/submission.py").read_text(encoding="utf-8")
    other = ast.parse(other_text)
    other_cuda = literal(other, "_ONE_SRC")
    penv = {k: literal(other, k) for k in ("_ONE_FLAT", "_ONE_Q8_PROD", "_ONE_PROD", "_ONE_TARGET")}
    plan = function(other, "_one_plan", penv)
    result["standalone_comparator"] = dict(source_lf_sha256=digest(other_text),
        cuda_sha256=digest(other_cuda), plans=[])
    for length in (1, 8):
        for batch in (16, 32):
            got = plan(dict(batch=batch, q_len=length, page_size=1, window=0), 86008,
                       {(length, 49 if length == 1 else 51): 264})
            assert got == (264, 1, 49 if length == 1 else 51)
            result["standalone_comparator"]["plans"].append(dict(batch=batch, q_len=length, selected=got))
    for marker in ("for (int bb = b_first; bb <= b_last; ++bb)",
                   "if (++pst == NS)", "Leave only once every stage has been released"):
        assert marker in other_cuda, marker
    result["receipt_identities"] = []
    receipts = [
        ("docs/steering/evidence/2026-11-02-captain-fresh-incumbent-profile/control.cc474883117947c0.json", "cc474883117947c0"),
        ("muse_attention_decode/profiles/exact_split.4862f8e0dc8a4b5a.json", "4862f8e0dc8a4b5a"),
        ("muse_attention_decode/profiles/bench_a0.5575e7d15d3e49a8.json", "5575e7d15d3e49a8"),
        ("muse_attention_decode/profiles/bench_b.318ae6fc3c3746ce.json", "318ae6fc3c3746ce"),
        ("muse_attention_decode/profiles/pr_qk2.fe75e566c1ae4d51.json", "fe75e566c1ae4d51"),
        ("muse_attention_decode/profiles/vp2_bf16_private.c79ddddb840a4fb7.receipt.json", "c79ddddb840a4fb7"),
        ("muse_attention_decode/profiles/wait-trace.3f44f891d0ef47db.receipt.json", "3f44f891d0ef47db"),
    ]
    for path, job in receipts:
        raw = (ROOT / path).read_bytes()
        d = json.loads(raw)
        assert d.get("id", d.get("job_id")) == job, path
        r = d.get("result", d)
        result["receipt_identities"].append(dict(path=path, job=job,
            sha256=hashlib.sha256(raw).hexdigest(), board=d.get("board", d.get("problem")),
            mode=d.get("mode"), passed=r.get("passed"), returncode=r.get("returncode")))
    result["scope"] = "CPU stand-ins; no host/device compilation, GPU numerics, ordering or timing established"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["status"], result["cpu_checks"])
