"""Generate one isolated layer attention continuity proposal; never dispatch."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "muse_decode_layer/cand/g1b4_stage_release.py"
BASE_SHA = "ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab"
ALLOCATION = "48111acef2ddccb42e166d06fa5cb32d33b43c76"
OUT = ROOT / "docs/steering/evidence/2026-11-02-worker-attention-continuity"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("missing or ambiguous source anchor: " + old[:90])
    return text.replace(old, new, 1)


def literal(text, name):
    return next(ast.literal_eval(node.value) for node in ast.parse(text).body
                if isinstance(node, ast.Assign)
                and any(getattr(target, "id", None) == name for target in node.targets))


def native_edits():
    return [
        ("#define OKC_CONT 0", "#define OKC_CONT 1"),
        ("template <int L, bool F>\nstruct Smem {",
         "template <bool Enabled>\nstruct ContinuityMergeStorage {};\n"
         "template <>\nstruct ContinuityMergeStorage<true> {\n"
         "  alignas(16) float continuity_merge[64 * 18];\n};\n\n"
         "template <int L, bool F>\n"
         "struct Smem : ContinuityMergeStorage<(L == 1 && F && OKC_CONT)> {"),
        ("if (L == 1 && !CONT && !(OKC_NOXCH1 && !M_CL)) {",
         "if (L == 1 && !(OKC_NOXCH1 && !M_CL)) {"),
        ("    float* xs = reinterpret_cast<float*>(S.k) + ((warp & 1) * 32 + lane) * 18;",
         "    float* xs;\n"
         "    if constexpr (L == 1 && CONT) xs = S.continuity_merge + ((warp & 1) * 32 + lane) * 18;\n"
         "    else xs = reinterpret_cast<float*>(S.k) + ((warp & 1) * 32 + lane) * 18;"),
        ("      mbar_fence_init();\n    }\n  }\n  // segment list in shared memory",
         "      mbar_fence_init();\n    }\n    __syncthreads();\n  }\n  // segment list in shared memory"),
    ]


def build(base):
    if sha(base) != BASE_SHA:
        raise ValueError("wrong incumbent LF identity")
    cuda = literal(base, "ATT7_CUDA_SRC")
    treatment = cuda
    for old, new in native_edits():
        treatment = replace_once(treatment, old, new)
    tree = ast.parse(base)
    builder = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_att7")
    original_builder = ast.get_source_segment(base, builder)
    new_builder = original_builder.replace("_ATT7", "_ATT7_CONTINUITY").replace("def _att7():", "def _att7_continuity():")
    new_builder = replace_once(new_builder, 'name=f"muse_att_v298w1pf_{arch}"',
                               'name=f"muse_att_cont_merge_' + sha(treatment)[:12] + '_{arch}"')
    new_builder = replace_once(new_builder, "cuda_sources=ATT7_CUDA_SRC", "cuda_sources=_attention_continuity_cuda7()")
    helpers = '\n\n_ATT7_CONTINUITY = None\n\n\n'
    helpers += 'def _attention_continuity_cuda7():\n    import hashlib\n'
    helpers += '    source = ATT7_CUDA_SRC\n'
    helpers += '    if hashlib.sha256(source.encode()).hexdigest() != ' + repr(sha(cuda)) + ':\n'
    helpers += '        raise ValueError("attention control source identity mismatch")\n'
    helpers += '    edits = ' + repr(native_edits()) + '\n'
    helpers += '    for old, new in edits:\n'
    helpers += '        if source.count(old) != 1:\n            raise ValueError("attention continuity source anchor mismatch")\n'
    helpers += '        source = source.replace(old, new, 1)\n'
    helpers += '    return source\n\n\n' + new_builder + '\n\n\n'
    helpers += '''def _prepare_attention_modules7(state):
    state["attention_modules7"] = (_att7(), _att7_continuity())


def _attention_module7(cfg, plan, tma, device, state, requested):
    modules = state["attention_modules7"]
    target = (cfg["batch"] in (16, 32) and cfg["q_len"] in (1, 8)
              and cfg["window"] == 0 and cfg["page_size"] == 1
              and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2
              and cfg["head_dim"] == 64)
    if target:
        expected = (2 * _num_sms7(device), 1, 1, 0, 1)
        actual = tuple(plan[key] for key in ("n_splits", "flat", "nc", "cs", "ncl"))
        expected_tma = (1024 << 2) if cfg["q_len"] == 1 else ((1135 << 2) | 2)
        if actual != expected or tma != expected_tma:
            raise ValueError("attention continuity launch flags or geometry drift")
    selected = modules[int(target)]
    if requested is not None and requested is not selected:
        raise ValueError("attention continuity module outside its selected scope")
    return selected
'''
    variant = replace_once(base, "\n\n_WS7 = {}", helpers + "\n\n_WS7 = {}")
    variant = replace_once(variant, "builders = [_att7, bfp__get_ext,", "builders = [lambda: _prepare_attention_modules7(st), bfp__get_ext,")
    variant = replace_once(variant, "(_mod or _att7()).run(qkv,", "_attention_module7(cfg, plan, tma, device, attention_state, _mod).run(qkv,")
    if "\r" in variant or len(variant.encode()) > 1_000_000:
        raise ValueError("proposal must be plain LF and at most 1MB")
    ast.parse(variant)
    return variant, cuda, treatment


def generate(out=OUT):
    base = BASE.read_text(encoding="utf-8")
    variant, control, treatment = build(base)
    out.mkdir(parents=True, exist_ok=True)
    files = {"proposal.py": variant, "control-attention.cu": control, "treatment-attention.cu": treatment,
             "native.diff": "".join(difflib.unified_diff(control.splitlines(True), treatment.splitlines(True),
                                                      fromfile="control-attention.cu", tofile="treatment-attention.cu")),
             "proposal.diff": "".join(difflib.unified_diff(base.splitlines(True), variant.splitlines(True),
                                                        fromfile=str(BASE.relative_to(ROOT)), tofile="proposal.py"))}
    for name, text in files.items():
        (out / name).write_bytes(text.encode())
    manifest = dict(allocation=ALLOCATION, base_lf_sha256=BASE_SHA,
                    scope="one source/CPU/fixed-secondary proposal; no performance qualification or dispatch authority",
                    edits=native_edits(), files={name: dict(sha256=sha(text), bytes=len(text.encode()))
                                                for name, text in files.items()})
    (out / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    print(json.dumps(generate(parser.parse_args().out), indent=2))
