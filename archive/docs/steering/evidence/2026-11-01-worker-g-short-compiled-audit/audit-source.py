import ast
import difflib
import hashlib
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
from g_native_patch_ladder import flags
from g_native_repair_inventory import inventory
from prod_size_compile_g import bindings

SOURCES = {
    "donor": (".build/today/kel/kernels/muse_decode_layer.py", "1ec189dc6f306cec1694946eb62d64b54daf378c6fc1ada2f0f5e501e973ff2a"),
    "public607": ("muse_decode_layer/cand/g1b4_stage_release.py", "ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab"),
    "common_off": (".build/g-minimal-rootreview/common/on.py", "4cbb13a9a359c9fe870a94ebcabc4706447778849f71d4aa54e47d69abbd1280"),
}
OUT = Path(__file__).parent
BUILD = ROOT / ".build/worker-g-short-audit"

def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def capture(text, donor):
    namespace = bindings(text)
    namespace.update(flags(text))
    calls = []
    setters = []
    def load_inline(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(okc_set_pdl=lambda *args: setters.append(args))
    class Wrapper:
        def __init__(self, module):
            self._module = module
        def __getattr__(self, name):
            return getattr(self._module, name)
    namespace.update(torch=types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda: (9, 0))), load_inline=load_inline, _RouterExt=Wrapper)
    tree = ast.parse(text)
    selector = "_moe" if donor else "_fork_native_moe"
    for name in ("_ma_ext", "_main_defines", selector):
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
        node = nodes[-1] if name == selector else nodes[0]
        exec(ast.get_source_segment(text, node), namespace)
    state = dict(ma_ext={}, ma_sfx=("-DOKC_PRMTSC=1",), ma_sc6=True, _srm_on=False)
    for tokens in (1, 8):
        namespace[selector](tokens, state)
    assert len(calls) == 2
    return calls, setters, flags(text)


result = {"scope": "Exact source and two actual selected native factory calls under CPU stubs; no submission import or kernel execution. Source comparison, not causal latency attribution.", "sources": {}, "comparisons": {}, "artifact_manifest_inventory": []}
texts = {}
all_calls = {}
for label, (relative, expected) in SOURCES.items():
    path = ROOT / relative
    raw = path.read_bytes()
    assert digest(raw) == expected, label
    text = path.read_text(encoding="utf8")
    texts[label] = text
    calls, setters, policies = capture(text, label == "donor")
    all_calls[label] = calls
    record = {"path": relative, "raw_sha256": digest(raw), "lf_sha256": digest(text.encode()), "bytes": len(raw), "selected": [], "base_policy": policies, "stubbed_set_pdl_calls_not_runtime_edge_proof": setters}
    for family, call in zip(("tiny", "tiny2"), calls):
        cuda = call["cuda_sources"]
        folder = BUILD / label / family
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "input.cu").write_text(cuda, encoding="utf8", newline="\n")
        record["selected"].append({"family": family, "cuda_sha256": digest(cuda.encode()), "device_prefix_sha256": digest(cuda.split("#include <torch/extension.h>")[0].encode()), "cpp_sha256": digest(call["cpp_sources"].encode()), "flags": call["extra_cuda_cflags"], "functions": call["functions"], "input": str((folder / "input.cu").relative_to(ROOT))})
    result["sources"][label] = record
for before, after in (("donor", "public607"), ("public607", "common_off")):
    comparison = {}
    for index, family in enumerate(("tiny", "tiny2")):
        old = all_calls[before][index]
        new = all_calls[after][index]
        old_cuda, new_cuda = old["cuda_sources"], new["cuda_sources"]
        patch = "".join(difflib.unified_diff(old_cuda.splitlines(True), new_cuda.splitlines(True), fromfile=before + "/" + family, tofile=after + "/" + family))
        patch_path = OUT / (before + "-to-" + after + "-" + family + ".patch")
        patch_path.write_text(patch, encoding="utf8", newline="\n")
        comparison[family] = {"cuda_equal": old_cuda == new_cuda, "device_prefix_equal": old_cuda.split("#include <torch/extension.h>")[0] == new_cuda.split("#include <torch/extension.h>")[0], "flags_equal": old["extra_cuda_cflags"] == new["extra_cuda_cflags"], "cpp_equal": old["cpp_sources"] == new["cpp_sources"], "patch": str(patch_path.relative_to(ROOT))}
    result["comparisons"][before + "_to_" + after] = comparison
wanted = {entry["cuda_sha256"] for value in result["sources"].values() for entry in value["selected"]}
for base in (ROOT / ".build", ROOT / "docs/steering/evidence"):
    for path in base.rglob("manifest.json"):
        try:
            manifest = json.loads(path.read_text())
        except (ValueError, UnicodeError):
            continue
        options = manifest.get("options", [])
        if manifest.get("source_sha256") in wanted and any("OKC_TINY=1" in flag for flag in options):
            result["artifact_manifest_inventory"].append({"path": str(path.relative_to(ROOT)), "sha256": digest(path.read_bytes()), "source_sha256": manifest["source_sha256"], "device_prefix_sha256": manifest.get("original_device_prefix_sha256"), "flags": [flag for flag in options if flag.startswith("-D")], "resources": manifest.get("resources"), "ptx_sha256": manifest.get("ptx_sha256"), "cubin_sha256": manifest.get("cubin_sha256")})
(OUT / "source-audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"sources": {label: [{"family": entry["family"], "sha256": entry["cuda_sha256"], "prefix": entry["device_prefix_sha256"]} for entry in value["selected"]] for label, value in result["sources"].items()}, "comparisons": result["comparisons"], "matching_full_source_manifests": len(result["artifact_manifest_inventory"])}, indent=2))

