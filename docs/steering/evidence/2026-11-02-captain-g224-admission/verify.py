"""Pinned source/model admission screen; never imports target modules."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
import g_expert_g224_chunk_phased as generator


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def function(tree, name):
    return [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name][-1]


def native_contract(source):
    required = [
        "prop.major==9 && prop.minor==0 && prop.cooperativeLaunch",
        "(grid *= 2)>0 && grid==2*prop.multiProcessorCount && grid<=okc::G224_MAX_GRID && grid<okc::WGS_SLOTS",
        "p.M>0 && p.M<=okc::MMAX && p.nloc==32 && p.lo==0",
        "cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm)==cudaSuccess",
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,okc::NTL,sm)==cudaSuccess",
        "resident>=2 && grid<=resident*prop.multiProcessorCount",
        "cudaLaunchCooperativeKernel((const void*)kernel,dim3((unsigned)grid)",
        "dim3(okc::NTL),args,sm,st)",
        "TORCH_CHECK(launch==cudaSuccess",
        "constexpr int G224_WGS = 2, G224_MAX_GRID = 264;",
        "constexpr int WGS_SLOTS = 265;",
    ]
    assert all(source.count(term) == 1 for term in required)
    host = source[source.rindex("  // Ordinary same-stream cooperative launch:"):]
    assert host.index("resident>=2") < host.index("cudaLaunchCooperativeKernel")
    assert source.count("cudaLaunchCooperativeKernel(") == 1
    return required


def main():
    with tempfile.TemporaryDirectory() as folder:
        variant, native, manifest = generator.generate(ROOT, Path(folder))
    assert digest(native) == "53d09576f86e93b9baaa0145dcbdf1b60d8736743677d0a4de60e89864d18492"
    assert digest(variant) == "d67dc8d1c32ad512048c2998229448a97aee3d7d530c3bb2bd6b4accfdc9dd05"
    archive = ROOT / "docs/steering/evidence/2026-11-02-worker-g224-chunk-phased/selected.cu.gz"
    assert native.encode() == gzip.decompress(archive.read_bytes())
    required = native_contract(native)
    negatives = 0
    for term in required:
        try:
            native_contract(native.replace(term, "REMOVED_ADMISSION_CLAUSE", 1))
        except AssertionError:
            negatives += 1
        else:
            raise AssertionError(term)
    tree = ast.parse(variant)
    base = (ROOT / "muse_decode_layer/cand/g1b4_stage_release.py").read_text()
    base_tree = ast.parse(base)
    names = ("prepare", "_release_expert_self_test", "_release_routes", "_release_verify", "_release_private_scratch")
    for name in names:
        assert ast.dump(function(tree, name)) == ast.dump(function(base_tree, name))
    startup = function(tree, "_release_expert_self_test")
    assert ast.literal_eval(startup.args.defaults[0]) == (128, 32, 33, 256, 1, 8, 16)
    startup_text = ast.get_source_segment(variant, startup)
    for term in ("graph.replay()", "_release_verify(torch, out, refs[i])",
                 "torch.count_nonzero(empty_out)", "torch.count_nonzero(helper_out)",
                 "private['ma_ext'] = {}", "production_storage", "banks[i]", "routes[i]"):
        assert term in startup_text
    fallback = function(tree, "_fork_install_fallback")
    assert len(fallback.body) == 1 and isinstance(fallback.body[0], ast.Raise)
    assert "G224_STOP no fallback" in ast.get_source_segment(variant, fallback)
    cases = 0
    for sms in range(1, 133):
        for occupancy in range(4):
            for supplied_grid in (sms - 1, sms, sms + 1):
                grid = 2 * supplied_grid
                accepted = grid > 0 and grid == 2 * sms and grid <= 264 and grid < 265 and occupancy >= 2 and grid <= occupancy * sms
                expected = supplied_grid == sms and occupancy >= 2
                assert accepted == expected
                cases += 1
    result = dict(
        scope="Source assertions and bounded Python model, not C++ execution or target qualification",
        native_sha256=digest(native), variant_sha256=digest(variant),
        variant_bytes=len(variant.encode()), clauses=len(required), negative_cases=negatives,
        admission_model_cases=cases, unchanged_startup_functions=list(names),
        startup_sizes=list(ast.literal_eval(startup.args.defaults[0])),
        fallback="explicit fatal rejection, no Marlin timing substitution",
        target_occupancy="NOT_RUN", target_numerics="NOT_RUN", target_timings="NOT_RUN",
        caveats=["Assertions depend on exact pinned literal spellings", "Bounded integer grid model is not execution of the CUDART wrapper",
                 "Startup AST identity proves retained test code, not that target tests passed",
                 "Append-only variant remains above the upload-size gate; packed successor requires separate verification"],
    )
    destination = Path(__file__).with_name("verification.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
