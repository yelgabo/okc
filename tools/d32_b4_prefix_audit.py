"""Capture the pinned layer's actual native factory inputs without compiling."""
import argparse
import json
from pathlib import Path
import types

from g_expert_g224_prefix import BASE, BASE_SHA, NATIVE_SHA, function_source, sha
from g_native_patch_ladder import flags
from prod_size_compile_g import bindings

HELPER_SHA = "b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716"
TOKENS = (1, 8, 16, 17, 24, 32, 33, 128, 256, 384)


def body(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    depth, cursor = 1, opening + 1
    while depth:
        depth += (source[cursor] == "{") - (source[cursor] == "}")
        cursor += 1
    return source[start:cursor]


def capture(source):
    env = bindings(source)
    env.update(flags(source))
    builds = []

    def load_inline(**kwargs):
        native = kwargs["cuda_sources"]
        producer = body(native, "__device__ void wgs_xproducer(")
        record = dict(module=kwargs["name"], flags=kwargs["extra_cuda_cflags"],
                      native_sha256=sha(native), tokens=[],
                      producer_body=producer,
                      range_body=body(native, "__device__ __noinline__ void wgs_ranges("))
        builds.append(record)
        return types.SimpleNamespace(record=record, okc_set_pdl=lambda value: None)

    env.update(torch=types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda: (9, 0))),
               load_inline=load_inline, _MA_EXTS={}, _MA_MAIN_NS=3, _MA_PDL=False,
               _RouterExt=lambda value: value)
    for name in ("_main_defines", "_ma_ext", "_fork_native_moe"):
        exec(function_source(source, name), env)
    state = dict(ma_sfx=("-DOKC_PRMTSC=1",), ma_sc6=True, ma_ext={})
    for tokens in TOKENS:
        ext = env["_fork_native_moe"](tokens, state)
        ext.record["tokens"].append(tokens)
    return builds


def verify(builds):
    b4 = [r for r in builds if "-DOKC_B4FIN=1" in r["flags"]]
    assert len(b4) == 1, "expected one B4 build"
    assert b4[0]["tokens"] == [17, 24, 32], "B4 token selection changed"
    main = next(r for r in builds if 33 in r["tokens"])
    assert main["tokens"] == [33, 128, 256, 384]
    for record in (b4[0], main):
        assert record["native_sha256"] == NATIVE_SHA, "selected native literal changed"
        producer, ranges = record["producer_body"], record["range_body"]
        assert "wgs_ranges(p, S);  // all 32 lanes" in producer
        assert "if ((TIDX & 31) == 0) wgs_ranges" not in producer
        for token in ("__shfl_up_sync(0xffffffffu", "__shfl_sync(0xffffffffu, sa, 31)",
                      "__shfl_sync(0xffffffffu, sb, 31)", "if (lane < 4) S.wgr[lane] = res;",
                      "wgs_cost(0, S.jC[j]) * 16u + OKC_WGSU) << 5",
                      "wgs_cost(1, S.jC[j]) * 16u + OKC_WGSUB) << 4"):
            assert token in ranges, token
        assert ranges.count("__syncwarp();") == 2
    helpers = [r for r in builds if r["native_sha256"] == HELPER_SHA]
    assert len(helpers) == 3
    for record in helpers:
        assert "if ((TIDX & 31) == 0) wgs_ranges(p, S);" in record["producer_body"]
        assert "wgs_prefix(S, nj);" in record["range_body"]


def audit(source):
    assert sha(source) == BASE_SHA, "frozen public607 source changed"
    builds = capture(source)
    verify(builds)
    return dict(base=BASE, base_sha256=sha(source), builds=builds,
                scope="Actual final Python factory with CPU stand-in compiler, private state PRMTSC/SC6 enabled",
                compiled=False, gpu_execution=False, candidate_created=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = (Path(__file__).resolve().parents[1] / BASE).read_text()
    result = audit(source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("B4 and MAIN select the existing parallel scan; three helper builds select the serial scan")
