"""Read-only public-source discovery and CPU geometry/work accounting. No CUDA imports."""
import ast
import ctypes
import datetime
import hashlib
import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).parent


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "okc-bounded-source-research"})
    with urllib.request.urlopen(request, timeout=25) as response:
        return response.read()


def layout(warps, jobs):
    integer = ctypes.c_int32
    class Header(ctypes.Structure):
        _fields_ = [
            ("lut", ctypes.c_uint32 * 256), ("red", ctypes.c_float * 4),
            ("bars", ctypes.c_uint64 * 24), ("wglast", integer * 4),
            ("wcp", integer * (jobs + 1)), ("wcpB", integer * (jobs + 1)),
            ("jrow", integer * (jobs + 1)), ("totals", ctypes.c_uint32 * 2),
            ("wgr", integer * 4), ("seqs", integer * 4), ("fcnt", integer * 32),
            ("xbuf_ptr", ctypes.c_uint64), ("cnt_off", integer * 128),
            ("wc", integer * (warps * 32)), ("jobs", integer * (4 * jobs)),
            ("nj_unit", integer * 3), ("gen", integer), ("ssp", ctypes.c_uint64),
            ("uslot", integer * 8), ("dync", integer * jobs),
            ("next_units", integer * 2), ("eid", ctypes.c_int8 * 3072),
            ("ent", ctypes.c_int16 * 3072), ("pos", ctypes.c_int16 * 3072),
        ]
    size = ctypes.sizeof(Header)
    return {"sizeof": size, "ring_start": (size + 127) // 128 * 128}


raw = (ROOT / "muse_decode_layer/cand/g1b4_stage_release.py").read_bytes()
assert hashlib.sha256(raw).hexdigest() == "ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab"
tree = ast.parse(raw.decode())
cuda = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
            and any(getattr(target, "id", "") == "MOE_CUDA_SRC" for target in node.targets))
assert hashlib.sha256(cuda.encode()).hexdigest() == "c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71"
baseline = layout(16, 88)
assert baseline == {"sizeof": 22240, "ring_start": 22272}
baseline["shared_bytes"] = baseline["ring_start"] + 3 * 36864 + 3 * 32768
assert baseline["shared_bytes"] == 231168
two_cta = layout(8, 168)
two_cta["packed_interval"] = [two_cta["ring_start"], two_cta["ring_start"] + 3 * 18432]
two_cta["x_interval"] = [two_cta["packed_interval"][1], two_cta["packed_interval"][1] + 3 * 12288]
two_cta["shared_bytes"] = two_cta["x_interval"][1]
two_cta["pair_shared_bytes"] = 2 * two_cta["shared_bytes"]
two_cta["pair_registers_if_uniform_96"] = 2 * 320 * 96
assert two_cta["pair_shared_bytes"] <= 233472
assert two_cta["pair_registers_if_uniform_96"] <= 65536
two_cta["not_launchable_as_define_change"] = [
    "gather NT*8==HD assertion", "JMAX168 > MF_XRS96 unconditional assertion; literal136 doneAs allocation",
    "hardcoded four-WG slots, 256-row unit mapping and publication parent ub>>1",
    "host grid<=SM_count and grid<=136; WGS_SLOTS160",
]

scope_path = ROOT / "docs/steering/evidence/2026-11-01-captain-opportunity-sizing.json"
scope_input = json.loads(scope_path.read_text())
selected = []
for load in ("decode 32", "verify 16", "verify 32"):
    cells = {cell["context"]: cell for cell in scope_input["cells"] if cell["load"] == load}
    selected.append((cells["sliding"]["adjusted_us"], cells["full"]["adjusted_us"],
                     cells["sliding"]["expert_profile_us"], cells["full"]["expert_profile_us"]))


def scope(fraction):
    log_ratio = 0.0
    for sliding, full, expert_sliding, expert_full in selected:
        before = 46 * sliding + 16 * full
        after = before - fraction * (46 * expert_sliding + 16 * expert_full)
        log_ratio += math.log(before / after)
    return math.exp(log_ratio / 6)


lower, upper = 0.0, 1.0
for iteration in range(60):
    middle = (lower + upper) / 2
    if scope(middle) < 1.03:
        lower = middle
    else:
        upper = middle

stage = json.loads((ROOT / "docs/steering/evidence/2026-11-01-g-stage-probe/f147d06a64504c1e.decoded.json").read_text())
populations = []
for row in stage["report"]["cases"]:
    counts = row[-1]
    jobs64 = sum((count + 63) // 64 for count in counts if count)
    jobs32 = sum((count + 31) // 32 for count in counts if count)
    jobs24 = sum((count + 23) // 24 for count in counts if count)
    populations.append({"standalone_tokens": row[0], "seed": row[1], "jobs64": jobs64,
                        "jobs32": jobs32, "jobs24": jobs24, "weight_job_ratio_32": jobs32 / jobs64,
                        "weight_job_ratio_24": jobs24 / jobs64})

queries = ['"muse_decode_layer" in:readme pushed:>=2026-10-30',
           '"muse_moe_w4a16" in:readme pushed:>=2026-10-30',
           '"Open Kernel Challenge" in:readme pushed:>=2026-10-31']
searches = []
for query in queries:
    data = json.loads(fetch("https://api.github.com/search/repositories?" + urllib.parse.urlencode({"q": query, "per_page": 100})))
    searches.append({"query": query, "total_count": data["total_count"],
                     "incomplete_results": data["incomplete_results"],
                     "items": [{key: item[key] for key in ("full_name", "pushed_at", "html_url")} for item in data["items"]]})

pins = [
    ("ruwwww/open-kernel-challenge", "c035bf9d5a7d2445d8c8ee5ce49e63b8bb36f49d", "muse_moe_w4a16/submission.py"),
    ("ruwwww/open-kernel-challenge", "c035bf9d5a7d2445d8c8ee5ce49e63b8bb36f49d", "muse_decode_layer/submission.py"),
    ("ruwwww/open-kernel-challenge", "c035bf9d5a7d2445d8c8ee5ce49e63b8bb36f49d", "README.md"),
    ("novitalabs/chord", "6199d44a863a165139d2fd8387b307bbe21f1ca9", "chord_kernels/operator/include/deep_gemm/impls/sm90_w4a16_gemm.cuh"),
    ("novitalabs/chord", "6199d44a863a165139d2fd8387b307bbe21f1ca9", "LICENSE"),
    ("novitalabs/chord", "6199d44a863a165139d2fd8387b307bbe21f1ca9", "chord_kernels/operator/grouped/heuristics.py"),
    ("novitalabs/chord", "6199d44a863a165139d2fd8387b307bbe21f1ca9", "docs/design.md"),
    ("novitalabs/chord", "6199d44a863a165139d2fd8387b307bbe21f1ca9", "docs/performance.md"),
    ("inclusionAI/humming", "948e6fd547e703065be51529ac1db5569256774e", "humming/include/humming/kernel/humming_ws.cuh"),
]
sources = []
for repo, commit, path in pins:
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    data = fetch(url)
    text = data.decode()
    sources.append({"repo": repo, "commit": commit, "path": path, "url": url,
                    "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                    "locators": {term: [index + 1 for index, line in enumerate(text.splitlines()) if term in line][:12]
                                 for term in ("wgs_phase", "wgs_seg", "WgsPend", "p.inter", "WGS_NS", "kW4A16Wait1", "transform_b", "INT4", "_ctas_per_sm", "num_stages < 3")}})

result = {
    "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "mode": "SOURCE/CPU only; no GPU, CUDA import, compiler or challenge CLI",
    "local_source_sha256": hashlib.sha256(raw).hexdigest(), "selected_cuda_sha256": hashlib.sha256(cuda.encode()).hexdigest(),
    "baseline": baseline, "two_cta_rs_concept": two_cta,
    "decoded_ss_added_logical_shared_bytes_per_256row_K4096_unit": 64 * 32768 * 2,
    "extra_x_fraction_of_original_packed_plus_x": {str(ntt): ntt / (9 + ntt) for ntt in (1, 2, 4, 8)},
    "scope": {"affected": "D32/V16/V32, both contexts", "whole_six_scenarios": {str(frac): scope(frac) for frac in (0.10, 0.15, 0.20)},
              "expert_fraction_for_3pct_scenario": upper, "limitation": "Mixed profile/adjusted medians, full exposure, zero replacement cost: NOT a bound or forecast."},
    "standalone_job_repartition": populations, "standalone_not_layer": True,
    "local_input_hashes": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (scope_path, ROOT / "docs/steering/evidence/2026-11-01-g-stage-probe/f147d06a64504c1e.decoded.json")},
    "searches": searches, "public_sources": sources,
}
(OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in result.items() if key not in ("searches", "public_sources")}, indent=2))
