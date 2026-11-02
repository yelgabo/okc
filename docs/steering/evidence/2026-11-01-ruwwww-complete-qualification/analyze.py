import ast
import base64
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[4]
PIN = "c035bf9d5a7d2445d8c8ee5ce49e63b8bb36f49d"
API = "https://api.github.com/repos/ruwwww/open-kernel-challenge"


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "okc-source-qualification"})
    return urllib.request.urlopen(request, timeout=30).read()


def api(path):
    return json.loads(fetch(API + path))


def identity(raw):
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "git_blob": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest(),
    }


def functions(source):
    return {
        node.name: {"line": node.lineno, "text": ast.get_source_segment(source, node)}
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def metadata_keys(value, path=""):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            location = path + "/" + key
            if any(term in key.lower() for term in ("hash", "sha", "source")):
                found.append({"path": location, "value": str(item)[:200]})
            found.extend(metadata_keys(item, location))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(metadata_keys(item, path + "/" + str(index)))
    return found


sys.path.insert(0, str(ROOT / "tools"))
from g_integration_common import build

public = (ROOT / "muse_decode_layer/cand/g1b4_stage_release.py").read_text()
common, manifest = build(public, True)
tree = api("/git/trees/" + PIN + "?recursive=1")
source_path = "muse_decode_layer/submission.py"
source_entry = next(item for item in tree["tree"] if item["path"] == source_path)
raw = base64.b64decode(api("/git/blobs/" + source_entry["sha"])["content"])
assert identity(raw)["git_blob"] == source_entry["sha"]
assert raw == fetch("https://raw.githubusercontent.com/ruwwww/open-kernel-challenge/" + PIN + "/" + source_path)
remote = raw.decode()
assert hashlib.sha256(common.encode()).hexdigest() == "4cbb13a9a359c9fe870a94ebcabc4706447778849f71d4aa54e47d69abbd1280"
receipt_path = "muse_decode_layer/profiles/submission.f60b136cfdc242cb.json"
receipt_entry = next(item for item in tree["tree"] if item["path"] == receipt_path)
receipt_raw = base64.b64decode(api("/git/blobs/" + receipt_entry["sha"])["content"])
receipt = json.loads(receipt_raw)
history = {}
for path in (source_path, receipt_path):
    history[path] = [
        {"sha": item["sha"], "date": item["commit"]["committer"]["date"], "message": item["commit"]["message"]}
        for item in api("/commits?sha=" + PIN + "&path=" + path + "&per_page=3")
    ]
source_commit = history[source_path][0]["sha"]
old_entry = api("/contents/" + source_path + "?ref=" + source_commit)
assert old_entry["sha"] == source_entry["sha"]
repo = api("")
head = api("/commits/" + PIN)
receipt_txt = fetch("https://raw.githubusercontent.com/ruwwww/open-kernel-challenge/" + PIN + "/muse_decode_layer/profiles/submission.f60b136cfdc242cb.txt")
public_receipt_path = ROOT / "muse_decode_layer/cand/profiles/g1b4_stage_release.607eb52bdaf24160.json"
public_receipt_raw = public_receipt_path.read_bytes()
public_receipt = json.loads(public_receipt_raw)
remote_functions, common_functions = functions(remote), functions(common)
selected = {}
for name in ("dense", "prepare", "_after_attention", "_boundary1", "_boundary2", "_route", "_route_split", "_moe", "_ma_prepare", "_launch7", "bfp__static_choice"):
    source_function = remote_functions.get(name)
    target_function = common_functions.get(name)
    record = {"remote": source_function, "common_final": target_function}
    if source_function and target_function:
        record["diff"] = "\n".join(difflib.unified_diff(source_function["text"].splitlines(), target_function["text"].splitlines(), n=2))
    selected[name] = record
literal_hashes = {}
for label, text in (("remote", remote), ("common", common)):
    literals = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) and len(node.value.value) > 50000:
            literals.append({"targets": [ast.unparse(target) for target in node.targets], "line": node.lineno, **identity(node.value.value.encode())})
    literal_hashes[label] = literals
chord = {}
for path in ("chord_kernels/operator/grouped/heuristics.py", "docs/performance.md"):
    url = "https://api.github.com/repos/novitalabs/chord/contents/" + path + "?ref=6199d44a863a165139d2fd8387b307bbe21f1ca9"
    entry = json.loads(fetch(url))
    content = base64.b64decode(entry["content"])
    lines = content.decode().splitlines()
    chord[path] = {**identity(content), "line_count": len(lines), "anchors": [{"line": index + 1, "text": line} for index, line in enumerate(lines) if any(term in line for term in ("def _ctas_per_sm", "3-8", "7-15", "num_stages < 3", "2 * 2 *", "stages >= 3"))]}
result = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "checkout": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "scope": "Pinned GitHub tree/blobs/history/receipt and AST/source only; no module import, compiler, GPU, challenge CLI or donor census.",
    "remote_pin": PIN, "source": identity(raw), "source_path": source_path,
    "public607_lf": identity(public.encode()), "common": identity(common.encode()),
    "head_changed_paths": [item["filename"] for item in head["files"]],
    "history": history, "unchanged_source_since": source_commit,
    "receipt_txt": {**identity(receipt_txt), "identity_or_pick_lines": [line for line in receipt_txt.decode().splitlines() if any(term in line.lower() for term in ("sha256", "source hash", "uploaded", "mm_pick", "gluev2", "tune"))]},
    "historical607": {"job_id": public_receipt["job_id"], **identity(public_receipt_raw), "cells": [{"case": item["case"], "median_us": item["median_ns"] / 1000} for item in public_receipt["benchmarks"]]},
    "source_anchors": {label: [{"line": index + 1, "text": line} for index, line in enumerate(text.splitlines()) if any(term in line for term in ("cudaLaunchCooperativeKernel", "cudaLaunchKernelEx(&cfg, okc::moe_fused", "OKC_WGSFCI", "pr8(float", "okc_set_pdl(1 if _pdl", "KPERM3", "-DOKC_SC6W=1"))] for label, text in (("remote", remote), ("common", common))},
    "license_metadata": repo.get("license"), "tree_truncated": tree["truncated"],
    "license_like_paths": [item["path"] for item in tree["tree"] if any(term in item["path"].lower() for term in ("license", "copying", "notice"))],
    "receipt": {"path": receipt_path, **identity(receipt_raw), "job_id": receipt["job_id"], "problem": receipt["problem"], "submission": receipt["submission"], "summary": receipt["summary"], "source_identity_keys": metadata_keys(receipt), "cells": [{"case": item["case"], "median_us": item["median_ns"] / 1000, "baseline_us": item["baseline_ns"] / 1000} for item in receipt["benchmarks"]], "profiles": receipt["profiles"]},
    "selected_functions": selected, "large_literal_identities": literal_hashes, "chord_correction": chord,
}
destination = Path(__file__).with_name("analysis.json")
destination.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
print(json.dumps({"output": str(destination), "source": result["source"], "receipt": result["receipt"]["summary"], "changed": result["head_changed_paths"], "license": result["license_metadata"], "source_identity_keys": result["receipt"]["source_identity_keys"]}, indent=2))
