import base64
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[3]
SCRATCH = ROOT / ".build/captain-handle-delta"
SCRATCH.mkdir(parents=True, exist_ok=True)
REPOSITORIES = (
    "tonybolivar/tonybolivar",
    "thomgardiner/sglang-sm120-nvfp4",
    "abhishekgahlot2/mixture-of-kittens",
    "Fangzhou66/torchtitan",
)
records = []

def fetch(endpoint):
    process = subprocess.run(["gh", "api", endpoint], capture_output=True)
    receipt = {"endpoint": endpoint, "at_utc": datetime.now(timezone.utc).isoformat(),
               "returncode": process.returncode,
               "response_sha256": hashlib.sha256(process.stdout).hexdigest(),
               "stderr": process.stderr.decode("utf8", errors="replace")}
    return receipt, json.loads(process.stdout) if process.returncode == 0 else None

for repository in REPOSITORIES:
    requests = []
    receipt, metadata = fetch("repos/" + repository)
    requests.append(receipt)
    if metadata is None:
        records.append({"repository": repository, "requests": requests})
        continue
    receipt, commit = fetch("repos/" + repository + "/commits/" + metadata["default_branch"])
    requests.append(receipt)
    if commit is None:
        records.append({"repository": repository, "requests": requests})
        continue
    pin = commit["sha"]
    receipt, tree = fetch("repos/" + repository + "/git/trees/" + pin + "?recursive=1")
    requests.append(receipt)
    selected_files = []
    if tree is not None:
        (SCRATCH / (repository.replace("/", "__") + "-tree.json")).write_text(json.dumps(tree), encoding="utf8")
        for entry in tree.get("tree", []):
            if entry["type"] == "blob" and entry["path"].lower() in ("readme.md", "readme.rst", "readme"):
                receipt, blob = fetch("repos/" + repository + "/git/blobs/" + entry["sha"])
                requests.append(receipt)
                if blob is None:
                    continue
                content = base64.b64decode(blob["content"])
                blob_hash = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
                assert blob_hash == entry["sha"]
                destination = SCRATCH / (repository.replace("/", "__") + "-" + entry["path"].replace("/", "__"))
                destination.write_bytes(content)
                selected_files.append({"path": entry["path"], "git_blob": blob_hash,
                                       "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                                       "scratch_path": str(destination.relative_to(ROOT))})
    records.append({"repository": repository, "pin": pin, "default_branch": metadata["default_branch"],
                    "tree_truncated": tree.get("truncated") if tree else None,
                    "tree_entries": len(tree.get("tree", [])) if tree else None,
                    "name_hits": [entry["path"] for entry in tree.get("tree", [])
                                  if any(term in entry["path"].lower() for term in ("muse", "openkernel", "okc", "nvfp4", "marlin", "sm120"))] if tree else [],
                    "files": selected_files, "requests": requests})
(OUTPUT / "followups.json").write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf8")
print(json.dumps([{key: record.get(key) for key in ("repository", "pin", "tree_entries", "tree_truncated", "name_hits", "files")}
                  for record in records], indent=2))

