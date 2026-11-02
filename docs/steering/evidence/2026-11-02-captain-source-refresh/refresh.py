import base64
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

destination = Path(__file__).resolve().parent
records = []

def fetch(endpoint, fields=None):
    command = ["C:/Program Files/GitHub CLI/gh.exe", "api", "-X", "GET", endpoint]
    for key, value in (fields or {}).items():
        command.extend(["-f", key + "=" + str(value)])
    completed = subprocess.run(command, capture_output=True)
    record = dict(endpoint=endpoint, fields=fields, at=datetime.datetime.now(datetime.timezone.utc).isoformat(), returncode=completed.returncode, stdout_sha256=hashlib.sha256(completed.stdout).hexdigest(), stderr=completed.stderr.decode("utf8", errors="replace"))
    if completed.returncode == 0:
        record["data"] = json.loads(completed.stdout)
    records.append(record)
    return record.get("data")

for query in ('"Open Kernel Challenge" pushed:>=2026-11-02', 'muse kernel pushed:>=2026-11-02'):
    fetch("search/repositories", dict(q=query, per_page=100))
fetch("search/code", dict(q='"muse_decode_layer" NOT repo:yelgabo/okc NOT repo:lee101/openkernels NOT repo:rohanarun/Auto-research-agent', per_page=100))
for repository in ("keell0renz/muse-open-kernel-challenge", "lilaclilac09/open-kernel-challenge", "alygg77/okc-muse-spark-kernels", "vjeux/okc-2026", "Kushagrabainsla/muse-spark-open", "jacobpmiller/open-kernel-challenge-2026", "ruwwww/open-kernel-challenge"):
    fetch("repos/" + repository + "/commits/main")
tree = fetch("repos/keell0renz/muse-open-kernel-challenge/git/trees/8e30e1be994db4310794b8d04e90d18d895deb84", dict(recursive=1))
source_record = fetch("repos/keell0renz/muse-open-kernel-challenge/contents/kernels/muse_decode_layer.py", dict(ref="8e30e1be994db4310794b8d04e90d18d895deb84"))
source = base64.b64decode(source_record["content"])
source_summary = dict(repository="keell0renz/muse-open-kernel-challenge", ref="8e30e1be994db4310794b8d04e90d18d895deb84", path=source_record["path"], blob=source_record["sha"], bytes=len(source), sha256=hashlib.sha256(source).hexdigest(), complete_tree=tree["truncated"] is False, has_prepare=b"def prepare(" in source, has_kernel_fn=b"def kernel_fn(" in source)
source_record.pop("content")
source_record.pop("encoding", None)
destination.joinpath("retrieval.json").write_bytes((json.dumps(dict(records=records, source_summary=source_summary), indent=2) + "\n").encode())
print(json.dumps(source_summary))
print(json.dumps([dict(query=record["fields"]["q"], count=record["data"]["total_count"], repos=sorted({item.get("full_name", item.get("repository", {}).get("full_name")) for item in record["data"]["items"]})) for record in records if record["endpoint"].startswith("search/")]))
