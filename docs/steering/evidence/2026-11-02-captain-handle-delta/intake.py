import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[3]
OWNERS = ("tonybolivar", "thomgardiner", "abhishekgahlot2", "Fangzhou66")
records = []
for owner in OWNERS:
    for resource in ("repos", "gists"):
        endpoint = f"users/{owner}/{resource}?per_page=100"
        if resource == "repos":
            endpoint += "&type=owner&sort=updated"
        command = ["gh", "api", "--paginate", "--slurp", endpoint]
        process = subprocess.run(command, capture_output=True)
        record = {
            "owner": owner,
            "resource": resource,
            "endpoint": endpoint,
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": process.returncode,
            "response_sha256": hashlib.sha256(process.stdout).hexdigest(),
            "stderr": process.stderr.decode("utf8", errors="replace"),
        }
        if process.returncode == 0:
            pages = json.loads(process.stdout)
            record["page_counts"] = [len(page) for page in pages]
            items = [item for page in pages for item in page]
            if resource == "repos":
                fields = ("full_name", "description", "pushed_at", "updated_at", "default_branch", "fork", "html_url")
                record["items"] = [{field: item.get(field) for field in fields} for item in items]
            else:
                record["items"] = [{
                    "id": item["id"], "description": item.get("description"),
                    "updated_at": item.get("updated_at"), "html_url": item.get("html_url"),
                    "files": [{"filename": value.get("filename"), "raw_url": value.get("raw_url"),
                               "size": value.get("size")} for value in item.get("files", {}).values()],
                } for item in items]
        records.append(record)
result = {"scope": "Four exact board handles; public metadata only, paginated to terminal by gh.",
          "records": records}
(OUTPUT / "owners.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
print(json.dumps([{key: record[key] for key in ("owner", "resource", "returncode", "page_counts") if key in record}
                  for record in records], indent=2))

