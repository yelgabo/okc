import json
from pathlib import Path
import subprocess
import sys

job="9adfd4dbc98f4998"
folder=Path(__file__).resolve().parent
command=["/usr/local/bin/okc","status",job,"--wait","--json"]
result=subprocess.run(command,capture_output=True,text=True)
(folder/"own-terminal-receipt.json").write_text(result.stdout,encoding="utf8")
(folder/"own-terminal-transport.json").write_text(json.dumps(dict(command=command,exitcode=result.returncode,stderr=result.stderr),indent=2)+"\n")
receipt=json.loads(result.stdout)
assert receipt["id"]==job and receipt["board"]=="muse_decode_layer" and receipt["mode"]=="check"
print(json.dumps(dict(job=job,status=receipt["status"],exitcode=result.returncode,result_status=(receipt.get("result") or {}).get("status"),passed=(receipt.get("result") or {}).get("passed"))),flush=True)
sys.exit(result.returncode)
