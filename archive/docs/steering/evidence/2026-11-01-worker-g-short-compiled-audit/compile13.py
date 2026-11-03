import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
from prod_size_compile_g import sections

OUT = Path(__file__).parent
BUILD = ROOT / ".build/worker-g-short-audit"
CACHE = Path("/root/.cache/okc-cuda130-88")
source_audit = json.loads((OUT / "source-audit.json").read_text())
driver_path = ROOT / ".build/research-gamma13/artifacts/adapted-driver.py"
driver = driver_path.read_text()
assert "nvidia/cu13/lib/libnvrtc.so.13" in driver
assert "*args.define" in driver
namespace = {"__name__": "_worker_g_short_exact_secondary13"}
exec(compile(driver, str(driver_path), "exec"), namespace)
result = {"scope": "Secondary device-only NVRTC13/PTXAS13 sm90a, exact captured selected source and flags; no target NVCC equivalence, host wrapper compile/link, CUDA driver, GPU execution, judge or timing", "driver_sha256": hashlib.sha256(driver.encode()).hexdigest(), "started_utc_epoch": time.time(), "records": {}}
for label in ("donor", "public607"):
    for selected in source_audit["sources"][label]["selected"]:
        family = selected["family"]
        folder = BUILD / label / family
        path = folder / "input.cu"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == selected["cuda_sha256"]
        flags = [flag for flag in selected["flags"] if flag.startswith("-D")]
        args = types.SimpleNamespace(source=path, out=folder, cache=CACHE, sc6=1, define=flags, forward_declare_wgs=True)
        namespace["compile_source"](args)
        disassembly = subprocess.run([str(CACHE / "cuda_nvdisasm-linux-x86_64-13.0.85-archive/bin/nvdisasm"), str(folder / "device.cubin")], capture_output=True, text=True, check=True)
        (folder / "device.sass").write_text(disassembly.stdout)
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["source_sha256"] == selected["cuda_sha256"]
        assert manifest["original_device_prefix_sha256"] == selected["device_prefix_sha256"]
        assert len(manifest["resources"]) == 2
        operation_sections = sections(folder / "device.sass")
        record = {"manifest": manifest, "artifacts": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in ("input.cu", "device.ptx", "device.cubin", "device.sass", "ptxas.log", "nvrtc.log")}, "section_instruction_counts": {symbol: len(operations) for symbol, operations in operation_sections.items()}}
        result["records"][label + "/" + family] = record
        (OUT / "compile13.json").write_text(json.dumps(result, indent=2) + "\n")
        print("COMPLETED", label, family, manifest["resources"], flush=True)
result["common_off_reuses_public607"] = source_audit["comparisons"]["public607_to_common_off"]
result["finished_utc_epoch"] = time.time()
(OUT / "compile13.json").write_text(json.dumps(result, indent=2) + "\n")

