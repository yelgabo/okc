import collections
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import types


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).parent
BUILD = ROOT / ".build/research-short-scalar-interleave"
AUDIT = ROOT / "docs/steering/evidence/2026-11-01-worker-g-short-compiled-audit"
CACHE = Path("/root/.cache/okc-cuda130-88")
sys.path.insert(0, str(ROOT / "tools"))
from prod_size_compile_g import sections

OLD = """#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        if (tk < cnt) {
          const float gt = fork_bf16((h ? sm.y : sm.x) * gsc), up = fork_bf16((h ? sm.w : sm.z) * gsc);
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      }"""
NEW = """      float gates[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        if (tk < cnt) gates[h] = fork_bf16((h ? sm.y : sm.x) * gsc);
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        if (tk < cnt) {
          const float gt = gates[h], up = fork_bf16((h ? sm.w : sm.z) * gsc);
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      }"""


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def opcodes(operations):
    return dict(collections.Counter(re.sub(r"^@!?(?:U?P)\d+\s+", "", operation).split()[0].rstrip(";") for operation in operations))


def normalized_ptx(text):
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", ".file", ".loc", ".section .debug", ".section\t.debug")):
            continue
        lines.append(stripped)
    return lines


driver_path = ROOT / ".build/research-gamma13/artifacts/adapted-driver.py"
driver = driver_path.read_text()
prior = json.loads((AUDIT / "compile13.json").read_text())
assert digest(driver.encode()) == prior["driver_sha256"]
namespace = {"__name__": "_scalar_interleave_secondary13"}
exec(compile(driver, str(driver_path), "exec"), namespace)
audit = json.loads((AUDIT / "source-audit.json").read_text())
selected_common = audit["sources"]["common_off"]["selected"]
result = {"scope": "ONE guarded cross-token scalar gate preparation scheduling change, selected Tiny/Tiny2 both bools, secondary NVRTC13/PTXAS13 only; no GPU/driver/host link", "started_epoch": time.time(), "driver_sha256": digest(driver.encode()), "records": {}, "toolchain": {}}
for name, relative in {"nvrtc": "nvidia/cu13/lib/libnvrtc.so.13", "ptxas": "nvidia/cu13/bin/ptxas", "nvdisasm": "cuda_nvdisasm-linux-x86_64-13.0.85-archive/bin/nvdisasm"}.items():
    path = CACHE / relative
    result["toolchain"][name] = {"path": str(path), "sha256": digest(path.read_bytes())}
    if name != "nvrtc":
        result["toolchain"][name]["version"] = subprocess.run([str(path), "--version"], capture_output=True, text=True, check=True).stdout
for selected in selected_common:
    family = selected["family"]
    source_path = ROOT / selected["input"].replace("\\", "/")
    source = source_path.read_text()
    assert digest(source.encode()) == selected["cuda_sha256"]
    assert source.count(OLD) == 1
    changed = source.replace(OLD, NEW)
    assert changed.replace(NEW, OLD) == source
    evidence = OUT / family
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "source.patch").write_text("".join(difflib.unified_diff(source.splitlines(True), changed.splitlines(True), fromfile="common_off/" + family, tofile="scalar_interleave/" + family)))
    folder = BUILD / family
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "input.cu").write_text(changed)
    args = types.SimpleNamespace(source=folder / "input.cu", out=folder, cache=CACHE, sc6=1, define=[flag for flag in selected["flags"] if flag.startswith("-D")], forward_declare_wgs=True)
    namespace["compile_source"](args)
    disassembly = subprocess.run([str(CACHE / "cuda_nvdisasm-linux-x86_64-13.0.85-archive/bin/nvdisasm"), str(folder / "device.cubin")], capture_output=True, text=True, check=True)
    (folder / "device.sass").write_text(disassembly.stdout)
    for name in ("input.cu", "device.ptx", "device.cubin", "device.sass"):
        with gzip.open(evidence / (name + ".gz"), "wb") as output:
            output.write((folder / name).read_bytes())
    for name in ("manifest.json", "ptxas.log", "nvrtc.log"):
        shutil.copyfile(folder / name, evidence / name)
    control = AUDIT / "compiled/public607" / family
    control_sass_path = folder / "control.sass"
    control_sass_path.write_bytes(gzip.decompress((control / "device.sass.gz").read_bytes()))
    before, after = sections(control_sass_path), sections(folder / "device.sass")
    control_manifest = json.loads((control / "manifest.json").read_text())
    treatment_manifest = json.loads((folder / "manifest.json").read_text())
    assert control_manifest["options"] == treatment_manifest["options"]
    record = {"control_source_sha256": selected["cuda_sha256"], "treatment_source_sha256": digest(changed.encode()), "factory_flags": selected["flags"], "control_manifest": control_manifest, "treatment_manifest": treatment_manifest, "entries": {}}
    for symbol, operations in after.items():
        original = before[symbol]
        windows = {}
        for label, stream in (("control", original), ("treatment", operations)):
            windows[label] = [{"instruction_index": index, "instructions": stream[max(0, index - 14):index + 17]} for index, instruction in enumerate(stream) if "MUFU.EX2" in instruction]
        record["entries"][symbol] = {"normalized_sass_equal": original == operations, "control_instructions": len(original), "treatment_instructions": len(operations), "control_opcodes": opcodes(original), "treatment_opcodes": opcodes(operations), "windows": windows}
    control_ptx = gzip.decompress((control / "device.ptx.gz").read_bytes()).decode()
    treatment_ptx = (folder / "device.ptx").read_text()
    record["normalized_ptx_equal"] = normalized_ptx(control_ptx) == normalized_ptx(treatment_ptx)
    (evidence / "ptx.patch").write_text("\n".join(difflib.unified_diff(normalized_ptx(control_ptx), normalized_ptx(treatment_ptx), fromfile="control.ptx", tofile="treatment.ptx")) + "\n")
    result["records"][family] = record
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("COMPLETED", family, treatment_manifest["resources"], flush=True)
result["finished_epoch"] = time.time()
(OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
