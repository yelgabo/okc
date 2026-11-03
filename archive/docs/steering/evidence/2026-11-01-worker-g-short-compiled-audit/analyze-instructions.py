from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
from prod_size_compile_g import sections

OUT = Path(__file__).parent
BUILD = ROOT / ".build/worker-g-short-audit"
result = {"scope": "Static selected SASS instruction counts/dependencies, not dynamic executions, latency, exposed cycles, or a cost attribution", "entries": {}, "comparisons": {}, "dependencies": {}}

def opcode(operation):
    return re.sub(r"^@!?(?:U?P)\d+\s+", "", operation).split()[0].rstrip(";")

for label in ("donor", "public607"):
    for family in ("tiny", "tiny2"):
        folder = BUILD / label / family
        operation_sections = sections(folder / "device.sass")
        entry_records = {}
        for symbol, operations in operation_sections.items():
            counts = Counter(opcode(operation) for operation in operations)
            entry_records[symbol] = {"instructions": len(operations), "non_nop": len(operations) - counts["NOP"], "opcodes": dict(sorted(counts.items()))}
            if "ILb0E" in symbol:
                windows = []
                for index, operation in enumerate(operations):
                    if any(key in operation for key in ("MUFU.EX2", "FENCE.VIEW.ASYNC.S", "ATOM.E.CAST.SPIN", "MEMBAR.ALL.GPU")):
                        windows.append({"instruction_index": index, "operation": operation, "start_index": max(0, index - 7), "instructions": operations[max(0, index - 7):index + 9]})
                result["dependencies"][label + "/" + family] = windows
        result["entries"][label + "/" + family] = entry_records
        evidence = OUT / "compiled" / label / family
        evidence.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.json", "ptxas.log", "nvrtc.log"):
            shutil.copyfile(folder / name, evidence / name)
        for name in ("device.ptx", "device.sass"):
            (evidence / (name + ".gz")).write_bytes(gzip.compress((folder / name).read_bytes(), mtime=0))
for family in ("tiny", "tiny2"):
    donor = result["entries"]["donor/" + family]
    repaired = result["entries"]["public607/" + family]
    deltas = {}
    for symbol in donor:
        before, after = donor[symbol], repaired[symbol]
        deltas[symbol] = {"non_nop_delta": after["non_nop"] - before["non_nop"], "instruction_delta": after["instructions"] - before["instructions"], "opcode_delta": {name: after["opcodes"].get(name, 0) - before["opcodes"].get(name, 0) for name in sorted(before["opcodes"].keys() | after["opcodes"].keys()) if after["opcodes"].get(name, 0) != before["opcodes"].get(name, 0)}}
    result["comparisons"][family] = deltas
(OUT / "instruction-audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"counts": {label: {symbol: {key: value for key, value in entry.items() if key != "opcodes"} for symbol, entry in entries.items()} for label, entries in result["entries"].items()}, "comparisons": result["comparisons"]}, indent=2))
