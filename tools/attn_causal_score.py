"""Evaluate the registered d1-full-only four-arm attention gate from a receipt.

Uses rounded ABSUM times and official replay calibration, so results are score
sensitivity estimates, not official candidate scores. All unchanged shapes stay
exactly at the primary control. No missing row or missing input is a passing gate.
Verify separately that the owning run used the registered value2 specialization;
applying this diagnostic to value1 does not satisfy value1's broader original gate.
"""
import argparse
import json
import math
from pathlib import Path
import re


def evaluate(path):
    receipt = json.loads(Path(path).read_text())
    assert receipt["board"] == "muse_attention_decode"
    result = receipt["result"]
    rows, inputs, overheads = {}, {}, {}
    for line in result["log_tail"].splitlines():
        m = re.match(r"ABSUM cold b=\s*(\d+)\s+(sliding|full)\s+L=(\d+)\s+(.*?) null:", line)
        if m:
            rows[int(m[1]), m[2], int(m[3])] = {
                k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", m[4])}
        m = re.match(r"ABIN b=\s*1\s+full\s+L=1 in=(\d+)\s+(.*)", line)
        if m:
            inputs[int(m[1])] = {k: float(v) for k, v in re.findall(r"(\w+)=([+-][\d.]+)", m[2])}
    for r in result["records"]:
        if r.get("event") == "benchmark":
            c = r["case"]
            key = c["batch"], c["layer"], 8 if c["load"].startswith("verify") else 1
            overheads[key] = r["replay_overhead"]["overhead_ns"] / 1000
    assert len(rows) == len(overheads) == 12, "missing shape rows"
    assert set(inputs) == {0, 1, 2}, "missing held-out input rows"
    scores = {}
    for arm in ("ctrl", "ctrl2", "late1", "late2"):
        ratios = []
        for length in (1, 8):
            for batch in (1, 16, 32):
                ref = candidate = 0
                for kind, weight in (("sliding", 46), ("full", 16)):
                    key = batch, kind, length
                    c = rows[key]["ctrl"] - overheads[key]
                    a = rows[key][arm] - overheads[key] if key == (1, "full", 1) else c
                    assert min(c, a) > 0
                    ref += weight * c
                    candidate += weight * a
                ratios.append(ref / candidate)
        scores[arm] = math.prod(ratios) ** (1 / 6)
    rnull = math.log(scores["ctrl2"])
    lnull = math.log(scores["late2"] / scores["late1"])
    gain = (math.log(scores["late1"]) + math.log(scores["late2"]) - rnull) / 2
    crosses = {i: {f"{l}/{r}": d[l] - (0 if r == "ctrl" else d[r])
                    for l in ("late1", "late2") for r in ("ctrl", "ctrl2")}
               for i, d in inputs.items()}
    all_faster = all(v < 0 for c in crosses.values() for v in c.values())
    null_gate = gain > 2 * max(abs(rnull), abs(lnull))
    return {"job": receipt["id"], "official_control_passed": result["passed"],
            "full_only_sensitivity_ratios": scores, "pooled_cross_ratio": math.exp(gain),
            "null_ratios": {"control": math.exp(rnull), "candidate": math.exp(lnull)},
            "per_input_candidate_minus_control_us": crosses,
            "all_crosses_faster": all_faster, "twice_null_gate": null_gate,
            "numerical_gate": bool(result["passed"] and all_faster and null_gate),
            "qualification": "rounded medians/borrowed official calibration; not candidate scores; inspect correctness tallies separately"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt")
    print(json.dumps(evaluate(p.parse_args().receipt), indent=2))
