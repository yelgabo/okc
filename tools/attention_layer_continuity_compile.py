"""Exactly L1/L8 flat control+treatment on installed NVRTC13/PTXAS13."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from attention_layer_continuity import OUT, ROOT


PROBES = '''
extern "C" {
__device__ __constant__ unsigned int okc_layout[] = {
  sizeof(Smem<1, true>), sizeof(Smem<1, false>),
  sizeof(Smem<8, true>), sizeof(Smem<8, false>),
  Cfg<1, true>::NTX, Cfg<8, true>::NTX,
  Cfg<1, true>::STAGES, Cfg<8, true>::STAGES,
  Cfg<1, true>::PREGS, Cfg<1, true>::CREGS,
  Cfg<8, true>::PREGS, Cfg<8, true>::CREGS,
  Cfg<1, true>::MINB, Cfg<8, true>::MINB
};
}
'''


def resources(log):
    frame = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", log)
    registers = re.search(r"Used (\d+) registers", log)
    barriers = re.search(r"used (\d+) barriers", log)
    assert frame and registers and barriers, log
    return dict(stack_bytes=int(frame[1]), spill_store_bytes=int(frame[2]), spill_load_bytes=int(frame[3]),
                registers=int(registers[1]), barriers=int(barriers[1]))


def run(out):
    driver_path = ROOT / "tools/attention_continuity_compile.py"
    original = driver_path.read_text()
    anchor = "    cache = Path('/root/.cache/okc-cuda130-88')"
    assert original.count(anchor) == 1
    adapted = original.replace(anchor, "    device += " + repr(PROBES) + "\n" + anchor)
    namespace = {"__name__": "fixed_attention_layer_continuity_secondary"}
    exec(compile(adapted, str(driver_path), "exec"), namespace)
    result = dict(started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  driver_sha256=hashlib.sha256(original.encode()).hexdigest(),
                  adapted_driver_sha256=hashlib.sha256(adapted.encode()).hexdigest(),
                  adapter="existing declaration-only virtual standard headers and host-prefix removal; emitted sizeof/Cfg constant probes only",
                  probes=PROBES, records=[], scope="secondary CPU-only, no driver or GPU; exactly four selected instances")
    (out / "adapted-compile-driver.py").write_text(adapted)
    summary = out / "compile-result.json"
    try:
        for length in (1, 8):
            for arm in ("control", "treatment"):
                folder = out / f"{arm}-L{length}flat"
                namespace["run"](out / f"{arm}-attention.cu", folder, length)
                manifest = json.loads((folder / "manifest.json").read_text())
                ptx = (folder / "device.ptx").read_text()
                probe = re.search(r"okc_layout\[56\]\s*=\s*\{([^}]+)\}", ptx)
                assert probe, "missing emitted sizeof/Cfg constants"
                layout = [int(value.strip()) for value in probe[1].split(",")]
                log = (folder / "ptxas.log").read_text()
                record = dict(length=length, arm=arm, manifest=manifest, resources=resources(log), layout_probe=layout,
                              ptx_barrier_instructions={token:ptx.count(token) for token in
                                                       ("bar.sync", "barrier.cta", "mbarrier.init", "fence.proxy.async", "wgmma.wait_group")})
                result["records"].append(record)
                summary.write_text(json.dumps(result, indent=2) + "\n")
        pairs = []
        for length in (1, 8):
            control, treatment = [entry for entry in result["records"] if entry["length"] == length]
            worsening = [key for key in ("stack_bytes", "spill_store_bytes", "spill_load_bytes", "barriers")
                         if treatment["resources"][key] > control["resources"][key]]
            assert not worsening, "STOP corresponding resources worsened: " + repr(worsening)
            assert control["layout_probe"][1:] == treatment["layout_probe"][1:], "STOP untargeted shared layout/config changed"
            pairs.append(dict(length=length, resource_worsening=worsening,
                              decode_shared_delta=treatment["layout_probe"][0] - control["layout_probe"][0]))
        result.update(decision="FIXED_SECONDARY_PASS_NOT_TARGET_QUALIFIED", comparisons=pairs)
    except Exception as error:
        result.update(decision="STOP_FIXED_SECONDARY", error=repr(error))
        raise
    finally:
        result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        summary.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    run(parser.parse_args().out)
