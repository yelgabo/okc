"""Readback audit of the concurrently published allocation, no compile or dispatch."""
import ast
import hashlib
import json
from pathlib import Path
import re
import time

from attention_continuity import generate
from attention_layer_continuity import BASE, OUT, ROOT
from attention_top100_screen import function
from test_attention_layer_continuity import Tensor, config, environment, invoke


PUBLISHED = ROOT / "docs/steering/evidence/2026-11-02-attention-continuity"


def run():
    candidate = (PUBLISHED / "candidate.py").read_text()
    base = BASE.read_text()
    regenerated, control, treatment, native_diff = generate(base)
    assert candidate == regenerated
    tree = ast.parse(candidate)
    archive = json.loads((PUBLISHED / "archive-manifest.json").read_text())
    verified = []
    for name, pin in archive.items():
        raw = (ROOT / name).read_bytes()
        assert len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"], name
        verified.append(name)
    records = []
    cases = [(batch, length, window, 1) for length in (1, 8) for batch in (1, 16, 32) for window in (2048, 0)]
    cases += [(33, 1, 0, 1), (4, 1, 2048, 64), (4, 8, 0, 64), (16, 1, 0, 64)]
    for batch, length, window, page in cases:
        captures, controls, owners = [], [], []
        env, original_module, continuous_module = environment(candidate, captures, owners)
        function(tree, "_attention_continuity_module", env)
        env["_prepare_attention_modules7"] = lambda state: state.update(attention_continuity_module=continuous_module)
        reference, _, _ = environment(base, controls, [])
        cfg = config(batch, length, window, page)
        state, data, output = invoke(env, cfg)
        invoke(reference, cfg)
        assert captures[-1][1] == controls[-1][1]
        target = batch in (16, 32) and window == 0 and page == 1
        assert captures[-1][0] == ("continuity" if target else "control")
        assert owners[-1] is state["attention_scratch"]
        records.append(dict(batch=batch, length=length, window=window, page=page,
                            actual_final_kernel_module=captures[-1][0], full_argument_tuple_equal=True))
    captures, owners = [], []
    env, original_module, continuous_module = environment(candidate, captures, owners)
    function(tree, "_attention_continuity_module", env)
    env["_prepare_attention_modules7"] = lambda state: state.update(attention_continuity_module=continuous_module)
    state, data, output = invoke(env, config(33, 1, 0))
    assert captures[-1][0] == "control"
    direct_rejected = False
    try:
        env["_launch7"](data, _mod=continuous_module, attention_state=state, pdl=False)
    except ValueError:
        direct_rejected = True
    assert not direct_rejected and captures[-1][0] == "continuity"
    resource_rows = []
    for length in (1, 8):
        arms = []
        for arm in ("control", "treatment"):
            folder = PUBLISHED / f"{arm}-L{length}"
            manifest = json.loads((folder / "manifest.json").read_text())
            assert manifest["nvrtc_status"] == manifest["ptxas_status"] == 0
            for name, key in (("device.cu", "device_sha256"), ("device.ptx", "ptx_sha256"), ("device.cubin", "cubin_sha256")):
                assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == manifest[key]
            log = (folder / "ptxas.log").read_text()
            stack = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", log)
            registers = re.search(r"Used (\d+) registers, used (\d+) barriers", log)
            arms.append(dict(arm=arm, registers=int(registers[1]), barriers=int(registers[2]),
                             stack_spill_bytes=[int(value) for value in stack.groups()]))
        assert arms[0]["registers"] == arms[1]["registers"]
        assert arms[0]["barriers"] == arms[1]["barriers"]
        assert arms[0]["stack_spill_bytes"] == arms[1]["stack_spill_bytes"]
        resource_rows.append(dict(length=length, resources=arms))
    result = dict(observed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  publication_commits=["19ab26cdda573a57db020dfa84decfe4c7fea52b", "6e1811647d0f525be53a0ebf28bc98e9ca10731e"],
                  candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(), candidate_lf_bytes=len(candidate.encode()),
                  archive_paths_verified=verified, actual_final_kernel_cases=records,
                  resources_reused_not_recompiled=resource_rows,
                  negative=dict(case="B33 L1 full P1 direct _launch7 explicit prepared continuity module",
                                final_kernel_correctly_selected_control=True, direct_entry_rejected=direct_rejected,
                                direct_entry_ran_continuity=True,
                                scope="Python stand-in source-scope guard violation, not an observed GPU corruption"),
                  decision="NO_DUPLICATE_CANDIDATE_OR_COMPILER_RETRY; CANONICAL_DIRECT_ENTRY_SCOPE_GAP_FOR_CAPTAIN")
    (OUT / "published-reconciliation.json").write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps({key: value for key, value in result.items() if key not in ("archive_paths_verified", "actual_final_kernel_cases")}, indent=2))


if __name__ == "__main__":
    run()
