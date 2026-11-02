"""Prepare-end selection receipt only, on frozen graph-owned PR source."""

import argparse
import hashlib
import json
from pathlib import Path

import g_pr_graph_owner_relaxed as prior

BASE = prior.BASE
BASE_SHA = prior.BASE_SHA
HOST = prior.HOST
function = prior.function
transforms = prior.transforms
EVIDENCE = BASE.parents[2] / 'docs/steering/evidence/2026-11-02-worker-pr-target'
PREFIX_SHA = '4c6d3f13e0b9c72c4d4ab67f440a35f6ae39a954c818f78d35f3ff3d9090423b'
APPEND = '''

_pr_graph_receipt_prepare = prepare
def prepare(weights):
    state = _pr_graph_receipt_prepare(weights)
    import json as _pr_receipt_json
    accepted = []
    for tokens in (128, 256):
        plan = state.get("graft_lt", {}).get(("pr", tokens))
        owner = plan.get("_pr_graph_owner") if plan is not None else None
        if owner is not None and plan.get("weight") is owner.tensor:
            accepted.append(dict(tokens=tokens, shape=list(owner.tensor.shape),
                                 dtype=str(owner.tensor.dtype), owner_tagged=True))
    status = state.get("_pr_graph_status", "missing-status")
    outcome = "ACTIVE_TREATMENT" if status == "generic-owned" and len(accepted) == 2 else "NO_TREATMENT"
    print("PR_GRAPH_PREPARE_RECEIPT " + _pr_receipt_json.dumps(dict(schema_version=1,
          status=status, outcome=outcome, accepted_pr_plans=accepted), sort_keys=True,
          separators=(",", ":")), flush=True)
    return state
'''


def build(original):
    prefix = prior.build(original)
    if len(prefix.encode()) != 881883 or hashlib.sha256(prefix.encode()).hexdigest() != PREFIX_SHA:
        raise ValueError('not frozen source')
    return prefix + APPEND


def validate(source, original):
    if source != build(original) or not source.endswith(APPEND):
        raise ValueError('not exact receipt-only transform')
    result = prior.validate(source[:-len(APPEND)], original)
    if len(source.encode()) >= 1000000 or '\r' in source or not source.endswith('\n'):
        raise ValueError('physical packaging')
    result.update(sha256=hashlib.sha256(source.encode()).hexdigest(), bytes=len(source.encode()),
        frozen_prefix_sha256=PREFIX_SHA, host_sha256=hashlib.sha256(HOST.encode()).hexdigest(),
        receipt_schema_version=1, receipt_only_append=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=EVIDENCE / 'generated')
    args = parser.parse_args()
    original = BASE.read_bytes().decode()
    source = build(original)
    result = validate(source,original)
    args.out.mkdir(parents=True, exist_ok=True)
    for name,value in [('candidate.py',source),('receipt-append.py.txt',APPEND),
                       ('generation.json',json.dumps(result,indent=2)+'\n')]:
        (args.out/name).write_bytes(value.encode())
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

