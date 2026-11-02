"""Source-only counterexample model, not a CUDA runtime reproduction."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
source = (ROOT / 'tools/g_pr_graph_owner.py').read_text()
assignment = next(node for node in ast.parse(source).body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'HOST'
                          for target in node.targets))
host = ast.literal_eval(assignment.value)


def retirement_model(strategy):
    events = ['A eager read submitted', 'A host owner retired',
              'B independent stream capture starts', 'A helper resumes']
    if strategy == 'context':
        return dict(events=events + ['A helper requests context synchronization'],
                    context_contains_active_capture=True,
                    operation_permitted_by_capture_contract=False)
    return dict(events=events + ['A helper waits only A noncaptured completion event'],
                context_contains_active_capture=True,
                targets_unrelated_capture=False,
                operation_permitted_by_capture_contract=True,
                assumption='event was recorded after A eager read outside capture')


old = retirement_model('context')
proposed = retirement_model('owner-event')
assert not old['operation_permitted_by_capture_contract']
assert proposed['operation_permitted_by_capture_contract']
print(json.dumps(dict(
    inspected_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    inspected_host_sha256=hashlib.sha256(host.encode()).hexdigest(),
    context_sync_present='cuCtxSynchronize()' in host,
    event_record_present='cuEventRecord(' in host,
    context_sync_counterexample=old,
    proposed_event_scope_model=proposed,
    scope='Synthetic contract schedule only. No actual CUDA/driver/compiler call or observed judge failure.',
    event_protocol_implemented_or_qualified=False), indent=2))
