"""Narrow event-scoped completion amendment; no CUDA execution."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import g_pr_graph_owner as prior

BASE = prior.BASE
BASE_SHA = prior.BASE_SHA
EVIDENCE = prior.ROOT / 'docs/steering/evidence/2026-11-02-worker-pr-graph-owner-events'
function = prior.function


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('nonunique event amendment: ' + old[:80])
    return source.replace(old, new, 1)


HOST = once(prior.HOST, '#include <thread>', '#include <thread>\n#include <vector>')
HOST = once(HOST, 'mapped=false, copy_submitted=false;', 'mapped=false;\n  std::vector<CUevent> completions;')
HOST = once(HOST, '      if (copy_submitted) fatal(cuCtxSynchronize(), "completion");',
    '      for (CUevent event : completions) {\n        fatal(cuEventSynchronize(event), "owner-completion");\n        fatal(cuEventDestroy(event), "owner-event-destroy");\n      }')
HOST = once(HOST, '  void before_read(uint64_t stream_bits) {', '  uint64_t before_read(uint64_t stream_bits) {')
HOST = once(HOST, '      require(cuGraphRetainUserObject(graph, mapping->user, 1, 0), "graph-retain");\n    }\n  }',
    '      require(cuGraphRetainUserObject(graph, mapping->user, 1, 0), "graph-retain");\n      return 0;\n    }\n    CUevent event{};\n    require(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "reader-event-create");\n    try { mapping->completions.push_back(event); }\n    catch (...) { fatal(cuEventDestroy(event), "unsubmitted-event-destroy"); throw; }\n    return reinterpret_cast<uint64_t>(event);\n  }\n\n  void after_read(uint64_t stream_bits, uint64_t event_bits) {\n    if (event_bits) fatal(cuEventRecord(reinterpret_cast<CUevent>(event_bits),\n                          reinterpret_cast<CUstream>(stream_bits)), "reader-event-record");\n  }')
HOST = once(HOST, '    owner->copy_submitted=true;',
    '    CUevent copy_event{};\n    require(cuEventCreate(&copy_event, CU_EVENT_DISABLE_TIMING), "copy-event-create");\n    try { owner->completions.push_back(copy_event); }\n    catch (...) { fatal(cuEventDestroy(copy_event), "unsubmitted-copy-event"); throw; }')
HOST = once(HOST, '    require(cuMemcpyDtoDAsync(owner->address,\n            reinterpret_cast<CUdeviceptr>(source.data_ptr()), logical, stream), "bit-copy");\n    require(cuStreamSynchronize(stream), "copy-completion");',
    '    CUresult copy_result=cuMemcpyDtoDAsync(owner->address,\n            reinterpret_cast<CUdeviceptr>(source.data_ptr()), logical, stream);\n    fatal(cuEventRecord(copy_event, stream), "copy-event-record");\n    fatal(cuEventSynchronize(copy_event), "copy-completion");\n    require(copy_result, "bit-copy");')
HOST = once(HOST, '.def("before_read", &Allocation::before_read)',
    '.def("before_read", &Allocation::before_read)\n    .def("after_read", &Allocation::after_read)')

APPEND = once(prior.APPEND, 'name="pr_graph_owner_v1"', 'name="pr_graph_owner_events_v1"')
APPEND = once(APPEND, '        owner.before_read(stream)',
    '        return owner, owner.before_read(stream)\n    return None, 0\n\ndef _pr_graph_after_read(ticket, stream):\n    owner, event = ticket\n    if owner is not None:\n        owner.after_read(stream, event)')


def transforms(original):
    plan, unused_run = prior.transforms(original)
    run = function(original, '_graft_lt_run')
    changed = once(run, '    lt.call("cublasLtMatmul",',
        '    stream = torch.cuda.current_stream().cuda_stream\n    ticket = _pr_graph_before_read(plan, stream)\n    try:\n        lt.call("cublasLtMatmul",')
    changed = once(changed, '    return out',
        '    finally:\n        _pr_graph_after_read(ticket, stream)\n    return out')
    changed = once(changed, '_GP(torch.cuda.current_stream().cuda_stream))', '_GP(stream))')
    return [plan, (run, changed)]


def build(original):
    if hashlib.sha256(original.encode()).hexdigest() != BASE_SHA:
        raise ValueError('not exact ba9 base')
    source = original
    for old, new in transforms(original):
        source = once(source, old, new)
    return source + APPEND.format(host=HOST)


def validate(source, original):
    if source != build(original):
        raise ValueError('not exact event amendment')
    ast.parse(source)
    restored = source[:-len(APPEND.format(host=HOST))]
    for old, new in reversed(transforms(original)):
        restored = once(restored, new, old)
    if restored != original or '\r' in source or not source.endswith('\n') or len(source.encode()) >= 1000000:
        raise ValueError('identity or packaging')
    if 'cuCtxSynchronize' in HOST or 'cuStreamSynchronize' in HOST:
        raise ValueError('encompassing completion remains')
    return dict(sha256=hashlib.sha256(source.encode()).hexdigest(), bytes=len(source.encode()),
        exact_reversal=True, native_sources_unchanged=True, host_syntax_checked=False,
        target_qualified=False, completion='fresh per-eager-use events; graph references for captures')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=EVIDENCE / 'generated')
    args = parser.parse_args()
    original = BASE.read_bytes().decode()
    source = build(original)
    result = validate(source, original)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in [('candidate.py', source), ('host.cpp', HOST),
                        ('generation.json', json.dumps(result, indent=2) + '\n')]:
        (args.out / name).write_bytes(value.encode())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

