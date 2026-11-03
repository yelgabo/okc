"""Helper-thread-only capture-mode scope on the frozen owner-event amendment."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import g_pr_graph_owner_events as prior

BASE = prior.BASE
BASE_SHA = prior.BASE_SHA
EVIDENCE = prior.prior.ROOT / 'docs/steering/evidence/2026-11-02-worker-pr-graph-owner-relaxed'
function = prior.function
transforms = prior.transforms
once = prior.once

cleanup = prior.HOST[prior.HOST.index('  void cleanup() noexcept {'):prior.HOST.index('\n};')]
changed = once(cleanup.replace('fatal(', 'check('), '  void cleanup() noexcept {', '''  void cleanup(CUstreamCaptureMode* previous=nullptr) noexcept {
    auto check=[previous](CUresult result, const char* stage) noexcept {
      if (result != CUDA_SUCCESS && previous) {
        CUstreamCaptureMode restore=*previous;
        fatal(cuThreadExchangeStreamCaptureMode(&restore), "cleanup-mode-error-restore");
      }
      fatal(result, stage);
    };''')
changed = once(changed, '      check(cuCtxPopCurrent(&popped), "cleanup-pop");',
    '      check(cuCtxPopCurrent(&popped), "cleanup-pop");\n      if (popped != context) check(CUDA_ERROR_INVALID_CONTEXT, "cleanup-pop-identity");')
HOST = once(prior.HOST, cleanup, changed)
HOST = once(HOST, '  mapping->cleanup();\n  delete mapping;',
    '  CUstreamCaptureMode previous=CU_STREAM_CAPTURE_MODE_RELAXED;\n  fatal(cuThreadExchangeStreamCaptureMode(&previous), "cleanup-mode-enter");\n  mapping->cleanup(&previous);\n  fatal(cuThreadExchangeStreamCaptureMode(&previous), "cleanup-mode-restore");\n  delete mapping;')
APPEND = once(prior.APPEND, 'name="pr_graph_owner_events_v1"', 'name="pr_graph_owner_relaxed_v1"')


def build(original):
    source = prior.build(original)
    return once(source, prior.APPEND.format(host=prior.HOST), APPEND.format(host=HOST))


def validate(source, original):
    if source != build(original):
        raise ValueError('not exact helper-mode amendment')
    reverted = once(source, APPEND.format(host=HOST), prior.APPEND.format(host=prior.HOST))
    result = prior.validate(reverted, original)
    ast.parse(source)
    if '\r' in source or not source.endswith('\n') or len(source.encode()) >= 1000000:
        raise ValueError('physical packaging')
    result.update(sha256=hashlib.sha256(source.encode()).hexdigest(), bytes=len(source.encode()),
        capture_mode='helper-only RELAXED with exact previous restore; cleanup errors restore before abort')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=EVIDENCE / 'generated')
    args = parser.parse_args()
    original = BASE.read_bytes().decode()
    source = build(original)
    result = validate(source, original)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in [('candidate.py',source),('host.cpp',HOST),
                        ('generation.json',json.dumps(result,indent=2)+'\n')]:
        (args.out/name).write_bytes(value.encode())
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

