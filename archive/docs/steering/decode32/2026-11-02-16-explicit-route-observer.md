# D32-ATTN-06: explicit call-site routing observer

November 2, 2026 Pacific. Continues [the bounded observer contract](2026-11-02-15-upstream-route-observer.md) after a pre-dispatch rejection. The attention treatment and strict route gate remain unchanged.

## Rejected source and repair

D32-ATTN-05 source `a0998224`, published `0ddbc284`, received HTTP422 `rejected: reads harness frames`. No job ID was accepted. [Complete response](evidence/D32-ATTN-05/dispatch.stdout.json), [dispatch guard](evidence/D32-ATTN-05/dispatch-started.json), [subsequent shared job list](evidence/D32-ATTN-05/post-rejection-jobs.json). The shared list contained no new accepted job or live run. This is a source-policy rejection, without target execution or a measured kernel loss.

The observer used `sys._getframe` to label its own caller. The repair annotates the four actual pool/router call sites with fixed tags derived from their own source definitions. It forwards the original pool/router arguments unchanged and removes frame inspection. The root instruction file now retains this restriction.

## Corrected source and evidence

[Corrected payload](evidence/D32-ATTN-06/observer.py) SHA256 `00193aa2ef2f8cefd22e2949bfd98f90a47ebfef370639dbf7bebd776de07cb4`, 937032 bytes. [Manifest](evidence/D32-ATTN-06/observer.manifest.json). Six real-CPU-Torch tests pass, including full generated preparation/replay, exact forwarding, separate probe owners, same-set slot-swap rejection, unknown layouts, missing boundaries, FP8 diagnostics, malformed reports and explicit call-site labels. [Complete output](evidence/D32-ATTN-06/cpu-tests.txt).

[Source identity audit](evidence/D32-ATTN-06/source-identity.json) again verifies the same five native factory compiler inputs and attention device literals as frozen D32-ATTN-04. No new device code or compiler flags require compilation. The target observer also checks the actual loader's selected B4 source and flags. CPU tests simulate device execution and do not establish target behavior.

The existing one-run hypothesis, independent placements, numerical and exact route checks, boundary observations and stop conditions remain as registered in D32-ATTN-05. Combined-layout logits are CPU reconstructions from retained producer values; they are not claims about exact device sigmoid or selection scores. Tensor retention changes allocation. This source cannot qualify a timing gain and intentionally stops before timing even if all eight fresh replays pass. A mismatch must retain the original failure after recording observations.

Publish and verify these exact bytes, check fresh shared jobs/quota, then submit one private diagnostic. Record the accepted ID immediately and follow only it. Any ambiguous POST error requires shared-list reconciliation. A further source-policy rejection or one inconclusive diagnostic closes this instrument; do not continue blind source retries or replay sweeps.

## Next optimization queue

There is no qualified new D32 kernel treatment after this technical investigation. The precise missing evidence is selected-path exposure at the incumbent's B4 entry, A-to-B handoff or tail, plus a concrete treatment and C32 budget. CUDA Optimizer owns that independent attribution study. The observer will also record the actual T32 pool/router choice, which is required before proposing another fusion. Existing prefix, NS4, scale and grouping closures remain. Research should refill the D32 inbox with a source-qualified mechanism rather than another tuning sweep. This missing optimization evidence does not block the already implemented routing diagnostic.

## Accepted private diagnostic

Own check `8b1e364dafb54c71` was accepted for source `00193aa2`, published in `a7e08ee4`. [Accepted receipt](evidence/D32-ATTN-06/accepted.json) and fresh preflight are archived beside the payload. No jobs were live and 16 private layer checks were available. The existing foreground waiter is exec session `17959`, writing `status-wait.raw.json`. Follow that session and own ID only. Target outcome is pending; neither the rejection nor this acceptance establishes a kernel gain.
