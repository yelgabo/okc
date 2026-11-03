# Evaluator delta: declared workload equivalence, unresolved configuration semantics

November 2, 2026, 19:18 UTC. Evidence base `50921378` after attention allocation `48111ace` and T1 closure `5f687801`. This is a bounded offline receipt/source-access audit while the existing worker implements isolated attention continuity. No judge, GPU, Driver, compiler, submission or default action. The user's net full-layer ROI priority in [the latest report](2026-11-02-t1-layer-roi-and-review-status.md) remains binding.

## Outcome and decision

The historical baseline/public and fresh incumbent receipts declare the same twelve benchmark cases **after excluding only seed**. This includes the reported workload configuration fields, repetitions and layer weights. Their evaluator manifests have identical path sets, with nine identical file hashes and exactly one changed file hash: `configs/muse-spark-open.yaml`. The baseline, reference, task, checks, NVFP4 helper and four harness file hashes agree. Evaluator aggregate code and image identities differ. This narrows drift uncertainty; it does not resolve the config change or establish equivalent sampled tensors, dependency versions or baseline kernel selection.

No config bytes were recovered. No new candidate, readiness pass, baseline replay or target allocation follows. Do not attribute historical score drift to our source, this config, a driver or any individual baseline kernel. A fresh baseline-as-submission job is not justified merely to fill this metadata gap while layer private capacity is exhausted and a source treatment is in progress.

## Owning evidence and reproduction

- Public incumbent `607eb52bdaf24160`, mode `run`: `muse_decode_layer/profiles/scored.607eb52bdaf24160.json`.
- Fresh identical incumbent `cc474883117947c0`, mode `check`: `docs/steering/evidence/2026-11-02-captain-fresh-incumbent-profile/control.cc474883117947c0.json`.
- Historical baseline-as-submission `7d42771c3384449d`, mode `check`: `muse_decode_layer/baseline_profile/status.7d42771c3384449d.json`.

Executable [audit](../../../docs/steering/evidence/2026-11-02-captain-evaluator-delta/audit.py) reuses the existing strict owning receipt validator, validates job/mode/completeness and reconstructed scores, asserts the identical public/fresh server source identity, and compares full reported cases with only `seed` removed. Python JSON parsing preserves integer seeds; no JavaScript-number conversion is used. [Output](../../../docs/steering/evidence/2026-11-02-captain-evaluator-delta/audit.json) retains input SHA256s, complete evaluator manifests, cases and explicit limitations.

Command: `wsl -d Ubuntu --exec /root/.cache/okc-host-validate/bin/python /mnt/c/Users/gretc/Documents/GitHub/okc/docs/steering/evidence/2026-11-02-captain-evaluator-delta/audit.py`. Final run exits zero with twelve matching declared cases, three validated owning receipts and exactly one changed manifest path. This is offline evidence validation, not a fresh GPU run. During development an incorrect normalized-record field access and an overbroad seed-inclusive equality assertion failed before output; both were corrected to the actual validator schema and explicitly seed-excluded scope. No failed development run is qualification.

Config hashes: historical `98f07230e4e64fe91b7c76d9a3ce04948053c043d494d5f1fae56cfe2de033af`; fresh `60e242544d5f5a5fc143870461f7373b85accb677a536cb337581ded938c4a48`. These are SHA256s, not retrievable Git commit or blob identities. The receipt supplies no repository/ref or config contents. The image is an evaluator-local registry reference; it was not treated as a public source endpoint or permission to access that registry.

## Bounded source-access scope

Confirmed checkout/remote `C:/Users/gretc/Documents/GitHub/okc`, `https://github.com/yelgabo/okc.git`. Inventoried tracked paths with `git ls-files` and widened relevant filename inventory with `rg --files --hidden --no-ignore`, including ignored `.build` paths and excluding `.git`. No exact config filename was returned in that checkout scope. A missing local `muse_decode_layer/task.yml` was an explicit path error, not evidence the task was absent: the owning evaluator task hash and local `TASK.md` remain distinct artifacts.

Authenticated GitHub code search used `filename:muse-spark-open.yaml` and the narrower `filename:muse-spark-open.yaml path:configs`. [Search receipt](../../../docs/steering/evidence/2026-11-02-captain-evaluator-delta/search.json) records successful responses, timestamps, incompleteness flags and returned path/repository/blob identities. The broad query returned eight unrelated model/provider metadata paths; the `configs` query returned zero. Neither establishes that evaluator sources do not exist or that a private/unindexed repository is absent. Web search tools returned no usable response, so no web-search coverage is claimed. No external account, cache outside the checkout, private registry or contact was probed.

## Priorities, owner and next gates

1. Existing worker retains the single source/CPU/fixed-secondary attention allocation. No new message, duplicate implementation or target authorization is needed from this audit. Its preserved math, lifetime, resources and exact dispatch must earn independent captain review.
2. Before allocating a later target check, enforce `50921378`: credible net six-load layer benefit, exposed-time evidence rather than activity sums, replacement/shared-resource costs, strongest alternative and finite stop rule. A one-percent standalone component gate alone is insufficient. The current source allocation does not bypass that later ROI decision.
3. Config investigation stops at this source-access gap. Reopen only on actual old/new bytes with matching hashes or an authoritative source/ref or semantic change record. Do not repeat unchanged filename inventories, spend a check merely on this gap, or infer an implementation cause from the aggregate image hash.
4. A later authorized paired treatment instrument must keep controls/candidates under one current evaluator and matched inputs; current historical SGLang activities cannot become fresh operation-level attribution. A standalone win still cannot establish a combined-layer win.

The stale acceptance handoff for `56e0c8b5634545e6` is already superseded by its archived valid negative and captain closure; it triggers no restarted wait, POST or retry. Top100 is not achieved. Public incumbent remains unchanged and there is no new performance-qualified winner from this audit.
