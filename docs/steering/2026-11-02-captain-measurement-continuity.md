# Measurement continuity: existing CPU runtime and separate MoE capacity

November 2, 2026, after parent `e1989d2` and correction `1f59b96`. This reconciles the latest handoff; it creates no overlapping implementation, environment migration, policy exception or target allocation. Top100 remains unachieved; no new GPU performance result exists.

## Verified dependency scope

Captain independently invoked the **previously installed** `/root/.cache/okc-host-validate/bin/python` under the existing Ubuntu WSL distribution. Torch **2.10.0+cpu** imported successfully. Real tensors and views share storage; a view mutation advanced the original tensor's version from0 to1. Inference-mode tensors were created/read successfully and accessing their version counter raised the documented runtime exception. Exact argv, stdout, return0 and scope are in `evidence/2026-11-02-captain-existing-torch-probe/probe.json`.

No package was installed, no Windows Application Control setting changed, no blocked Windows Torch DLL executed, and no GPU/Driver call occurred. This is the same Linux CPU environment retained in the earlier captain host receipts, not a new workaround for the Windows4551 error. The earlier default-interpreter probes and isolated Windows result did not cover this environment. Parent independently recovered it and published that correction in `1f59b96`; the findings agree.

This establishes an executable **real-storage host-test environment**, not target Torch version/ABI equivalence. It does not turn the earlier CPU2.10 C++17 syntax pass into valid target compatibility evidence. In particular, inference/version-cache correctness needs execution of the real generated entry under both caller modes; merely proving import or testing stand-ins is insufficient. The current harness explicitly disables inference mode inside its private scope, which must be exercised rather than ignored when testing that version behavior.

## Current implementation and ownership

The full layer harness is still the held918254B/c6f4c25d draft, with its unconditional prepare hold intact. Parent owns the newly bounded real-Torch facade validation and concrete fixes. It has not completed here. Do not remove the hold, relabel the15 facade tests as the full storage gate, or start another validator. Real CUDA replay, target builds, preparation fit and frozen whole-six/null/regression gates remain separate.

Parent also owns the separately bounded **exact standalone MoE adapter** from `a58f058`. The saved read-only receipt reports ten MoE private checks while layer has zero. Capacity is problem-specific and must be freshly reconciled before any future POST; five unused scored slots are not private capacity or public permission. Adapter identity, global/local expert mapping, weight/scale slicing, supported shapes and actual fallback semantics must first survive its own source/host review. Existing public standalone source is not automatically the control.

## Next useful evidence and decision

Continue the two already allocated tasks without extending T1 or reopening PR, attention, G224, SiLU, serial or RMW variants. T1 remains frozen at the independently reviewed faed74a8 prototype; the worker is finished. Parent's candidate dff221ae remains unchanged and source/CPU accepted only.

An exact standalone adapter can make available capacity useful for target component compilation/numerics and a separately registered matched component discriminator. It cannot replace the complete layer outcome or transfer its score: routing adaptation, extra work, baseline and workload weighting must be reported in that board's own terms, not the layer's46:16 formula. A failed exact component may avoid a wasted layer slot; a component win is only a reason to test the whole layer later.

No dispatch is authorized by this reconciliation. Captain retains the single dispatch decision, parent/delegates retain their source tasks, and existing worker remains the only intended job owner after a later explicit allocation. Receive the exact adapter/harness artifacts or scoped STOPs next; do not substitute another source survey or generic readiness report. Evidence that changes priority is an actual contract blocker, qualified target result or complete-call measurement, not available slot count alone.
