# Public scout: AutoclickerI publishes its H200 decoder layer

November 3, 2026. Heartbeat scheduled at 14:05 UTC; GitHub and public-page captures began after 16:09 UTC. Instructions read through `95cdfe7c`. This is newly discovered public source, with historical October 27 measurements.

[AutoclickerI/muse-spark-open-kernels](https://github.com/AutoclickerI/muse-spark-open-kernels/tree/85aed1fb1365877b615542dd44980646816d6b26) now publishes its S22 decoder layer, receipts and experiment notes. [Publication commit da928989](https://github.com/AutoclickerI/muse-spark-open-kernels/commit/da928989be2e76eae6d613de73dc7e20b30dc4fc) is dated November 3 at 12:45:17 UTC; the inspected head is `85aed1fb1365877b615542dd44980646816d6b26`. Its one public branch, fork list and releases were checked. No competitor code was executed.

## What is useful

The [readable layer](https://github.com/AutoclickerI/muse-spark-open-kernels/blob/85aed1fb1365877b615542dd44980646816d6b26/final/muse_decode_layer/layer.py) exposes preparation, dispatch, compilation flags and embedded native sources. This gives us a complete H200 composition to inspect, including its losses, rather than isolated leaderboard claims.

- The [small-token MoE source](https://github.com/AutoclickerI/muse-spark-open-kernels/blob/85aed1fb1365877b615542dd44980646816d6b26/final/muse_decode_layer/src/SMALLM_VAR_CUDA.cu) uses 256 threads, eight consumer warps and thread-zero bulk prefetch into a ring of 36 KiB packed-weight tiles. Tensor-core math uses `mma.sync`. The binder selects flag 95 for M1 and 69 for the other configured small-token path. The M1 flags add direct routing-slot dispatch and register-held norm partials; both include dependent launch. Geometry and the selected flag words are source facts, not a performance or ordering qualification.
- Dense projections use split-K partials consumed by subsequent kernels, lossless BF16 coding and PDL. [Attribution](https://github.com/AutoclickerI/muse-spark-open-kernels/blob/85aed1fb1365877b615542dd44980646816d6b26/THIRD_PARTY.md) names Nathen Fernandes's projection code and operatorstack/lee101 attention ports, with retained notices. These mechanisms overlap sources already known here; publication does not make every ingredient new.
- [Attention selection and integration notes](https://github.com/AutoclickerI/muse-spark-open-kernels/blob/85aed1fb1365877b615542dd44980646816d6b26/final/muse_decode_layer/README.md) identify each sliding/full cell. They distinguish attention-call timing from whole-layer leave-one-out timing and disclose a slightly losing decode-32 direct-input hybrid inside the kept composition. Their experiment records are useful comparison material, without transferring those gains to our dispatcher.

## Source and receipt checks

All 75 captured source/document/receipt files match their pinned public Git blobs and SHA256 hashes. An independent, data-only reconstruction joins `layer.py` and all 61 embedded files to the archived wrapper's expanded bytes:

- Wrapper SHA256: `2287496067a3da750d43afd1943c12e69fb1d5337ce1c5e528d4b6c64649a6fd`.
- Expanded source SHA256: `fd5e0ca2ef4bbc1128b04b68aa6802c9543b61808e7663ca5ad10ec4d957110b`, 1,740,697 bytes.

The author's `build.py` was read as data and was not run. No imports, compilation, native execution, numerical validation or timing reproduction occurred.

The [uploaded final receipt](https://github.com/AutoclickerI/muse-spark-open-kernels/blob/85aed1fb1365877b615542dd44980646816d6b26/runs/muse_decode_layer/20261027T013731Z_submit_210d40e0c9cb4b71.json) reports job `210d40e0c9cb4b71`, H200, passed, no flags, **11,973.410 µs / 1.889211x**, CI **[1.860873, 1.917549]**. Its local profile path contains the matching wrapper-hash prefix. The receipt's server `content_hash` uses another representation; that server-to-upload binding has not been independently verified. These are published author records, not our reproduced result.

There is a documentation qualification gap. The [phase-1 results JSON](https://github.com/AutoclickerI/muse-spark-open-kernels/blob/85aed1fb1365877b615542dd44980646816d6b26/problems/muse_decode_layer/dev/s22/stages/p1/results.json) reports a paired ratio of 1.009997, CI [1.009669, 1.010326], but also `freeze_gate: false`, `TASK_pass: false`, `promote: false` and `selected_build_needs_new_confirmation: true`. Its host job `0e1f9bc8474648bc` ran on the BF16 board and deliberately fails setup after an import-time lab. The stage README calls the composition frozen. Keep this lab evidence separate from the later passing decoder-layer receipt; neither a lab-only failure nor its ratio settles full release qualification.

## Coverage and decision

Code search covered 8/5/5/4/2 pages for BF16/attention/MoE/decode/prefill: 1,720 unique file/blob hits. Of 236 newly indexed hits, 234 are from the already inspected Mithun repository and two are workstation documentation. Mithun's actual head diff changes only README wording. Nibague adds learning notes; Kempner changes container lookup/client configuration and unrelated serving recipes. Those changes establish no new challenge kernel gain.

The sweep also covers 32 repository/commit/issue/PR queries, three source-credit queries and 20 indexed web queries. Seven non-code queries stop at 100 results; code totals change between pages. Three transient secondary limits were retried after waits; all final requested search pages succeeded. Exact queries, timestamps, exclusions and captures are retained in the [evidence directory](evidence/2026-11-03-scout-autoclicker-s22-1405/).

The watched SGLang router and vLLM specialized-decode source heads, bodies, merge states and public review lists are unchanged. SGLang's KV-store test PR remains draft with GPU validation pending. Seventeen Guild pages still yield the same 64 normalized public body fragments. Original X profiles remain inaccessible; cached mirrors repeat known announcements, harness posts and rank claims. Known blogs and the LinkedIn participant post add no newly bound kernel receipt. Authenticated Slack/Discord, restricted Guild bodies and encrypted messages remain unread.

Priority stays with the existing short-load owners and their current queue. This publication adds source for a bounded comparison if an owner has an applicable allocation; it creates no new GPU job, adoption decision or public-submission authority. Qualification would require exact selected-source comparison, numerical and publication-order review, then an authorized matched complete-call test with unchanged controls. Stop this scout after publication, remote-byte verification and deduplicated state update.


## Dated selected-dispatch qualification — November3 16:26UTC, root

The originalSMALLM_VAR binder/flag description above is not the final selected body. Data-only inspection of layer.py7320–7326 finds a later override: M1usesRVC_NFIDX0,M8usesRVC_NBASE1,M16/M32useRVC_NBASE2; flags95/69still pass through. Root independently verified the selector and exact source hashes from the retained archive,without foreign execution. M1direct-slot/register-norm work is distinct from our selectedb8path; required BF16/down-scale/publication/progress/lifetime and complete-cost qualification remains open. One source-only reserve review was directly delivered to existing research,without preempting BCLM's first target slot or allocating GPU/public work. [Full source qualification and limits](2026-11-03-driver-monitor-1626.md).
