# D32-ATTN-01 implementation plan

Implement the [assigned D32 full-attention track](2026-11-02-01-full-decode-attention.md)
against control ba9. Preserve the complete layer and every other standard route.

1. Resolve L1-flat's selected producer/consumer loop and prior closures. Compare
   within-request next-tile overlap with the existing cross-request continuity
   treatment. Select one changed mechanism and explain its storage lifetimes.
2. Add `tools/d32_attn_worker_candidate.py` and focused tests. Generate one plain
   full-layer candidate under `evidence/D32-ATTN-01/`. Build its separate module
   during preparation and select it through metadata only. Test actual final
   bindings, unsupported shapes/scales, missing preparation and direct-entry
   guards, with all other standard cells retaining control.
3. Compile the exact selected device instance with an installed local compiler.
   Preserve the host launcher, ordering and numerical boundaries. Inspect the
   resulting resource report and instructions. Record limitations of device-only
   compilation and any compiler adapter separately from kernel failures.
4. Obtain an independent source/lifetime review, resolve required findings and
   rerun affected checks. Publish candidate, hashes, generation command and
   Driver's remaining target checks. Re-read the inbox between these steps.

Stop on a concrete math, lifetime or progress contradiction. A valid candidate
goes to Driver for one ordinary full-layer exploration. Speed, target numerical
qualification and actual H200 admission remain unresolved until that measurement.
