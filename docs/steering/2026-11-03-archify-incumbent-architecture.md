# Incumbent architecture in Archify

November 3, 2026. Source pin `0eb3e5e5b1aec29d81306d6ca568dd2a51c3eedf`.

The retained decoder layer is mapped in two interactive, standalone diagrams:

- [Complete decoder layer](../.archify/architecture-incumbent-20261103-014000/incumbent-layer.html)
- [B4 and MAIN MoE internals](../.archify/architecture-incumbent-20261103-014000/incumbent-moe.html)

The source is `muse_decode_layer/cand/g1b4_stage_release.py`, SHA256 `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`, retained public job `607eb52bdaf24160`. The MoE native-source identity remains `c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71`. Each component links to source lines at the pinned revision. Pending and losing private candidates are excluded.

The overview separates editable whole-layer scope from functional groupings. Dense projections include QKV, O, pooler, router and unpooler. Routing, norms and residual boundaries remain in layer scope. Cross-GPU collectives are outside this benchmark. Stage boxes can share fused launches.

The close-up describes B4 at T17-32 and MAIN above T32. Four compute warpgroups and two producer warps total 576 threads per CTA. The weight and activation rings each have three shared-memory slots. Phase A publishes BF16 intermediate rows in global memory. Phase B consumes the required ready rows through the activation producer. Tiny, Tiny2 and MID use separate warp-specialized paths, so this close-up does not describe T1 or T8.

## Appearance and reproducibility

The user requested Archify, then asked for a plain appearance after reviewing its Classic and Editorial presets. The final diagrams use neutral grays, sans-serif headings and simple arrows. The checked-in `neutral.css` applies to both themes. `build.py` copies the installed Archify package into ignored `.build/archify-neutral`, adds the CSS to its template before rendering, and runs the complete finalizer sequentially. The installed package is unchanged by this styling step.

Candidates, standalone HTML, delivery receipts and final browser receipts are saved together. Rebuild from the repository root with:

```sh
python3 docs/.archify/architecture-incumbent-20261103-014000/build.py --review review-next
```

This requires Archify 3.2.0, Node and its browser-check environment. A new review name preserves earlier evidence. The saved files work without the local review server.

## Evidence and limits

Both final artifacts passed Archify validation, atomic delivery, strict provenance checks and real-browser checks. Final receipts are in `layer-review-neutral-2/` and `moe-review-neutral-2/`. Visual inspection of both final browser views confirmed the neutral styling and readable flows. The saved FFN residual uses a clear three-bend detour. This inspection covered the default desktop view and the narrower layer view; no independent reviewer was available under the project rule against new agents.

The diagrams contain source-derived resource counts and dispatch conditions. Box sizes do not encode time. No new performance result, internal microsecond attribution or optimization conclusion is claimed. Existing performance diagrams are preserved. No kernel files or jobs changed.

The remaining profiling gap is exact-incumbent internal timing with unchanged controls and a measured instrumentation-overhead gate. That work remains with its existing owner. This diagram task ends after publication and remote-byte verification; it authorizes no additional experiment.
