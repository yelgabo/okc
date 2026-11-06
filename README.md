# okc

Experiments for the [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev). Each
`muse_*` folder is one problem as `okc init` wrote it.
Background and decisions are in [docs/context.md](docs/context.md).

The final decode-layer file is [muse_decode_layer/final/decode_layer.py](muse_decode_layer/final/decode_layer.py)
(SHA256 `1880297e`, best public run `908cc07e43364008`, 2.280x; mean of nine judged runs 2.250x).
Ledgers written before the rename call it `eat_b4` or `muse_decode_layer/cand/eat_b4.py`.
[decode_layer_fixed.py](muse_decode_layer/final/decode_layer_fixed.py) is that file with the prepare-time guard and
fallback fixes from [the final review](docs/decode-layer-final-review.md); it has not been run on a GPU or judged.
The write-up is [docs/paper/okc-decode-layer.pdf](docs/paper/okc-decode-layer.pdf).
Read [current steering](docs/steering/README.md) for active work.
Component `submission.py` files and their fallbacks remain working sources.

```sh
okc check muse_bf16_proj/submission.py
okc submit muse_bf16_proj/submission.py
uv run --with numpy python tools/emulate_router.py
tools/compile_check.sh muse_bf16_proj/submission.py
```

`okc` keeps each run's result in `profiles/` beside the submission; those are committed as the
record of what was tried.

Generated benchmarks are ignored. Use the retained generators when needed; the
[cleanup manifest](docs/steering/evidence/2026-11-02-context-cleanup/manifest.json)
records exact Git recovery commands for retired snapshots. Default `rg` searches skip
historical evidence/candidates. Use `git ls-files` or `rg --files --no-ignore` to include them.

## Engineering references

[H200 lessons from ~460 experiments](docs/research/h200-460-experiments.md) condenses the
user-supplied results: cold-cache benchmarking, streaming versus startup/tail costs,
structural wins, failed variants and CUDA graph traps. Reported measurements are attributed,
not independently reproduced. This reference does not change current work assignments.
