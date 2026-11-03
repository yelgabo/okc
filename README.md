# okc

Experiments for the [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev). Each
`muse_*` folder is one problem as `okc init` wrote it, with `submission.py` as our kernel.
Background and decisions are in [docs/context.md](docs/context.md).

```sh
okc check muse_bf16_proj/submission.py   # off-board run on an H200: tests, then timing
okc submit muse_bf16_proj/submission.py  # the same, on the board
uv run --with numpy python tools/emulate_router.py   # CPU check of the router kernel's indexing
tools/compile_check.sh muse_bf16_proj/submission.py  # sm_90a compile in Docker (colima start first)
```

`okc` keeps each run's result in `profiles/` beside the submission; those are committed as the
record of what was tried.

## Engineering references

[H200 lessons from ~460 experiments](docs/research/h200-460-experiments.md) condenses the
user-supplied results: cold-cache benchmarking, streaming versus startup/tail costs,
structural wins, failed variants and CUDA graph traps. Reported measurements are attributed,
not independently reproduced. This reference does not change current work assignments.
