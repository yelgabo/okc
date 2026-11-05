# efficiency-rebox checkpoint

Updated 2026-11-04 17:06 PST. Job: rebuild replacement box okc-h200b (pod gdmhzavzpcfiex) so every tool works through the `okc-h200` alias.

## Done
- ssh: `~/.ssh/config` Host okc-h200 now 152.236.142.240 port 16377 (backup ~/.ssh/config.bak-rebox). nvidia-smi: H200, driver 595.91.07.
- Toolkit: apt cuda-toolkit-13-0, nvcc 13.0.88 (V13.0.88), /usr/local/cuda -> /usr/local/cuda-13.0, ncu present.
- /root/venv: torch 2.14.1+cu130, triton 3.8.0, nvidia-cublas 13.1.1.3, sglang 0.0.0.dev1+ga55ab7d5a editable at /root/sglang/python, sglang-kernel 0.4.9, flashinfer-python 0.7.0.post1, apache-tvm-ffi 0.1.11, numpy 2.3.5.
- Repo: /root/okc = git ls-files of main 0faf0b62 + .build/repeats/*_base.py + .build/judge/g1b4_pdl8d.{f931abf568544018,4a737387f88b4d7d}.json; .build/box and .build/eat created.
- docs/box.md: dated Reaching-it note and a "Rebuilding the box" recipe.

## Running (box)
- Parity: ly_stk6.py (main 0faf0b62) --baseline decode 1,verify 1, runs 60, router seed 106218864389449120, pin-pdl8 1,16,32 --skip-pdl8-gate; log /root/okc/.build/box/rebox_stk6.{log,json}. First lines show "G1b4 backend native repaired".
- EAT eval of ly_mpdl (sha256 prefix a0ee638cf5342235) from detached worktree /Users/gretchengabo/GitHub/okc-eat-rebox at c7713684; box tag eval_a0ee638cf534_1793840782, queued on gpu.lock.

## Next
Record per-cell net times and EAT score in docs/box.md "Parity on okc-h200b, 2026-11-04"; report.
