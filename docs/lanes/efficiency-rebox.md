# efficiency-rebox checkpoint

Updated 2026-11-04 17:15 PST. Job: rebuild replacement box okc-h200b (pod gdmhzavzpcfiex) so every tool works through the `okc-h200` alias. STATUS: step 4 passed.

## Done
- ssh: `~/.ssh/config` Host okc-h200 now 152.236.142.240 port 16377 (backup ~/.ssh/config.bak-rebox). nvidia-smi: H200, driver 595.91.07.
- Toolkit: apt cuda-toolkit-13-0, nvcc 13.0.88 (V13.0.88), /usr/local/cuda -> /usr/local/cuda-13.0, ncu present.
- /root/venv: torch 2.14.1+cu130, triton 3.8.0, nvidia-cublas 13.1.1.3, sglang 0.0.0.dev1+ga55ab7d5a editable at /root/sglang/python, sglang-kernel 0.4.9, flashinfer-python 0.7.0.post1, apache-tvm-ffi 0.1.11, numpy 2.3.5.
- Repo: /root/okc = git ls-files of main 0faf0b62 + .build/repeats/*_base.py + .build/judge/g1b4_pdl8d.{f931abf568544018,4a737387f88b4d7d}.json; .build/box and .build/eat created.
- docs/box.md: dated Reaching-it note and a "Rebuilding the box" recipe.

## Results
- EAT eval ly_mpdl (a0ee638cf534, tag eval_a0ee638cf534_1793840782): SCORE 2.32968, 12/12 PASS, native G1b4, no Marlin (old pod 2.3578/2.3531/2.3559). Timed phase overlapped cold compiles of the first ly_stk6 run.
- ly_stk6 parity (decode 1,verify 1, runs 60): net 65.36 / 71.06 / 110.30 / 119.57 us (d1 sliding/full, v1 sliding/full), all PASS, native G1b4. Log /root/okc/.build/box/rebox_stk6.log.
- First ly_stk6 run failed cold: 354 s of builds > 295 s self-test budget (log rebox_stk6_cold_fail.log). Pod cgroup is 13.6 CPUs.
- docs/box.md: "Parity on okc-h200b, 2026-11-04".

## Next
Optional quiet rerun of the EAT eval to separate the 1.1% gap from compile load. Remove worktree /Users/gretchengabo/GitHub/okc-eat-rebox when done.
