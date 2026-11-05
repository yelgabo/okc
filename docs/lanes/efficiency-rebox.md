# efficiency-rebox checkpoint

Updated 2026-11-04 16:57 PST. Job: rebuild replacement box okc-h200b (pod gdmhzavzpcfiex) so every tool works through the `okc-h200` alias.

## Done
- ssh: `~/.ssh/config` Host okc-h200 now 152.236.142.240 port 16377 (backup ~/.ssh/config.bak-rebox). nvidia-smi: H200, driver 595.91.07, 143771 MiB.
- /root/venv (python 3.12.3): torch 2.14.1+cu130, triton 3.8.0, nvidia-cublas 13.1.1.3 (uv, cu130 index). Script /root/setup/torch.sh.
- SGLang a55ab7d5ae1b from `okc serve init` (Mac scratchpad) copied to /root/sglang.

## Running (box, /root/setup/*.log)
- cuda.sh: apt cuda-toolkit-13-0 + rsync.
- sgl.sh: uv editable install of /root/sglang/python with override torch/triton/cublas pins.

## Next
Repo rsync to /root/okc, .build/box and .build/eat dirs, then EAT eval and bx_box_run parity on ly_stk6 at decode 1,verify 1.
