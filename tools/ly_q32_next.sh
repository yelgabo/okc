#!/bin/bash
# After the q32n driver: a decode 16 / 32 kernel timeline of the control ly_stk2 (same instrument as tl_q32n), then
# ly_q32u's gate run, chk, two A/B rounds against ly_stk2 and its timeline (tools/ly_d16_ab.sh).
cd /root/okc && . tools/box_env.sh
while pgrep -f "ly_q32_run.sh" > /dev/null; do sleep 5; done
COMMON=(--lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext
        --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json)
LY_TRACE_DIR=.build/d16tl/tl_stk2 python tools/ly_tl_run.py muse_decode_layer muse_decode_layer/cand/ly_stk2.py \
  --loads "decode 16,decode 32" --runs 20 --kernels --pin-pdl8 1,8,16,32 --skip-pdl8-gate "${COMMON[@]}" \
  --out .build/d16tl/tl_stk2.json > .build/d16tl/tl_stk2.log 2>&1
echo "$(date -u +%T) tl stk2 rc=$?"
CTL=ly_stk2.py CAND=ly_q32u.py TAG=q32u bash tools/ly_d16_ab.sh
echo "$(date -u +%T) next done"
