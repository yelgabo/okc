#!/bin/bash
# pdl-moe lane on the box. PHASE=gate: ly_mpdl's own prepare (PDL8 gate, then the MPDL_GUARD rounds) and the
# reference check at all six loads (--runs 5), PDL8 pinned at 1, 8, 16, 32. PHASE=ab: two paired in-process
# rounds (tools/ly_arms.py) of edge off (the ly_stk4 launch path) against edge on at T = 1, 8, 16, 32. Every python
# run is under timeout and takes gpu.lock itself.
#   ssh okc-h200 'cd /root/okc && PHASE=gate setsid nohup tools/ly_mpdl_box.sh > .build/pdlmoe/gate.log 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
D=.build/pdlmoe
mkdir -p "$D"
F=muse_decode_layer/cand/ly_mpdl.py
C=(--router-seed 106218864389449120 --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext --pin-pdl8 1,8,16,32)
if [ "$PHASE" = gate ]; then
  echo "$(date -u +%T) gate start"
  timeout 3000 python tools/bx_box_run.py muse_decode_layer "$F" --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
    --loads "decode 1,decode 16,decode 32,verify 1,verify 16,verify 32" --runs 5 "${C[@]}" --out "$D/chk.json" > "$D/chk.log" 2>&1
  echo "$(date -u +%T) gate rc=$?"
else
  for r in ${ROUNDS:-r1 r2}; do
    timeout 3000 python tools/ly_arms.py muse_decode_layer "$F" --loads "decode 1,verify 1,decode 16,decode 32" --runs ${RUNS:-60} \
      --arm 'off={"state": {"moe_pdl": {}}}' --arm 'on={"state": {"moe_pdl": {1: True, 8: True, 16: True, 32: True}}}' \
      --skip-pdl8-gate "${C[@]}" --out "$D/ab_$r.json" > "$D/ab_$r.log" 2>&1
    echo "$(date -u +%T) ab $r rc=$?"
  done
fi
echo "$(date -u +%T) all done"
