#!/bin/bash
# moe-wsrel lane on the box. PHASE=gate: reference check of ly_wsrel at all six loads (--runs 5), flags as
# tools/ly_mpdl_box.sh. PHASE=ab: paired in-process rounds (tools/ly_arms.py) on ly_wsrel_ab, arm w0 (ly_mpdl's
# helper source, the reference) against sc (ly_wsrel's fence.sc) and ar (ly_wsrel_ar's fence.acq_rel) at decode 1,
# verify 1, decode 16. CAND=ly_wsrel_ar gates the acq_rel file. Every python run is under
# timeout and takes gpu.lock itself.
#   ssh okc-h200 'cd /root/okc && PHASE=gate setsid nohup tools/ly_wsrel_box.sh > .build/wsrel/gate.log 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
D=.build/wsrel
mkdir -p "$D"
C=(--router-seed 106218864389449120 --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext --pin-pdl8 1,8,16,32)
if [ "$PHASE" = gate ]; then
  echo "$(date -u +%T) gate start"
  timeout 3000 python tools/bx_box_run.py muse_decode_layer muse_decode_layer/cand/${CAND:-ly_wsrel}.py --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
    --loads "decode 1,decode 16,decode 32,verify 1,verify 16,verify 32" --runs 5 "${C[@]}" --out "$D/chk_${CAND:-ly_wsrel}.json" > "$D/chk_${CAND:-ly_wsrel}.log" 2>&1
  echo "$(date -u +%T) gate rc=$?"
else
  for r in ${ROUNDS:-r3 r4}; do
    timeout 3000 python tools/ly_arms.py muse_decode_layer muse_decode_layer/cand/ly_wsrel_ab.py --loads "decode 1,verify 1,decode 16" --runs ${RUNS:-60} \
      --arm 'w0={"state": {"wsrel_h": "w0"}}' --arm 'sc={}' --arm 'ar={"state": {"wsrel_h": "ar"}}' \
      --skip-pdl8-gate "${C[@]}" --out "$D/ab_$r.json" > "$D/ab_$r.log" 2>&1
    echo "$(date -u +%T) ab $r rc=$?"
  done
fi
echo "$(date -u +%T) all done"
