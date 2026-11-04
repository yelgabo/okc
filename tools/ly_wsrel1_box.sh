#!/bin/bash
# moe-wsrel2 lane on the box. PHASE=sass: tools/ly_wsrel1_sass.py (no GPU). PHASE=gate: reference check of
# ${CAND:-ly_wsrel1} at all six loads (--runs 5). PHASE=ab: paired in-process rounds (tools/ly_arms.py) on ly_wsrel1_ab,
# arm base (ly_mpdl's helper) against ar1 (fence.acq_rel after the relay, no per-lane fence) and sc1 (fence.sc) at
# decode 1, verify 1, decode 16. Every python run is under timeout and takes gpu.lock itself.
#   ssh okc-h200 'cd /root/okc && PHASE=ab setsid nohup tools/ly_wsrel1_box.sh > .build/wsrel1/ab.out 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
D=.build/wsrel1
mkdir -p "$D"
C=(--router-seed 106218864389449120 --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext --pin-pdl8 1,8,16,32)
for ph in ${PHASE:-sass gate ab}; do
  if [ "$ph" = sass ]; then
    python tools/ly_wsrel1_sass.py --out "$D/sass" > "$D/sass.log" 2>&1
    echo "$(date -u +%T) sass rc=$?"
  elif [ "$ph" = gate ]; then
    for c in ${CAND:-ly_wsrel1}; do
      timeout 3000 python tools/bx_box_run.py muse_decode_layer muse_decode_layer/cand/$c.py --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
        --loads "decode 1,decode 16,decode 32,verify 1,verify 16,verify 32" --runs 5 "${C[@]}" --out "$D/chk_$c.json" > "$D/chk_$c.log" 2>&1
      echo "$(date -u +%T) gate $c rc=$?"
    done
  else
    for r in ${ROUNDS:-r1 r2}; do
      timeout 3000 python tools/ly_arms.py muse_decode_layer muse_decode_layer/cand/ly_wsrel1_ab.py --loads "decode 1,verify 1,decode 16" --runs ${RUNS:-60} \
        --arm 'base={}' --arm 'ar1={"state": {"wsrel_h": "ar1"}}' --arm 'sc1={"state": {"wsrel_h": "sc1"}}' \
        --skip-pdl8-gate "${C[@]}" --out "$D/ab_$r.json" > "$D/ab_$r.log" 2>&1
      echo "$(date -u +%T) ab $r rc=$?"
    done
  fi
done
echo "$(date -u +%T) all done"
