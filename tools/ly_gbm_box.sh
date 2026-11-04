#!/bin/bash
# moe-gbm lane on the box: MFIN on the MAIN (RFIN) MoE build, ly_gbm against ly_stk4.
# PHASE=gate: tools/ly_gb_ab.sh gate (candidate prepare at decode 16 / 32 with the five in-prepare PDL8H guards, then
# chk at all six loads), then tools/ly_gbm_stress.py under gpu.lock (flock(1)) at M = 33 / 128 / 256 plus
# 48 / 97 / 200 / 255. PHASE=ab: tools/ly_gb_ab.sh ab (two paired rounds).
#   ssh okc-h200 'cd /root/okc && PHASE=gate setsid nohup tools/ly_gbm_box.sh > .build/moegb/gbm_gate.log 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
export CAND=ly_gbm.py TAG=gbm CTL=ly_stk4.py
mkdir -p .build/moegb
if [ "$PHASE" = gate ]; then
  PHASE=gate tools/ly_gb_ab.sh
  echo "$(date -u +%T) stress start"
  timeout 5400 flock /root/okc/.build/box/gpu.lock python tools/ly_gbm_stress.py muse_decode_layer/cand/$CTL \
    muse_decode_layer/cand/$CAND --router-seed 106218864389449120 --ms 33,128,256,48,97,200,255 \
    --reps 600,2000,2000,300,300,300,300 > .build/moegb/stress_gbm.log 2>&1
  echo "$(date -u +%T) stress rc=$?"
  grep -E "^STRESS (M=|PASS|FAIL)|Traceback|Error" .build/moegb/stress_gbm.log | cut -c1-400 | head -20
else
  PHASE=ab tools/ly_gb_ab.sh
fi
echo "$(date -u +%T) gbm done"
