#!/bin/bash
# Slack-free rebuild A/B on the box: pdl8hn, combo2n and pdl8d in one session, interleaved, each a separate
# bx_box_run.py process under gpu.lock, all six loads, no PDL pin (each file's own gate or guard decides).
#   ssh okc-h200 'cd /root/okc && nohup tools/ly_slackfree_ab.sh > .build/box/ly_ab.log 2>&1 &'
cd /root/okc && . tools/box_env.sh
C=muse_decode_layer/cand
run() {
  python tools/bx_box_run.py muse_decode_layer "$C/$2" --baseline \
    --judge .build/judge/g1b4_pdl8d.f931abf568544018.json --router-seed 106218864389449120 --runs 60 \
    --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
    --out ".build/box/$1.json" > ".build/box/$1.log" 2>&1
  local rc=$?
  echo "$(date -u +%T) $1 $2 rc=$rc"
}
# ROUNDS / ARMS override the defaults, e.g. ROUNDS="c d e" ARMS="hn c2n"
declare -A FILE=([hn]=g1b4_pdl8hn.py [c2n]=rv_k_combo2n.py [d]=g1b4_pdl8d.py)
for r in ${ROUNDS:-a b}; do
  for arm in ${ARMS:-hn c2n d}; do
    run "ly_${arm}_$r" "${FILE[$arm]}"
  done
done
