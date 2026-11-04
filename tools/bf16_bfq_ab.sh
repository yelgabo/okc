#!/bin/bash
# Paired A/B of ly_bfq on the box, all judge cells: lt (Lt qkv), bfq0 (wd qkv, attention edge closed), bfq (wd qkv
# plus the PDL attention edge). Round 1 runs the in-prepare PDL8 gate; later rounds skip it.
#   bash tools/bf16_bfq_ab.sh 1 && bash tools/bf16_bfq_ab.sh 2
set -e
cd /root/okc
. tools/box_env.sh
R=$1
SKIP=""
[ "$R" != "1" ] && SKIP="--skip-pdl8-gate"
mkdir -p .build/bf16r3
python tools/ly_arms.py muse_decode_layer muse_decode_layer/cand/ly_bfq.py \
  --judge .build/judge/g1b4_pdl8d.f931abf568544018.json --runs 24 \
  --arm 'lt={"state": {"bfq_on": False}}' --arm 'bfq0={"state": {"bfq_attn": False}}' --arm 'bfq={}' \
  --router-seed 106218864389449120 --pin-pdl8 1,8,16,32 $SKIP \
  --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
  --out .build/bf16r3/bfq_r$R.json > .build/bf16r3/bfq_r$R.log 2>&1
