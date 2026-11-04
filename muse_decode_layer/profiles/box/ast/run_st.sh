#!/bin/bash
cd /root/okc && . tools/box_env.sh >/dev/null 2>&1
for f in "$@"; do
  python tools/ly_box_ast.py muse_decode_layer/cand/ly_ast_$f.py --judge .build/judge/g1b4_pdl8d.f931abf568544018.json --router-seed 106218864389449120 --loads "decode 1" --layers "${LAYERS:-sliding}" --inputs 20 --lock /root/okc/.build/box/gpu.lock --ls-cache .build/box/ls_cache.json --out .build/ast/ast_$f.json > .build/ast/ast_$f.log 2>&1
  echo "$(date -u +%T) $f rc=$?"
done
