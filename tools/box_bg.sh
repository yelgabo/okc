#!/bin/bash
# usage: tools/box_bg.sh <tag> <box_run args...>; log in .build/box/<tag>.log, JSON in .build/box/<tag>.json
tag=$1; shift
. /root/okc/tools/box_env.sh
mkdir -p .build/box
nohup python tools/box_run.py "$@" --out .build/box/$tag.json > .build/box/$tag.log 2>&1 &
echo "pid $! log /root/okc/.build/box/$tag.log"
