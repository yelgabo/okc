#!/bin/bash
# usage: tools/rv_bg.sh <tag> <rv_ab.py args...>; log .build/box/<tag>.log, JSON .build/box/<tag>.json
# Same gpu.lock as tools/box_bg.sh, so in-process A/Bs queue behind the other lanes' runs.
tag=$1; shift
. /root/okc/tools/box_env.sh
mkdir -p .build/box
nohup flock /root/okc/.build/box/gpu.lock python tools/rv_ab.py "$@" --out .build/box/$tag.json > .build/box/$tag.log 2>&1 &
echo "pid $! log /root/okc/.build/box/$tag.log"
