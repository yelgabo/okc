#!/bin/bash
# usage: tools/box_bg.sh <tag> <box_run args...>; log in .build/box/<tag>.log, JSON in .build/box/<tag>.json
# One GPU, several lanes: flock on gpu.lock queues runs instead of sharing the card.
tag=$1; shift
. /root/okc/tools/box_env.sh
mkdir -p .build/box
nohup flock /root/okc/.build/box/gpu.lock python tools/box_run.py "$@" --out .build/box/$tag.json > .build/box/$tag.log 2>&1 &
echo "pid $! log /root/okc/.build/box/$tag.log (waits for gpu.lock if another run holds it)"
