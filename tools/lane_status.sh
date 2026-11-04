#!/bin/sh
# Snapshot of what the box lanes are doing: GPU processes, lock holders, the newest box logs with their
# bandwidth / span / verdict lines, and the lane branches' latest commits. Run from the repo root.
mins=${1:-40}
echo "== $(TZ=America/Los_Angeles date '+%H:%M PT')  box okc-h200"
ssh -o ConnectTimeout=20 -o BatchMode=yes okc-h200 "
  echo '-- GPU processes'; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  echo '-- flock on gpu.lock'; pgrep -af 'flock .*gpu.lock' | grep -v pgrep | cut -c1-140
  echo '-- logs touched in the last $mins min (newest first)'
  find /root/okc/.build /root/okc/muse_moe_w4a16/cl /root/okc/muse_decode_layer/profiles -type f -name '*.log' -mmin -$mins 2>/dev/null \
    | xargs -r ls -t | head -12 | while read f; do
      echo \"== \$f (\$(date -r \$f +%H:%M))\"
      grep -iE 'TB/s|GB/s|ratio|floor|span|median|delta|verdict|PASS|FAIL|Traceback' \"\$f\" | tail -3 | cut -c1-180
    done
" 2>&1
echo "-- lane branches (origin)"
git fetch -q origin 2>/dev/null
for b in $(git branch -r --list 'origin/worktree-agent-*' | sed 's#origin/##'); do
  ts=$(git log -1 --format=%ct origin/$b 2>/dev/null); now=$(date +%s)
  [ -n "$ts" ] && [ $((now - ts)) -lt $((3 * 3600)) ] && { echo "-- $b"; git log --format='   %h %ar %s' origin/$b -2; }
done
