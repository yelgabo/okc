#!/bin/sh
# Snapshot of what the box lanes are doing: GPU processes, lock holders, the newest box logs with their
# bandwidth / span / verdict lines, and the lane branches' latest commits. Run from the repo root.
mins=${1:-40}
echo "== $(TZ=America/Los_Angeles date '+%H:%M PT')  box okc-h200"
ssh -o ConnectTimeout=20 -o BatchMode=yes okc-h200 "
  echo '-- GPU processes (pid, elapsed, memory, command; HUNG? = older than 30 min with no log written in the last 20; NOLOCK = no flock gpu.lock ancestor, its timings and everyone else'"'"'s are suspect)'
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=, read pid mem; do
    et=\$(ps -o etimes= -p \$pid | tr -d ' '); cmd=\$(ps -o args= -p \$pid | cut -c1-110)
    newest=\$(find /root/okc/.build /root/okc/muse_moe_w4a16 -type f -name '*.log' -mmin -20 2>/dev/null | head -1)
    flag=''; [ \"\${et:-0}\" -gt 1800 ] && [ -z \"\$newest\" ] && flag=' HUNG?'
    q=\$pid; locked=0; while [ \"\$q\" -gt 1 ] 2>/dev/null; do ps -o args= -p \$q | grep -q 'flock .*gpu.lock' && { locked=1; break; }; q=\$(ps -o ppid= -p \$q | tr -d ' '); done
    [ \$locked -eq 0 ] && flag=\"\$flag NOLOCK\"
    echo \"  \$pid  \$((\${et:-0}/60)) min \$mem  \$cmd\$flag\"
  done
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
