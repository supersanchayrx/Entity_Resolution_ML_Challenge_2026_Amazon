#!/bin/bash
# Unattended v4 run (notes/v4_plan.md §8): reuse v3's work folder (prepare/block/rerank), rerun
# features -> train -> predict -> variants -> diagnose with the v4 features. OOF compares with v3.
set -uo pipefail
B=s3://your-s3-bucket/er2026
RUN=ec2-v4
PREV=ec2-v3
LOG=/var/log/er2026.log
exec > >(tee -a $LOG) 2>&1
echo "boot $(date -u)"
shutdown -h +840                      # safety net: never run longer than 14 h
fallocate -l 48G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
mkdir -p /opt/er && cd /opt/er
( while true; do
    aws s3 cp $LOG $B/runs/$RUN/bootstrap.log --quiet
    [ -d /opt/er/work/logs ] && aws s3 sync /opt/er/work/logs $B/runs/$RUN/logs --quiet
    free -g > /tmp/mem.txt; swapon --show >> /tmp/mem.txt; df -h / >> /tmp/mem.txt
    aws s3 cp /tmp/mem.txt $B/runs/$RUN/mem.txt --quiet
    free -g | awk '/Swap/{print strftime("%H:%M:%S"), $3}' >> /tmp/swaplog.txt
    aws s3 cp /tmp/swaplog.txt $B/runs/$RUN/swap_used_gb.txt --quiet
    sleep 60
  done ) &
fail() {  # fail <reason>: push the log, write STATUS=FAILED (+ reason), stop
  echo "$1 $(date -u)"; aws s3 cp $LOG $B/runs/$RUN/bootstrap.log --quiet
  echo "$1" | aws s3 cp - $B/runs/$RUN/FAIL_REASON
  echo FAILED | aws s3 cp - $B/runs/$RUN/STATUS; shutdown -h now; exit 1
}
# v4 needs a complete v3 work folder
[ "$(aws s3 cp $B/runs/$PREV/STATUS - 2>/dev/null | tr -d '\r\n ')" = SUCCEEDED ] || fail PREV_NOT_SUCCEEDED
aws s3 cp $B/code/code-$RUN.tgz code.tgz && tar xzf code.tgz || fail CODE_DOWNLOAD_FAILED
aws s3 sync $B/raw/ /opt/er/data/ --only-show-errors
export HOME=/root
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH
uv venv --python 3.11 /opt/er/venv
uv pip install --python /opt/er/venv/bin/python -r business_entity_resolution/requirements.txt
cd business_entity_resolution
export NUMBA_CACHE_DIR=/tmp/numba PYTHONUNBUFFERED=1
PY=/opt/er/venv/bin/python

# v3's full-run settings (its A/B choice) must be reused: they shape rec, IDF and the text fields
SETS=$(aws s3 cp $B/runs/$PREV/ab/SETTINGS - 2>/dev/null | sed 's/^full run settings: //')
[ -n "$SETS" ] || SETS="n_jobs=8 chunk_rows=500"
echo "settings from $PREV: $SETS"

# 1. parser + v4 self-test
$PY -m src.selftest || fail SELFTEST_FAILED
echo "selftest passed $(date -u)"

# 2. smoke test on 5k S1s along the exact full-run path: v3 steps into one folder, then
#    --from features with that folder as --work-in
if ! ( $PY run.py make-sample --data /opt/er/data --out /opt/er/smoke_data --n-s1 5000 &&
       $PY run.py all --to rerank --data /opt/er/smoke_data --work /opt/er/smoke_v3 --out /opt/er/smoke_out --set $SETS &&
       $PY run.py all --from features --data /opt/er/smoke_data --work /opt/er/smoke_v4 --work-in /opt/er/smoke_v3 \
           --out /opt/er/smoke_out --set $SETS ); then
  fail SMOKE_FAILED
fi
echo "smoke test passed $(date -u)"
rm -rf /opt/er/smoke_v3 /opt/er/smoke_v4

# 3. v3's work folder (~40 GB), read-only
aws s3 sync $B/work/$PREV/ /opt/er/v3work/ --only-show-errors || fail V3_WORK_DOWNLOAD_FAILED
echo "v3 work copied $(du -sh /opt/er/v3work | cut -f1) $(date -u)"

# 4. full run from features
if $PY run.py all --from features --data /opt/er/data --work /opt/er/work --work-in /opt/er/v3work \
     --out /opt/er/output --set $SETS; then
  STATUS=SUCCEEDED; else STATUS=FAILED; fi
echo "$STATUS $(date -u)"
cp -r /opt/er/work/diag /opt/er/output/diag 2>/dev/null
cp /opt/er/work/model/report.json /opt/er/work/model/decision.json /opt/er/output/ 2>/dev/null
aws s3 sync /opt/er/output $B/output/$RUN/ --only-show-errors
aws s3 sync /opt/er/work $B/work/$RUN/ --only-show-errors     # only v4's new files; v3's stay in work/ec2-v3
aws s3 cp $LOG $B/runs/$RUN/bootstrap.log --quiet
echo "$STATUS" | aws s3 cp - $B/runs/$RUN/STATUS
shutdown -h now
