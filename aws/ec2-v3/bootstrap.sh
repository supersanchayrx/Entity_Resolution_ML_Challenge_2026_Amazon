#!/bin/bash
# Unattended v3 run (notes/v3_plan.md §5): pull code+data from S3, self-test, smoke test,
# sample check A/B, full run, push outputs + work folder, write STATUS, shut down.
set -uo pipefail
B=s3://your-s3-bucket/er2026
RUN=ec2-v3
SETS="n_jobs=8 chunk_rows=500"            # v1 blocking caps (defaults); v3 settings are defaults
B_SETS="text_off=all idf_scope=split"      # sample-check run B: v2 parsing and split-wide IDF
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
fail() {  # fail <STATUS>: push the log, write STATUS, stop
  echo "$1 $(date -u)"; aws s3 cp $LOG $B/runs/$RUN/bootstrap.log --quiet
  echo "$1" | aws s3 cp - $B/runs/$RUN/STATUS; shutdown -h now; exit 1
}
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

# 1. parser self-test (§4.1)
$PY -m src.selftest || fail SELFTEST_FAILED
echo "selftest passed $(date -u)"

# 2. smoke test on a tiny sample: a code bug fails in minutes, not after the block step
if ! ( $PY run.py make-sample --data /opt/er/data --out /opt/er/smoke_data --n-s1 5000 &&
       $PY run.py all --data /opt/er/smoke_data --work /opt/er/smoke_work --out /opt/er/smoke_out --set n_jobs=8 ); then
  fail SMOKE_FAILED
fi
echo "smoke test passed $(date -u)"

# 3. sample check on 100k training S1s (§4.3): A = all v3 settings, B = text_off=all idf_scope=split.
#    The sample's test files are 1/4 the size, so train_size_cap=10 keeps all 100k S1s (k = 1);
#    density still matches test. Both runs stop after train (OOF score only).
$PY run.py make-sample --data /opt/er/data --out /opt/er/ab_data --n-s1 100000 || fail AB_SAMPLE_FAILED
$PY run.py all --to train --data /opt/er/ab_data --work /opt/er/ab_A --out /opt/er/ab_out_A \
    --set $SETS train_size_cap=10 || fail AB_RUN_A_FAILED
$PY run.py all --to train --data /opt/er/ab_data --work /opt/er/ab_B --out /opt/er/ab_out_B \
    --set $SETS train_size_cap=10 $B_SETS || fail AB_RUN_B_FAILED
for x in A B; do
  aws s3 cp /opt/er/ab_$x/model/report.json $B/runs/$RUN/ab/report_$x.json --quiet
  aws s3 sync /opt/er/ab_$x/logs $B/runs/$RUN/ab/logs_$x --quiet
done
CHOICE=$($PY - <<'EOF'
import json
a = json.load(open("/opt/er/ab_A/model/report.json"))["oof_f05"]
b = json.load(open("/opt/er/ab_B/model/report.json"))["oof_f05"]
print("A" if a >= b - 0.001 else "B", f"{a:.5f}", f"{b:.5f}")
EOF
)
echo "sample check: $CHOICE (choice, OOF A, OOF B) $(date -u)"
echo "$CHOICE" | aws s3 cp - $B/runs/$RUN/ab/CHOICE
FULL_SETS="$SETS"
[ "${CHOICE%% *}" = "B" ] && FULL_SETS="$SETS $B_SETS"
rm -rf /opt/er/ab_A /opt/er/ab_B /opt/er/smoke_work

# 4. full run: prepare -> block -> rerank -> features -> train -> predict -> variants -> diagnose
if $PY run.py all --data /opt/er/data --work /opt/er/work --out /opt/er/output --set $FULL_SETS; then
  STATUS=SUCCEEDED; else STATUS=FAILED; fi
# STATUS stays exactly SUCCEEDED/FAILED (the launch watcher gates v4 on it); the A/B choice
# is in runs/$RUN/ab/CHOICE and SETTINGS
echo "full run settings: $FULL_SETS" | aws s3 cp - $B/runs/$RUN/ab/SETTINGS
echo "$STATUS $(date -u)"
cp -r /opt/er/work/diag /opt/er/output/diag 2>/dev/null
cp /opt/er/work/model/report.json /opt/er/work/model/decision.json /opt/er/output/ 2>/dev/null
aws s3 sync /opt/er/output $B/output/$RUN/ --only-show-errors
aws s3 sync /opt/er/work $B/work/$RUN/ --only-show-errors
aws s3 cp $LOG $B/runs/$RUN/bootstrap.log --quiet
echo "$STATUS" | aws s3 cp - $B/runs/$RUN/STATUS
shutdown -h now
