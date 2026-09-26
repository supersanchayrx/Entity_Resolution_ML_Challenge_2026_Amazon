#!/bin/bash
# v5-lite on EC2 (v5 plan section 10): unattended run as EC2 user data. Pulls code + data from S3,
# runs the parser self-test and a smoke test, then the full run with the v5_lite profile, pushes
# logs/outputs/work to S3 and shuts down. r6i.2xlarge (8 vCPU, 64 GB + 48 GB swap): ~5-6 h.
# Upload the code first: tar czf code-ec2-v5.tgz business_entity_resolution && aws s3 cp ... $B/code/
set -uo pipefail
B=s3://your-s3-bucket/er2026
RUN=ec2-v5
PROFILE=v5_lite
SETS="n_jobs=8 lgb_threads=8 chunk_rows=500 deadline_min=780"
LOG=/var/log/er2026.log
exec > >(tee -a $LOG) 2>&1
echo "boot $(date -u)"
export ER_T0=$(date +%s)
shutdown -h +840                      # safety net: never run longer than 14 h
fallocate -l 48G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
mkdir -p /opt/er && cd /opt/er
( while true; do
    aws s3 cp $LOG $B/runs/$RUN/bootstrap.log --quiet
    [ -d /opt/er/work/logs ] && aws s3 sync /opt/er/work/logs $B/runs/$RUN/logs --quiet
    free -g > /tmp/mem.txt; df -h / >> /tmp/mem.txt; aws s3 cp /tmp/mem.txt $B/runs/$RUN/mem.txt --quiet
    sleep 60
  done ) &
aws s3 cp $B/code/code-$RUN.tgz code.tgz && tar xzf code.tgz
aws s3 sync $B/raw/ /opt/er/data/ --only-show-errors
export HOME=/root
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH
uv venv --python 3.11 /opt/er/venv
uv pip install --python /opt/er/venv/bin/python -r business_entity_resolution/requirements.txt
cd business_entity_resolution
export NUMBA_CACHE_DIR=/tmp/numba PYTHONUNBUFFERED=1
PY=/opt/er/venv/bin/python
fail() {
  echo "$1 $(date -u)"; aws s3 cp $LOG $B/runs/$RUN/bootstrap.log --quiet
  echo "$1" | aws s3 cp - $B/runs/$RUN/STATUS; shutdown -h now; exit 1
}
$PY -m src.selftest || fail SELFTEST_FAILED
# smoke test on a tiny sample first: a code bug fails in minutes, not after the block step
( $PY run.py make-sample --data /opt/er/data --out /opt/er/smoke_data --n-s1 5000 &&
  $PY run.py all --data /opt/er/smoke_data --work /opt/er/smoke_work --out /opt/er/smoke_out \
      --profile $PROFILE --set n_jobs=8 lgb_rounds=300 lex_min_count=10 ) || fail SMOKE_FAILED
echo "smoke test passed $(date -u)"
if $PY run.py all --data /opt/er/data --work /opt/er/work --out /opt/er/output \
       --checkpoint /opt/er/ckpt --profile $PROFILE --set $SETS; then
  STATUS=SUCCEEDED; else STATUS=FAILED; fi
echo "$STATUS $(date -u)"
aws s3 sync /opt/er/output $B/output/$RUN/ --only-show-errors
aws s3 sync /opt/er/work $B/work/$RUN/ --only-show-errors
aws s3 cp $LOG $B/runs/$RUN/bootstrap.log --quiet
echo "$STATUS" | aws s3 cp - $B/runs/$RUN/STATUS
shutdown -h now
