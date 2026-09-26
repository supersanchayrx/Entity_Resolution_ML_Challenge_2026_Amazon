#!/bin/bash
# ec2-v57: ec2-v55 + per-country experts + unconstrained model (Spot). v5.5 (code/v5.5/) on r6i.2xlarge (8 vCPU, 64 GB + 48 GB swap). Unattended EC2 user data:
# pulls code + data from S3, runs the self-test and a smoke test, then the full run; pushes logs,
# checkpoints, outputs and work to S3 and shuts down.
# Profile v55_lite (3 stages, fold models on the test path, 5% holdout, LightGBM re-ranker, v5.5
# features, per-group decisions, pool per unseen test country, self-training variant; no experts or
# unconstrained model) plus ec2-v5's capacity (1.0M S1s per fit, min leaf 400, early stop 150, up to
# 4000 rounds) plus the cheap half of the recall push: k1 150, rev_k 10, k2 30 (more of the lists the
# same retrieval already scores; caps and prefix_m stay at v1 values, since v2's doubled caps gained nothing).
# Expected ~18 h (experts + unconstrained add ~2 stage-3 fold sets + their test predictions). Fits stop at 1060 min.
set -uo pipefail
B=s3://your-s3-bucket/er2026
RUN=ec2-v57
PROFILE=v55_lite
SETS=(n_jobs=8 lgb_threads=8 chunk_rows=500 deadline_min=1300 reserve_min=240
      max_train_s1=1000000 lgb_min_leaf=400 lgb_early_stop=150 lgb_rounds=4000
      k1=150 rev_k=10 k2=30 experts=true unconstrained=true)
LOG=/var/log/er2026.log
exec > >(tee -a $LOG) 2>&1
echo "boot $(date -u)"
export ER_T0=$(date +%s)
shutdown -h +1380                     # safety net: never run longer than 23 h
fallocate -l 48G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
mkdir -p /opt/er && cd /opt/er
( while true; do
    aws s3 cp $LOG $B/runs/$RUN/bootstrap.log --quiet
    [ -d /opt/er/work/logs ] && aws s3 sync /opt/er/work/logs $B/runs/$RUN/logs --quiet
    [ -d /opt/er/ckpt ] && aws s3 sync /opt/er/ckpt $B/runs/$RUN/ckpt --quiet
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
# smoke test on a tiny sample first (same profile/settings): a code bug fails in minutes, not hours
( $PY run.py make-sample --data /opt/er/data --out /opt/er/smoke_data --n-s1 5000 &&
  $PY run.py all --data /opt/er/smoke_data --work /opt/er/smoke_work --out /opt/er/smoke_out \
      --profile $PROFILE --set "${SETS[@]}" lgb_rounds=300 lex_min_count=10 holdout_frac=0.1 ) || fail SMOKE_FAILED
echo "smoke test passed $(date -u)"
rm -rf /opt/er/smoke_data /opt/er/smoke_work /opt/er/smoke_out
if $PY run.py all --data /opt/er/data --work /opt/er/work --out /opt/er/output \
       --checkpoint /opt/er/ckpt --profile $PROFILE --set "${SETS[@]}"; then
  STATUS=SUCCEEDED; else STATUS=FAILED; fi
echo "$STATUS $(date -u)"
aws s3 sync /opt/er/output $B/output/$RUN/ --only-show-errors
aws s3 sync /opt/er/ckpt $B/runs/$RUN/ckpt --only-show-errors
aws s3 sync /opt/er/work $B/work/$RUN/ --only-show-errors --exclude "*/cand_raw/*"
aws s3 cp $LOG $B/runs/$RUN/bootstrap.log --quiet
echo "$STATUS" | aws s3 cp - $B/runs/$RUN/STATUS
shutdown -h now
