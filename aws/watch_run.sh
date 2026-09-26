#!/usr/bin/env bash
# One EC2 run, launched and followed on its own (successor of watch_and_launch.sh for parallel runs).
# Launches RUN if its bundle passes the checks and its quota slot is free (On-Demand and Spot are
# separate 8-vCPU quotas, one r6i.2xlarge each), then follows ONLY that run's instance, and when it
# is gone fetches + validates the results into submissions/RUN/. Several of these can run at once.
#
#   bash watch_run.sh [--spot] [--dry-run] RUN
#
#   --spot      launch as a one-time Spot instance (terminated if AWS reclaims it: no STATUS then)
#   --dry-run   every check, then run-instances --dry-run (nothing starts)
# A RUN with a LAUNCHED file is never launched again; the watcher just follows its instance.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)                     # ML_Challenge_student_resource
export PATH="/path/to/AWSCLIV2:$PATH"
export AWS_PROFILE=er2026 AWS_DEFAULT_REGION=us-east-1 MSYS_NO_PATHCONV=1
B=s3://your-s3-bucket/er2026
POLL=${POLL:-120}                                   # seconds between checks
LOG=$HERE/watch.log

SPOT=0; DRY=0; RUN=""
while [ $# -gt 0 ]; do
  case $1 in
    --spot) SPOT=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -*) echo "unknown option $1"; exit 2 ;;
    *) RUN=$1; shift ;;
  esac
done
[ -n "$RUN" ] || { echo "usage: $0 [--spot] [--dry-run] RUN"; exit 2; }
D="$HERE/$RUN"
MARKET=$([ $SPOT = 1 ] && echo spot || echo on-demand)

log() { local m="[$(date -u '+%F %T') UTC] [$RUN] $*"; echo "$m" | tee -a "$LOG"; echo "$m" > "$D/STATE"; }
notify() {                                          # log + Windows balloon (best effort, non-blocking)
  log "NOTIFY: $*"
  local msg=${*//\'/}
  powershell.exe -NoProfile -WindowStyle Hidden -Command "Add-Type -AssemblyName System.Windows.Forms; \
    \$n=New-Object System.Windows.Forms.NotifyIcon; \$n.Icon=[System.Drawing.SystemIcons]::Information; \
    \$n.Visible=\$true; \$n.ShowBalloonTip(15000,'er2026 $RUN','${msg:0:250}','Info'); Start-Sleep 16; \$n.Dispose()" \
    >/dev/null 2>&1 &
}

mkdir -p "$D"
# one watcher per run
LOCK=$D/.watch.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  pid=$(cat "$LOCK/pid" 2>/dev/null)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "a watcher for $RUN is running (pid $pid)"; exit 1; fi
  rm -rf "$LOCK"; mkdir "$LOCK"
fi
echo $$ > "$LOCK/pid"
# keep Windows from idle-sleeping while we watch (closing the lid can still sleep it)
powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "$(cygpath -w "$HERE/keep_awake.ps1")" >/dev/null 2>&1 &
AWAKE=$!
trap 'kill $AWAKE 2>/dev/null; rm -rf "$LOCK"' EXIT

win() { cygpath -w "$1"; }                           # aws.exe is a Windows program: give it Windows paths
run_status() { aws s3 cp "$B/runs/$1/STATUS" - 2>/dev/null | tr -d '\r\n '; }

# live instances of this market type (they use its quota); UNKNOWN on API error
slot_busy() {
  local out filt
  if [ $SPOT = 1 ]; then filt="Name=instance-lifecycle,Values=spot"; else filt="Name=tag-key,Values=Name"; fi
  out=$(aws ec2 describe-instances --filters Name=instance-state-name,Values=pending,running,stopping,shutting-down "$filt" \
        --query 'Reservations[].Instances[].[InstanceId,Tags[?Key==`Name`]|[0].Value,State.Name,InstanceLifecycle]' \
        --output text 2>&1) || { echo "UNKNOWN(${out:0:80})"; return; }
  # On-Demand instances have no lifecycle field ("None"); Spot ones say "spot"
  if [ $SPOT = 1 ]; then echo "$out" | awk 'NF{printf "%s:%s:%s ", $1, $2, $3}'
  else echo "$out" | awk 'NF && $4!="spot"{printf "%s:%s:%s ", $1, $2, $3}'; fi
}

# state of this run's own instance: pending|running|...|gone ; UNKNOWN on other API errors
own_state() {
  local out
  out=$(aws ec2 describe-instances --instance-ids "$1" \
        --query 'Reservations[0].Instances[0].[State.Name,StateReason.Code]' --output text 2>&1)
  if [ $? != 0 ]; then
    echo "$out" | grep -q InvalidInstanceID && { echo gone; return; }
    echo "UNKNOWN(${out:0:80})"; return
  fi
  echo "$out" | tr '\t' ' '
}

# download a finished run's small outputs (no candidate_pairs) and validate every matching file
fetch_results() {
  local run=$1 dst="$ROOT/submissions/$1" free f res
  free=$(df -m "$ROOT" | awk 'NR==2{print $4}')
  mkdir -p "$dst"
  aws s3 cp "$B/work/$run/model/report.json" "$(win "$dst/report.json")" --only-show-errors
  aws s3 cp "$B/runs/$run/bootstrap.log" "$(win "$dst/bootstrap.log")" --only-show-errors
  if [ "${free:-0}" -lt 3000 ]; then
    notify "only ${free} MB free on C:, skipped downloading the submission files"; return
  fi
  aws s3 sync "$B/output/$run/" "$(win "$dst")" --exclude "*candidate_pairs*" --only-show-errors
  res=""
  while IFS= read -r f; do
    if (cd "$ROOT/student_resource" && python utils/validate_submission.py --matching "$(win "$f")" \
          --test-dir dataset/test > "$f.validate.txt" 2>&1); then
      res+="PASS ${f#$dst/}; "; else res+="FAIL ${f#$dst/}; "; fi
  done < <(find "$dst" -name matching_results.tsv)
  log "validator: ${res:-no matching_results.tsv found}"
  [ -f "$dst/report.json" ] && log "report: $(tr -d '\n ' < "$dst/report.json" | cut -c1-400)"
}

# every check that must pass before a bundle may be launched; prints the reason on failure
check_bundle() {
  local f
  for f in bootstrap.sh code.tgz READY; do [ -s "$D/$f" ] || { echo "missing $f"; return 1; }; done
  (cd "$D" && sha256sum -c --quiet READY >/dev/null 2>&1) || { echo "READY checksums do not match (files changed after READY?)"; return 1; }
  bash -n "$D/bootstrap.sh" 2>/dev/null || { echo "bootstrap.sh has a bash syntax error"; return 1; }
  grep -q "^RUN=$RUN\$" "$D/bootstrap.sh" || { echo "bootstrap.sh does not set RUN=$RUN"; return 1; }
  grep -q 'code-\$RUN.tgz\|code-\${RUN}.tgz' "$D/bootstrap.sh" || { echo "bootstrap.sh does not download code-\$RUN.tgz"; return 1; }
  grep -q 'shutdown -h +[0-9]' "$D/bootstrap.sh" || { echo "bootstrap.sh has no safety shutdown (shutdown -h +MIN)"; return 1; }
  grep -q 'STATUS' "$D/bootstrap.sh" || { echo "bootstrap.sh never writes STATUS"; return 1; }
  [ "$(stat -c %s "$D/bootstrap.sh")" -lt 16000 ] || { echo "bootstrap.sh exceeds the 16 KB user-data limit"; return 1; }
  local listing; listing=$(tar tzf "$D/code.tgz" 2>/dev/null)
  grep -qx 'business_entity_resolution/run.py' <<< "$listing" || { echo "code.tgz lacks business_entity_resolution/run.py"; return 1; }
  [ -z "$(run_status "$RUN")" ] || { echo "S3 already has runs/$RUN/STATUS (run name reused?)"; return 1; }
  return 0
}

launch() {
  local ami id market=()
  aws s3 cp "$(win "$D/code.tgz")" "$B/code/code-$RUN.tgz" --only-show-errors || return 1
  aws s3 cp "$(win "$D/bootstrap.sh")" "$B/code/bootstrap-$RUN.sh" --only-show-errors || return 1
  ami=$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
        --query Parameter.Value --output text) || return 1
  local dry=(); [ $DRY = 1 ] && dry=(--dry-run)
  # On-Demand: r6i.2xlarge, default placement. Spot: capacity comes and goes per zone and type, so
  # try every default subnet for each 8-vCPU / 64 GB type, Intel first, until one is accepted.
  local types=(r6i.2xlarge) subnets=("") t s rc=1
  if [ $SPOT = 1 ]; then
    market=(--instance-market-options
      'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
    types=(r6i.2xlarge r7i.2xlarge r5.2xlarge r6a.2xlarge r7a.2xlarge)
    mapfile -t subnets < <(aws ec2 describe-subnets --filters Name=default-for-az,Values=true \
                           --query 'Subnets[].SubnetId' --output text | tr '\t' '\n' | tr -d '\r' | grep .)
  fi
  for t in "${types[@]}"; do
    for s in "${subnets[@]}"; do
      local sub=(); [ -n "$s" ] && sub=(--subnet-id "$s")
      id=$(cd "$D" && aws ec2 run-instances "${dry[@]}" "${market[@]}" "${sub[@]}" --image-id "$ami" --instance-type "$t" \
            --iam-instance-profile Name=er2026-ec2-runner --instance-initiated-shutdown-behavior terminate \
            --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":200,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
            --user-data file://bootstrap.sh --metadata-options HttpTokens=required \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=er2026-$RUN}]" \
            --query 'Instances[0].InstanceId' --output text 2>&1)
      rc=$?
      if echo "$id" | grep -q 'InsufficientInstanceCapacity\|Unsupported'; then
        log "no capacity: $t ${s:-default}"; continue
      fi
      break 2
    done
  done
  MARKET="$MARKET $t ${s:-default}"
  if [ $DRY = 1 ]; then
    if echo "$id" | grep -q DryRunOperation; then log "DRY RUN OK ($MARKET; would launch now; AMI $ami)"; return 0; fi
    log "DRY RUN FAILED ($MARKET): $id"; return 1
  fi
  [ $rc = 0 ] && [[ $id == i-* ]] || { log "run-instances failed ($MARKET): $id"; return 1; }
  echo "$id $(date -u '+%F %T') UTC $MARKET" > "$D/LAUNCHED"
  notify "Launched on EC2 ($MARKET, $id). Logs: er2026/runs/$RUN/"
}

log "watcher start: market=$MARKET dry=$DRY poll=${POLL}s"
if [ ! -f "$D/LAUNCHED" ]; then
  last=""
  while :; do
    if reason=$(check_bundle); then
      busy=$(slot_busy)
      if [ -z "$busy" ]; then launch && break; notify "launch failed; retrying in ${POLL}s"
      else [ "$busy" != "$last" ] && log "waiting: $MARKET slot busy [$busy]"; last=$busy; fi
    else
      [ "$reason" != "$last" ] && notify "bundle failed a check: $reason"; last=$reason
    fi
    sleep "$POLL"
  done
  [ $DRY = 1 ] && { log "dry run: done"; exit 0; }
  sleep 120
fi

ID=$(awk '{print $1}' "$D/LAUNCHED")
log "following $ID"
prev=""
while :; do
  st=$(own_state "$ID")
  case $st in
    gone|terminated*) break ;;
  esac
  [ "$st" != "$prev" ] && log "instance $ID: $st (STATUS=$(run_status "$RUN"))"
  prev=$st
  sleep "$POLL"
done
status=$(run_status "$RUN")
if [ -z "$status" ]; then
  notify "instance ended WITHOUT a STATUS ($st)$( [[ $st == *Spot* ]] && echo ': reclaimed by AWS (Spot interruption)')"
else
  notify "finished: $status"
fi
fetch_results "$RUN"
log "done: ${status:-NO STATUS}"
