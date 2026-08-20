#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/scripts/lib.sh"

CONFIG=
FINAL_ACTION=retain
PREFLIGHT_ONLY=false
RESUME_ID=

while (($#)); do
  case "$1" in
    --config) CONFIG=${2:?}; shift 2 ;;
    --final-action) FINAL_ACTION=${2:?}; shift 2 ;;
    --preflight-only) PREFLIGHT_ONLY=true; shift ;;
    --resume) RESUME_ID=${2:?}; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$CONFIG" ]] || die "--config is required"
[[ "$FINAL_ACTION" == retain || "$FINAL_ACTION" == delete ]] || \
  die "--final-action must be retain or delete"
if [[ "$PREFLIGHT_ONLY" != true ]]; then
  [[ "${CONFIRM_IMAGE_RUN:-}" == approved-h200-run ]] || \
    die "set CONFIRM_IMAGE_RUN=approved-h200-run"
fi

load_config "$CONFIG"
require_vars RUNS_ROOT

if [[ -n "$RESUME_ID" ]]; then
  RUN_ID=$RESUME_ID
  RUN_DIR="$RUNS_ROOT/$RUN_ID"
  [[ -d "$RUN_DIR" ]] || die "resume run does not exist: $RUN_ID"
else
  RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
  RUN_DIR="$RUNS_ROOT/$RUN_ID"
  mkdir -p "$RUN_DIR"/{logs,state,evidence,release}
  printf '%s\n' "$RUN_ID" >"$RUN_DIR/state/run-id"
  ln -sfn "$RUN_DIR" "$RUNS_ROOT/latest.tmp"
  mv -Tf "$RUNS_ROOT/latest.tmp" "$RUNS_ROOT/latest" 2>/dev/null || {
    rm -f "$RUNS_ROOT/latest"
    ln -s "$RUN_DIR" "$RUNS_ROOT/latest"
  }
fi
export ROOT CONFIG RUN_ID RUN_DIR FINAL_ACTION
[[ "$RUN_ID" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || die "run ID has an unsafe format"

if [[ ! -e "$RUN_DIR/timeline.tsv" ]]; then
  printf 'run_id\tphase\tresult\telapsed_seconds\n' >"$RUN_DIR/timeline.tsv"
fi

on_exit() {
  local rc=$?
  if ((rc == 0)); then
    printf 'Run completed. Validate with: %s/validate-run.sh --run-dir %s\n' \
      "$ROOT" "$RUN_DIR" >"$RUN_DIR/next-action.txt"
  else
    printf 'Run stopped. Inspect %s/logs and resume only after diagnosis.\n' \
      "$RUN_DIR" >"$RUN_DIR/next-action.txt"
  fi
}
trap on_exit EXIT

run_phase() {
  local phase=$1 script=$2 start end elapsed marker log
  marker="$RUN_DIR/state/$phase.complete"
  log="$RUN_DIR/logs/$phase.log"
  if [[ -e "$marker" ]]; then
    printf 'PHASE=%s RESULT=SKIP REASON=already-complete\n' "$phase"
    return
  fi
  start=$(date +%s)
  if bash "$ROOT/scripts/$script" >"$log" 2>&1; then
    end=$(date +%s); elapsed=$((end - start))
    printf '%s\n' "$(date -u +%FT%TZ)" >"$marker"
    printf '%s\t%s\tPASS\t%s\n' "$RUN_ID" "$phase" "$elapsed" >>"$RUN_DIR/timeline.tsv"
    printf 'PHASE=%s RESULT=PASS ELAPSED=%ss LOG=%s\n' "$phase" "$elapsed" "$log"
  else
    end=$(date +%s); elapsed=$((end - start))
    printf '%s\t%s\tFAIL\t%s\n' "$RUN_ID" "$phase" "$elapsed" >>"$RUN_DIR/timeline.tsv"
    printf 'PHASE=%s RESULT=FAIL ELAPSED=%ss LOG=%s\n' "$phase" "$elapsed" "$log" >&2
    return 1
  fi
}

run_phase 00-preflight 00-preflight.sh
if [[ "$PREFLIGHT_ONLY" == true ]]; then
  exit 0
fi
run_phase 10-deploy 10-deploy.sh
run_phase 20-accept-guest 20-accept-guest.sh
run_phase 30-pilot 30-pilot.sh
run_phase 40-train 40-train.sh
run_phase 50-generate 50-generate.sh
run_phase 60-package 60-package.sh
run_phase 70-finalize 70-finalize.sh
