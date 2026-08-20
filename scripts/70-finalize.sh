#!/usr/bin/env bash
set -euo pipefail
source "$ROOT/scripts/lib.sh"
load_config "$CONFIG"
source "$OPENRC"

server_uuid=$(cat "$RUN_DIR/state/server-uuid")
if [[ "$FINAL_ACTION" == retain ]]; then
  cat >"$RUN_DIR/RETAINED.md" <<EOF
# Retained resource

- Server UUID: $server_uuid
- Run ID: $RUN_ID
- Purpose: accepted cute-bear LoRA inference
- Retained at: $(date -u +%FT%TZ)
- Required next action: assign owner and expiry, or run UUID-guarded deletion
EOF
  printf 'retain\n' >"$RUN_DIR/state/final-action"
  exit 0
fi

current_uuid=$(os server show "$server_uuid" -f value -c id)
[[ "$current_uuid" == "$server_uuid" ]] || die "server UUID revalidation failed"
os server show "$server_uuid" -f json >"$RUN_DIR/evidence/server-before-delete.json"
os server delete "$server_uuid"
printf 'delete-submitted\n' >"$RUN_DIR/state/final-action"

deadline=$((SECONDS + 1800))
while ((SECONDS < deadline)); do
  if ! os server show "$server_uuid" >/dev/null 2>&1; then
    state=$(system_os baremetal node show "$NODE_NAME" -f value -c provision_state)
    instance=$(system_os baremetal node show "$NODE_NAME" -f value -c instance_uuid)
    if [[ "$state" == available && -z "$instance" ]]; then
      system_os baremetal node show "$NODE_NAME" -f json \
        >"$RUN_DIR/evidence/node-after-delete.json"
      printf 'delete-complete\n' >"$RUN_DIR/state/final-action"
      exit 0
    fi
  fi
  sleep 15
done
die "delete was submitted but cleanup did not finish within 30 minutes"
