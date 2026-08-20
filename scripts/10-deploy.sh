#!/usr/bin/env bash
set -euo pipefail
source "$ROOT/scripts/lib.sh"
load_config "$CONFIG"
source "$OPENRC"

if [[ -n "${EXISTING_SERVER_UUID:-}" ]]; then
  server_uuid=$EXISTING_SERVER_UUID
  server_json=$(os server show "$server_uuid" -f json)
  node_json=$(system_os baremetal node show "$NODE_NAME" -f json)
  jq -e '.status == "ACTIVE"' <<<"$server_json" >/dev/null || \
    die "existing Nova server is not ACTIVE"
  jq -e --arg uuid "$server_uuid" \
    '.provision_state == "active" and .instance_uuid == $uuid and
     .maintenance == false and (.last_error == null or .last_error == "")' \
    <<<"$node_json" >/dev/null || \
    die "Ironic node does not belong to the requested existing server"
  server_name=$(jq -r '.name' <<<"$server_json")
  printf '%s\n' "$server_uuid" >"$RUN_DIR/state/server-uuid"
  printf '%s\n' "$server_name" >"$RUN_DIR/state/server-name"
  printf '%s\n' "$server_json" >"$RUN_DIR/evidence/server-active.json"
  printf '%s\n' "$node_json" >"$RUN_DIR/evidence/node-active.json"
  guest_ip=$(os port show "$ETH_PORT_ID" -f json | jq -r '.fixed_ips[0].ip_address // empty')
  [[ "$guest_ip" =~ ^[0-9a-fA-F:.]+$ ]] || die "Ethernet port has no usable fixed IP"
  printf '%s\n' "$guest_ip" >"$RUN_DIR/state/guest-ip"
  exit 0
fi

server_name="${SERVER_NAME_PREFIX}-${RUN_ID}"
os port set "$ETH_PORT_ID" --mac-address "$ETH_MAC"
network_args=(--nic "port-id=$ETH_PORT_ID")
if [[ -n "${IPOIB_PORT_ID:-}" ]]; then
  os port set "$IPOIB_PORT_ID" --mac-address "$IPOIB_MAC"
  network_args+=(--nic "port-id=$IPOIB_PORT_ID")
fi
server_uuid=$(os server create "$server_name" \
  --image "$IMAGE_NAME" \
  --flavor "$FLAVOR" \
  "${network_args[@]}" \
  --key-name "$KEYPAIR" \
  --user-data "$ROOT/$USER_DATA_FILE" \
  --config-drive True \
  -f value -c id)
[[ "$server_uuid" =~ ^[0-9a-fA-F-]{36}$ ]] || die "server create returned an invalid UUID"
printf '%s\n' "$server_uuid" >"$RUN_DIR/state/server-uuid"
printf '%s\n' "$server_name" >"$RUN_DIR/state/server-name"

complete=false
deadline=$((SECONDS + DEPLOY_TIMEOUT))
while ((SECONDS < deadline)); do
  server_json=$(os server show "$server_uuid" -f json 2>/dev/null || true)
  node_json=$(system_os baremetal node show "$NODE_NAME" -f json)
  nova_state=$(jq -r '.status // "UNKNOWN"' <<<"$server_json")
  ironic_state=$(jq -r '.provision_state // "UNKNOWN"' <<<"$node_json")
  printf '%s nova=%s ironic=%s\n' "$(date -u +%FT%TZ)" "$nova_state" "$ironic_state"
  if [[ "$nova_state" == ACTIVE && "$ironic_state" == active ]]; then
    complete=true
    break
  fi
  [[ "$nova_state" != ERROR && "$ironic_state" != 'deploy failed' ]] || break
  sleep 15
done

os server show "$server_uuid" -f json >"$RUN_DIR/evidence/server-active.json" || true
system_os baremetal node show "$NODE_NAME" -f json >"$RUN_DIR/evidence/node-active.json"
[[ "$complete" == true ]] || die "deployment did not reach Nova ACTIVE and Ironic active"

guest_ip=$(os port show "$ETH_PORT_ID" -f json | jq -r '.fixed_ips[0].ip_address // empty')
[[ "$guest_ip" =~ ^[0-9a-fA-F:.]+$ ]] || die "Ethernet port has no usable fixed IP"
printf '%s\n' "$guest_ip" >"$RUN_DIR/state/guest-ip"
