#!/usr/bin/env bash

set -euo pipefail

readonly CONFIG_KEYS=(
  OPENRC NODE_NAME IMAGE_NAME USER_DATA_FILE FLAVOR ETH_PORT_ID ETH_MAC
  IPOIB_PORT_ID IPOIB_MAC KEYPAIR
  SERVER_NAME_PREFIX EXISTING_SERVER_UUID
  GUEST_USER GUEST_SSH_KEY GUEST_PROXY_JUMP GUEST_EGRESS_PROXY
  WORKLOAD_DIGEST WORKLOAD_ARCHIVE BASE_MODEL_DIR DATASET_DIR REMOTE_INPUT_ROOT RUNS_ROOT
  TRAIN_STEPS PILOT_STEPS LORA_RANK RESOLUTION TRAIN_SEED PROMPTS_FILE
  MAX_DATASET_IMAGES OPENSTACK_TIMEOUT DEPLOY_TIMEOUT SSH_TIMEOUT TRAIN_TIMEOUT
)

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

is_allowed_key() {
  local wanted=$1 key
  for key in "${CONFIG_KEYS[@]}"; do
    [[ "$key" == "$wanted" ]] && return 0
  done
  return 1
}

load_config() {
  local path=$1 line key value
  [[ -r "$path" ]] || die "configuration is not readable: $path"
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" == *=* ]] || die "invalid configuration line"
    key=${line%%=*}
    value=${line#*=}
    [[ "$key" =~ ^[A-Z][A-Z0-9_]*$ ]] || die "invalid configuration key: $key"
    is_allowed_key "$key" || die "configuration key is not allowed: $key"
    [[ "$value" != *$'\n'* && "$value" != *$'\r'* ]] || die "multiline value rejected: $key"
    [[ "$value" =~ ^[A-Za-z0-9_./:@,+=-]*$ ]] || die "unsafe characters in configuration: $key"
    printf -v "$key" '%s' "$value"
    export "$key"
  done <"$path"

  if grep -Eiq '(^|_)(PASSWORD|TOKEN|SECRET|CREDENTIAL|SIGNED_URL)=' "$path"; then
    die "credentials are forbidden in config.env"
  fi
}

require_vars() {
  local key
  for key in "$@"; do
    [[ -n "${!key:-}" ]] || die "required configuration is empty: $key"
    [[ "${!key}" != *replace-with* && "${!key}" != *replace ]] || \
      die "placeholder remains in configuration: $key"
  done
}

require_uint() {
  local key=$1 value=${!1:-}
  [[ "$value" =~ ^[0-9]+$ ]] || die "$key must be an unsigned integer"
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

json_value() {
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}

os() {
  timeout --foreground "${OPENSTACK_TIMEOUT:-120}" openstack "$@"
}

system_os() (
  unset OS_PROJECT_ID OS_PROJECT_NAME OS_PROJECT_DOMAIN_ID OS_PROJECT_DOMAIN_NAME
  export OS_SYSTEM_SCOPE=all
  os "$@"
)

ssh_args() {
  SSH_ARGS=(-i "$GUEST_SSH_KEY" -o BatchMode=yes -o ConnectTimeout=12 \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
  [[ -n "${GUEST_PROXY_JUMP:-}" ]] && SSH_ARGS+=(-J "$GUEST_PROXY_JUMP")
  return 0
}

guest_host() {
  [[ -r "$RUN_DIR/state/guest-ip" ]] || die "guest IP is not recorded"
  cat "$RUN_DIR/state/guest-ip"
}

guest_ssh() {
  local host
  host=$(guest_host)
  ssh_args
  ssh "${SSH_ARGS[@]}" "$GUEST_USER@$host" "$@"
}

guest_copy_to() {
  local source=$1 destination=$2 host
  host=$(guest_host)
  ssh_args
  scp "${SSH_ARGS[@]}" -r "$source" "$GUEST_USER@$host:$destination"
}
