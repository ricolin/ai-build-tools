#!/usr/bin/env bash
set -euo pipefail
source "$ROOT/scripts/lib.sh"
load_config "$CONFIG"
mkdir -p "$RUN_DIR/evidence/preflight"

require_vars OPENRC NODE_NAME IMAGE_NAME USER_DATA_FILE FLAVOR ETH_PORT_ID ETH_MAC KEYPAIR \
  WORKLOAD_DIGEST BASE_MODEL_DIR DATASET_DIR RUNS_ROOT TRAIN_STEPS PILOT_STEPS \
  LORA_RANK RESOLUTION TRAIN_SEED PROMPTS_FILE MAX_DATASET_IMAGES
for key in TRAIN_STEPS PILOT_STEPS LORA_RANK RESOLUTION TRAIN_SEED \
  MAX_DATASET_IMAGES OPENSTACK_TIMEOUT DEPLOY_TIMEOUT SSH_TIMEOUT TRAIN_TIMEOUT; do
  require_uint "$key"
done
[[ "$TRAIN_STEPS" -le 300 && "$PILOT_STEPS" -le 40 ]] || die "step budget exceeded"
[[ "$LORA_RANK" -eq 32 && "$RESOLUTION" -eq 1024 ]] || die "accepted scale changed"
[[ "$ETH_MAC" =~ ^([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}$ ]] || die "ETH_MAC is invalid"
if [[ -n "${IPOIB_PORT_ID:-}" ]]; then
  require_vars IPOIB_MAC
  [[ "$IPOIB_MAC" =~ ^([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}$ ]] || \
    die "IPOIB_MAC is invalid"
fi
if [[ -n "${WORKLOAD_ARCHIVE:-}" ]]; then
  [[ -r "$WORKLOAD_ARCHIVE" ]] || die "workload archive is missing: $WORKLOAD_ARCHIVE"
  sha256sum "$WORKLOAD_ARCHIVE" >"$RUN_DIR/evidence/preflight/workload-archive.sha256"
else
  [[ "$WORKLOAD_DIGEST" == *@sha256:* ]] || die "registry workload image must use a digest"
fi
[[ -d "$BASE_MODEL_DIR" ]] || die "staged base model is missing: $BASE_MODEL_DIR"
[[ -d "$DATASET_DIR" ]] || die "staged dataset is missing: $DATASET_DIR"
[[ -r "$ROOT/$PROMPTS_FILE" ]] || die "prompt file is missing: $PROMPTS_FILE"
[[ -r "$ROOT/$USER_DATA_FILE" ]] || die "guest user-data is missing: $USER_DATA_FILE"
[[ "$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$ROOT/$PROMPTS_FILE")" -eq 3 ]] || \
  die "acceptance prompt set must contain exactly three prompts"

source "$OPENRC"
os image show "$IMAGE_NAME" -f json >"$RUN_DIR/evidence/preflight/image.json"
os flavor show "$FLAVOR" -f json >"$RUN_DIR/evidence/preflight/flavor.json"
system_os baremetal node show "$NODE_NAME" -f json >"$RUN_DIR/evidence/preflight/node.json"
os port show "$ETH_PORT_ID" -f json >"$RUN_DIR/evidence/preflight/ethernet-port.json"
eth_subnet_id=$(jq -r '.fixed_ips[0].subnet_id // empty' \
  "$RUN_DIR/evidence/preflight/ethernet-port.json")
[[ "$eth_subnet_id" =~ ^[0-9a-fA-F-]{36}$ ]] || \
  die "Ethernet port has no fixed-IP subnet"
os subnet show "$eth_subnet_id" -f json >"$RUN_DIR/evidence/preflight/ethernet-subnet.json"
if [[ -n "${IPOIB_PORT_ID:-}" ]]; then
  os port show "$IPOIB_PORT_ID" -f json >"$RUN_DIR/evidence/preflight/ipoib-port.json"
fi

if [[ -n "${EXISTING_SERVER_UUID:-}" ]]; then
  [[ "$EXISTING_SERVER_UUID" =~ ^[0-9a-fA-F-]{36}$ ]] || \
    die "EXISTING_SERVER_UUID is invalid"
  os server show "$EXISTING_SERVER_UUID" -f json \
    >"$RUN_DIR/evidence/preflight/existing-server.json"
  jq -e --arg uuid "$EXISTING_SERVER_UUID" \
    '.provision_state == "active" and .maintenance == false and
     .instance_uuid == $uuid and
     (.last_error == null or .last_error == "")' \
    "$RUN_DIR/evidence/preflight/node.json" >/dev/null || \
    die "Ironic node does not match the requested active server"
  jq -e '.status == "ACTIVE"' \
    "$RUN_DIR/evidence/preflight/existing-server.json" >/dev/null || \
    die "existing Nova server is not ACTIVE"
else
  jq -e '.provision_state == "available" and .maintenance == false and
    (.instance_uuid == null or .instance_uuid == "") and
    (.last_error == null or .last_error == "")' \
    "$RUN_DIR/evidence/preflight/node.json" >/dev/null || \
    die "Ironic node is not available for a fresh deployment"
fi
if [[ -n "${IPOIB_PORT_ID:-}" ]]; then
  jq -e '.network_interface == "ipoib"' \
    "$RUN_DIR/evidence/preflight/node.json" >/dev/null || \
    die "IPoIB VIF requested but Ironic network_interface is not ipoib"
fi
if [[ -z "${EXISTING_SERVER_UUID:-}" ]]; then
  jq -e '(.device_id == null or .device_id == "")' \
    "$RUN_DIR/evidence/preflight/ethernet-port.json" >/dev/null
fi
jq -e '.enable_dhcp == false' \
  "$RUN_DIR/evidence/preflight/ethernet-subnet.json" >/dev/null || \
  die "bare-metal Ethernet subnet must disable DHCP for static ConfigDrive data"
if [[ -n "${IPOIB_PORT_ID:-}" && -z "${EXISTING_SERVER_UUID:-}" ]]; then
  jq -e '(.device_id == null or .device_id == "")' \
    "$RUN_DIR/evidence/preflight/ipoib-port.json" >/dev/null
fi

if python3 -c 'import PIL' >/dev/null 2>&1; then
  python3 "$ROOT/scripts/prepare_dataset.py" \
    --dataset-dir "$DATASET_DIR" --max-images "$MAX_DATASET_IMAGES" \
    --summary "$RUN_DIR/evidence/preflight/dataset.json"
else
  docker run --rm --network none \
    -v "$DATASET_DIR:/dataset:ro" \
    -v "$RUN_DIR/evidence/preflight:/evidence" \
    "$WORKLOAD_DIGEST" \
    python3 /opt/ai-build-tools/scripts/prepare_dataset.py \
      --dataset-dir /dataset --max-images "$MAX_DATASET_IMAGES" \
      --summary /evidence/dataset.json
fi

find "$BASE_MODEL_DIR" -type f -print0 | sort -z | xargs -0 sha256sum \
  >"$RUN_DIR/evidence/preflight/base-model-SHA256SUMS"
find "$DATASET_DIR" -type f -print0 | sort -z | xargs -0 sha256sum \
  >"$RUN_DIR/evidence/preflight/dataset-SHA256SUMS"
