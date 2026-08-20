#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "accept-guest failed at line %s (rc=%s)\n" "$LINENO" "$rc" >&2' ERR
source "$ROOT/scripts/lib.sh"
load_config "$CONFIG"

deadline=$((SECONDS + SSH_TIMEOUT))
until guest_ssh 'true' 2>/dev/null; do
  ((SECONDS < deadline)) || die "guest SSH did not become ready"
  sleep 10
done

mkdir -p "$RUN_DIR/evidence/guest"
guest_ssh 'cloud-init status --wait --long' >"$RUN_DIR/evidence/guest/cloud-init.txt"
guest_ssh 'systemctl --failed --no-pager' >"$RUN_DIR/evidence/guest/systemd-failed.txt" || true
guest_ssh 'sudo test -s /etc/machine-id && sudo netplan generate'
guest_ssh 'ip -br address; ip route; networkctl status --no-pager' \
  >"$RUN_DIR/evidence/guest/network.txt"
guest_ip=$(guest_host)
guest_ssh "ip -4 -o address show | grep -F ' $guest_ip/'" >/dev/null || \
  die "guest Ethernet fixed IP is not configured"

proxy_pid=
cleanup_proxy() {
  if [[ -n "$proxy_pid" ]]; then
    kill "$proxy_pid" >/dev/null 2>&1 || true
    wait "$proxy_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup_proxy EXIT
if [[ -n "${GUEST_EGRESS_PROXY:-}" ]]; then
  [[ "$GUEST_EGRESS_PROXY" =~ ^socks5h://127\.0\.0\.1:([0-9]{2,5})$ ]] || \
    die "GUEST_EGRESS_PROXY must be a loopback socks5h URL"
  proxy_port=${BASH_REMATCH[1]}
  host=$(guest_host)
  ssh_args
  ssh "${SSH_ARGS[@]}" -N -R "127.0.0.1:$proxy_port" \
    "$GUEST_USER@$host" &
  proxy_pid=$!
  for _ in $(seq 1 20); do
    guest_ssh "ss -lnt | grep -q '127.0.0.1:$proxy_port'" && break
    kill -0 "$proxy_pid" 2>/dev/null || die "guest egress tunnel exited"
    sleep 1
  done
  guest_ssh "ss -lnt | grep -q '127.0.0.1:$proxy_port'" || \
    die "guest egress tunnel did not become ready"
fi

guest_copy_to "$ROOT/scripts/bootstrap_gpu_runtime.sh" /tmp/bootstrap_gpu_runtime.sh
guest_ssh 'sudo install -m 0755 /tmp/bootstrap_gpu_runtime.sh /usr/local/sbin/bootstrap-gpu-runtime'
guest_ssh "sudo env EGRESS_PROXY='${GUEST_EGRESS_PROXY:-}' \
  /usr/local/sbin/bootstrap-gpu-runtime" \
  >"$RUN_DIR/evidence/guest/runtime-install.txt" 2>&1
cleanup_proxy
proxy_pid=

# A new boot is required before the newly installed NVIDIA kernel module can
# be accepted. SSH disconnect during systemctl reboot is expected.
host=$(guest_host); ssh_args
ssh "${SSH_ARGS[@]}" "$GUEST_USER@$host" 'sudo systemctl reboot' >/dev/null 2>&1 || true
sleep 15
for _ in $(seq 1 60); do
  ssh "${SSH_ARGS[@]}" "$GUEST_USER@$host" 'true' >/dev/null 2>&1 || break
  sleep 2
done
deadline=$((SECONDS + SSH_TIMEOUT))
until ssh "${SSH_ARGS[@]}" "$GUEST_USER@$host" 'nvidia-smi --query-gpu=index --format=csv,noheader' >/dev/null 2>&1; do
  ((SECONDS < deadline)) || die "guest did not return after GPU runtime reboot"
  sleep 10
done

guest_ssh 'nvidia-smi -q' >"$RUN_DIR/evidence/guest/nvidia-smi-q.txt"
guest_ssh 'nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv,noheader' \
  >"$RUN_DIR/evidence/guest/nvidia-inventory.csv"
guest_ssh 'command -v docker >/dev/null && docker info >/dev/null'

guest_ssh "sudo install -d -m 0775 -o '$GUEST_USER' -g '$GUEST_USER' '$REMOTE_INPUT_ROOT' && \
  sudo install -d -m 0775 -o '$GUEST_USER' -g '$GUEST_USER' \
    '$REMOTE_INPUT_ROOT/base' '$REMOTE_INPUT_ROOT/dataset' \
    '$REMOTE_INPUT_ROOT/runs' '$REMOTE_INPUT_ROOT/release' '$REMOTE_INPUT_ROOT/prompts'"

host=$(guest_host); ssh_args
rsync_ssh=$(printf '%q ' ssh "${SSH_ARGS[@]}")
rsync -a --delete -e "$rsync_ssh" "$BASE_MODEL_DIR/" \
  "$GUEST_USER@$host:$REMOTE_INPUT_ROOT/base/"
rsync -a --delete -e "$rsync_ssh" "$DATASET_DIR/" \
  "$GUEST_USER@$host:$REMOTE_INPUT_ROOT/dataset/"
scp "${SSH_ARGS[@]}" "$ROOT/$PROMPTS_FILE" \
  "$GUEST_USER@$host:$REMOTE_INPUT_ROOT/prompts/bear-v1.json"

if [[ -n "${WORKLOAD_ARCHIVE:-}" ]]; then
  archive_name=$(basename "$WORKLOAD_ARCHIVE")
  scp "${SSH_ARGS[@]}" "$WORKLOAD_ARCHIVE" \
    "$GUEST_USER@$host:$REMOTE_INPUT_ROOT/$archive_name"
  guest_ssh "docker load -i '$REMOTE_INPUT_ROOT/$archive_name'"
else
  guest_ssh "docker pull '$WORKLOAD_DIGEST'"
fi
guest_ssh "docker image inspect '$WORKLOAD_DIGEST' --format '{{.Id}}'" \
  >"$RUN_DIR/evidence/guest/workload-image-id.txt"
guest_ssh "docker run --rm --gpus all --network none '$WORKLOAD_DIGEST' \
  python3 /opt/ai-build-tools/scripts/preflight.py" \
  >"$RUN_DIR/evidence/guest/torch-preflight.json"
guest_ssh "docker run --rm --gpus all --network none '$WORKLOAD_DIGEST' \
  python3 /opt/ai-build-tools/scripts/jax_gpu_smoke.py" \
  >"$RUN_DIR/evidence/guest/jax-preflight.json"
