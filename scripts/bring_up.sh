#!/usr/bin/env bash
# ForkWise — cluster bring-up script.
# Runs on node1 after kubespray + post-kubespray setup (Step J in provision notebook).
# Assumes: kubectl works, helm installed, local-path-provisioner + metrics-server deployed.

set -euo pipefail

GREEN=$'\e[32m' YELLOW=$'\e[33m' RED=$'\e[31m' RESET=$'\e[0m'
log()  { echo -e "${GREEN}[bring_up]${RESET} $*"; }
warn() { echo -e "${YELLOW}[bring_up]${RESET} $*"; }
die()  { echo -e "${RED}[bring_up]${RESET} $*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

log "repo root: $REPO_ROOT"

# --- 0. IPv6 fix (Chameleon IPv6 doesn't route; breaks HuggingFace, NLTK, pip) ---
log "disabling IPv6 (except loopback for kubectl)..."
sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1 >/dev/null
sudo sysctl -w net.ipv6.conf.lo.disable_ipv6=0 >/dev/null

# --- 0b. Security Groups ---
log "ensuring security groups exist and are attached..."
OS_CLOUD="${OS_CLOUD:-openstack}"
declare -A SG_PORTS=(
    ["allow-ssh"]=22
    ["allow-30900"]=30900
    ["allow-30808"]=30808
    ["allow-30500"]=30500
)

existing_sgs=$(openstack --os-cloud "$OS_CLOUD" security group list -f value -c Name 2>/dev/null || echo "")
for sg_name in "${!SG_PORTS[@]}"; do
    port="${SG_PORTS[$sg_name]}"
    if ! grep -qx "$sg_name" <<<"$existing_sgs"; then
        log "  creating SG $sg_name (tcp/$port)"
        openstack --os-cloud "$OS_CLOUD" security group create "$sg_name" --description "auto: forkwise" >/dev/null 2>&1 || true
    fi
    openstack --os-cloud "$OS_CLOUD" security group rule create --protocol tcp --dst-port "$port" --remote-ip 0.0.0.0/0 "$sg_name" >/dev/null 2>&1 || true
done

# Attach SGs to node1's sharednet1 port
NODE1_SHAREDNET_PORT=$(openstack --os-cloud "$OS_CLOUD" port list --server "$(hostname)" -f value -c ID 2>/dev/null | head -1)
if [[ -n "$NODE1_SHAREDNET_PORT" ]]; then
    for sg_name in "${!SG_PORTS[@]}"; do
        sg_id=$(openstack --os-cloud "$OS_CLOUD" security group list -f value -c ID -c Name 2>/dev/null | grep " ${sg_name}$" | head -1 | awk '{print $1}')
        if [[ -n "$sg_id" ]]; then
            openstack --os-cloud "$OS_CLOUD" port set "$NODE1_SHAREDNET_PORT" --security-group "$sg_id" >/dev/null 2>&1 || true
        fi
    done
    log "  SGs attached to node1 port"
else
    warn "  could not find node1 port — attach SGs manually"
fi

# --- 1. Namespaces ---
log "creating namespaces..."
kubectl apply -f "$REPO_ROOT/k8s/mealie/namespace.yaml"
kubectl apply -f "$REPO_ROOT/k8s/platform/namespace.yaml"

# --- 2. Platform DB ---
log "deploying platform postgres..."
kubectl -n forkwise-platform create configmap platform-db-init \
    --from-file=init.sql="$REPO_ROOT/db/init.sql" \
    --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f "$REPO_ROOT/k8s/platform/postgres.yaml"
log "waiting for platform-db..."
kubectl -n forkwise-platform rollout status statefulset/platform-db --timeout=3m

# --- 3. Qdrant ---
log "deploying qdrant..."
kubectl apply -f "$REPO_ROOT/k8s/platform/qdrant.yaml"
log "waiting for qdrant..."
kubectl -n forkwise-platform rollout status deployment/qdrant --timeout=3m

# --- 4. Mealie ---
log "deploying mealie + mealie-db..."
kubectl apply -f "$REPO_ROOT/k8s/mealie/mealie.yaml"
log "waiting for mealie-db..."
kubectl -n forkwise-app rollout status deployment/mealie-db --timeout=3m
log "waiting for mealie..."
kubectl -n forkwise-app rollout status deployment/mealie --timeout=5m

# Fix: populate reference_id for ingredients (Mealie fork bug — NULLs break substitution lookups)
log "fixing ingredient reference_ids in mealie DB..."
for i in {1..15}; do
    if kubectl -n forkwise-app exec deploy/mealie-db -- psql -U mealie -d mealie -c "SELECT 1" >/dev/null 2>&1; then break; fi
    sleep 2
done
kubectl -n forkwise-app exec deploy/mealie-db -- psql -U mealie -d mealie -c \
    "UPDATE recipes_ingredients SET reference_id = gen_random_uuid() WHERE reference_id IS NULL" >/dev/null 2>&1 || warn "reference_id fix failed (may not be needed)"

# --- 5. Ingest Service ---
log "deploying ingest-api..."
kubectl apply -f "$REPO_ROOT/k8s/platform/ingest-api.yaml"
log "waiting for ingest-api..."
kubectl -n forkwise-platform rollout status deployment/ingest-api --timeout=3m

# --- 6. Feature Worker ---
log "deploying feature-worker..."
kubectl apply -f "$REPO_ROOT/k8s/platform/feature-worker.yaml"
log "waiting for feature-worker..."
kubectl -n forkwise-platform rollout status deployment/feature-worker --timeout=3m

# --- 7. MLflow ---
log "ensuring mlflow database exists in postgres..."
for i in {1..30}; do
    if kubectl -n forkwise-platform exec platform-db-0 -- pg_isready -U forkwise >/dev/null 2>&1; then break; fi
    sleep 2
done
kubectl -n forkwise-platform exec platform-db-0 -- psql -U forkwise -d forkwise_mlops -tAc \
    "SELECT 1 FROM pg_database WHERE datname='mlflow'" | grep -q 1 \
    || kubectl -n forkwise-platform exec platform-db-0 -- psql -U forkwise -d forkwise_mlops -c "CREATE DATABASE mlflow" >/dev/null

log "deploying mlflow..."
kubectl apply -f "$REPO_ROOT/k8s/platform/mlflow.yaml"
log "waiting for mlflow..."
kubectl -n forkwise-platform rollout status deployment/mlflow --timeout=3m

# --- 8. Substitution API ---
log "deploying substitution-api..."
kubectl apply -f "$REPO_ROOT/k8s/platform/substitution-api.yaml"
log "waiting for substitution-api..."
kubectl -n forkwise-platform rollout status deployment/substitution-api --timeout=3m

# --- Summary ---
log ""
log "============================================"
log "  bring-up complete"
log "============================================"

# Detect floating IP
NODE1_IP="$(curl -s --max-time 5 https://api.ipify.org || echo '<unknown>')"

log ""
log "--- Services ---"
log "Mealie        : http://${NODE1_IP}:30900"
log "               default login: create account on first visit"
log "Substitution  : http://${NODE1_IP}:30808/substitute"
log "  try: curl -X POST http://${NODE1_IP}:30808/substitute -H 'Content-Type: application/json' -d '{\"ingredient\":\"butter\",\"recipe_name\":\"Classic Pancakes\",\"top_k\":5}'"
log ""
log "--- Platform ---"
log "Platform DB   : platform-db.forkwise-platform:5432"
log "Qdrant        : qdrant.forkwise-platform:6333"
log "MLflow        : http://${NODE1_IP}:30500"
log "Ingest API    : polling mealie every 30s"
log "Feature Worker: polling feature_jobs every 5s"
log ""
log "--- Next Steps ---"
log "1. Open Mealie, create an account, add recipes"
log "2. Ingest service will auto-detect and store them"
log "3. Feature worker will compute embeddings into Qdrant"
