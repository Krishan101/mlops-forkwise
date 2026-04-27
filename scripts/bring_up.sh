#!/usr/bin/env bash
# ForkWise — cluster bring-up script with automatic S3 restore.
# Runs on node1 after kubespray + post-kubespray setup.
# If backups exist in S3, restores them. Otherwise does a fresh deploy.

set -euo pipefail

GREEN=$'\e[32m' YELLOW=$'\e[33m' RED=$'\e[31m' RESET=$'\e[0m'
log()  { echo -e "${GREEN}[bring_up]${RESET} $*"; }
warn() { echo -e "${YELLOW}[bring_up]${RESET} $*"; }
die()  { echo -e "${RED}[bring_up]${RESET} $*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

log "repo root: $REPO_ROOT"

# Load S3 helpers
source "$REPO_ROOT/scripts/_s3_common.sh"
ensure_aws_cli

# --- 0. IPv6 fix ---
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
    ["allow-30300"]=30300
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

# --- Check for existing backups ---
HAS_BACKUPS=false
if s3_has_backup "platform-db" "latest.sql.gz"; then
    HAS_BACKUPS=true
    log "S3 backups detected — will restore after deploy"
else
    log "no S3 backups found — fresh deploy"
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

# Wait for postgres to be ready
for i in {1..30}; do
    if kubectl -n forkwise-platform exec platform-db-0 -- pg_isready -U forkwise >/dev/null 2>&1; then break; fi
    sleep 2
done

# Restore platform DB if backup exists
if $HAS_BACKUPS && s3_has_backup "platform-db" "latest.sql.gz"; then
    log "restoring platform-db from S3..."
    s3_cmd cp "$(s3_path platform-db)/latest.sql.gz" "$WORKDIR/platform-db.sql.gz" --quiet \
        && gunzip -c "$WORKDIR/platform-db.sql.gz" \
            | kubectl -n forkwise-platform exec -i platform-db-0 -- psql -U forkwise -d postgres -q >/dev/null 2>&1 \
        && log "  platform-db restore OK" \
        || warn "  platform-db restore FAILED (continuing with fresh DB)"
fi

# --- 3. Qdrant ---
log "deploying qdrant..."
kubectl apply -f "$REPO_ROOT/k8s/platform/qdrant.yaml"
log "waiting for qdrant..."
kubectl -n forkwise-platform rollout status deployment/qdrant --timeout=3m

# Restore qdrant if backup exists
if $HAS_BACKUPS && s3_has_backup "qdrant" "latest.snapshot"; then
    log "restoring qdrant from S3..."
    s3_cmd cp "$(s3_path qdrant)/latest.snapshot" "$WORKDIR/qdrant.snapshot" --quiet
    if [[ -f "$WORKDIR/qdrant.snapshot" ]]; then
        kubectl -n forkwise-platform port-forward svc/qdrant 16333:6333 >/dev/null 2>&1 &
        pf_pid=$!
        sleep 3
        curl -sS -X POST \
            -F "snapshot=@${WORKDIR}/qdrant.snapshot" \
            "http://localhost:16333/collections/ingredient_embeddings/snapshots/upload?priority=snapshot" >/dev/null 2>&1 \
            && log "  qdrant restore OK" \
            || warn "  qdrant restore FAILED (feature-worker will rebuild)"
        kill $pf_pid 2>/dev/null || true
    fi
fi

# --- 4. Mealie ---
log "deploying mealie + mealie-db..."
kubectl apply -f "$REPO_ROOT/k8s/mealie/mealie.yaml"
log "waiting for mealie-db..."
kubectl -n forkwise-app rollout status deployment/mealie-db --timeout=3m

# Wait for mealie-db postgres to be ready
for i in {1..15}; do
    if kubectl -n forkwise-app exec deploy/mealie-db -- psql -U mealie -d mealie -c "SELECT 1" >/dev/null 2>&1; then break; fi
    sleep 2
done

# Restore mealie DB if backup exists
if $HAS_BACKUPS && s3_has_backup "mealie-db" "latest.sql.gz"; then
    log "restoring mealie-db from S3..."
    s3_cmd cp "$(s3_path mealie-db)/latest.sql.gz" "$WORKDIR/mealie-db.sql.gz" --quiet \
        && gunzip -c "$WORKDIR/mealie-db.sql.gz" \
            | kubectl -n forkwise-app exec -i deploy/mealie-db -- psql -U mealie -d mealie -q >/dev/null 2>&1 \
        && log "  mealie-db restore OK" \
        || warn "  mealie-db restore FAILED (starting fresh)"
fi

log "waiting for mealie..."
kubectl -n forkwise-app rollout status deployment/mealie --timeout=5m

# Restore mealie data PVC if backup exists
if $HAS_BACKUPS && s3_has_backup "mealie-data" "latest.tar.gz"; then
    log "restoring mealie-data PVC from S3..."
    s3_cmd cp "$(s3_path mealie-data)/latest.tar.gz" "$WORKDIR/mealie-data.tar.gz" --quiet
    if [[ -f "$WORKDIR/mealie-data.tar.gz" ]]; then
        kubectl -n forkwise-app exec -i deploy/mealie -- tar -xzf - -C /app < "$WORKDIR/mealie-data.tar.gz" \
            && log "  mealie-data restore OK" \
            || warn "  mealie-data restore FAILED"
    fi
fi

# Fix: populate reference_id for ingredients
log "fixing ingredient reference_ids in mealie DB..."
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
kubectl -n forkwise-platform exec platform-db-0 -- psql -U forkwise -d forkwise_mlops -tAc \
    "SELECT 1 FROM pg_database WHERE datname='mlflow'" | grep -q 1 \
    || kubectl -n forkwise-platform exec platform-db-0 -- psql -U forkwise -d forkwise_mlops -c "CREATE DATABASE mlflow" >/dev/null

log "deploying mlflow..."
kubectl apply -f "$REPO_ROOT/k8s/platform/mlflow.yaml"
log "waiting for mlflow..."
kubectl -n forkwise-platform rollout status deployment/mlflow --timeout=3m

# --- 8. Substitution API (with GISMo model) ---
log "deploying substitution-api (with GISMo ONNX reranking)..."
kubectl apply -f "$REPO_ROOT/k8s/platform/substitution-api.yaml"
log "waiting for substitution-api..."
kubectl -n forkwise-platform rollout status deployment/substitution-api --timeout=5m

# Add service label for Prometheus ServiceMonitor
kubectl -n forkwise-platform label svc substitution-api app=substitution-api --overwrite 2>/dev/null || true

# --- 9. Monitoring (Prometheus + Grafana) ---
log "setting up monitoring..."
bash "$REPO_ROOT/scripts/setup_monitoring.sh"

# --- 10. GISMo-specific resources ---
log "deploying GISMo resources (ServiceMonitor, CronJob)..."

# Apply ServiceMonitor for substitution API metrics scraping
if [[ -f "$REPO_ROOT/k8s/monitoring/servicemonitor.yaml" ]]; then
    kubectl apply -f "$REPO_ROOT/k8s/monitoring/servicemonitor.yaml"
    log "  ServiceMonitor applied"
fi

# Create SSH key secret for retraining CronJob
if [[ -f "$HOME/.ssh/forkwise-key" ]]; then
    kubectl -n forkwise-platform create secret generic retrain-ssh-key \
        --from-file=forkwise-key="$HOME/.ssh/forkwise-key" \
        --dry-run=client -o yaml | kubectl apply -f -
    log "  retrain SSH key secret created"
else
    warn "  SSH key not found at ~/.ssh/forkwise-key — CronJob won't be able to SSH to GPU"
fi

# Apply retraining CronJob + RBAC
if [[ -f "$REPO_ROOT/k8s/platform/retrain-cronjob.yaml" ]]; then
    kubectl apply -f "$REPO_ROOT/k8s/platform/retrain-cronjob.yaml"
    log "  retraining CronJob applied"
fi

# --- Summary ---
log ""
log "============================================"
if $HAS_BACKUPS; then
    log "  bring-up complete (restored from S3)"
else
    log "  bring-up complete (fresh deploy)"
fi
log "============================================"

NODE1_IP="$(curl -s --max-time 5 https://api.ipify.org || echo '<unknown>')"

log ""
log "--- Services ---"
log "Mealie        : http://${NODE1_IP}:30900"
log "Substitution  : http://${NODE1_IP}:30808/substitute"
log ""
log "--- Platform ---"
log "Platform DB   : platform-db.forkwise-platform:5432"
log "Qdrant        : qdrant.forkwise-platform:6333"
log "MLflow        : http://${NODE1_IP}:30500"
log "Grafana       : http://${NODE1_IP}:30300  (admin / forkwise-admin)"
log "Ingest API    : polling mealie every 30s"
log "Feature Worker: polling feature_jobs every 5s"
log ""
log "--- GISMo ---"
log "Model         : ONNX decoder from s3://data-proj01/models/v2/"
log "Reranking     : enabled (GISMO_ENABLED=true)"
log "CronJob       : retrain-check (every 6h, threshold 50 feedback events)"
log "ServiceMonitor: substitution-api -> Prometheus"
log ""
log "--- Model Info ---"
log "  curl http://${NODE1_IP}:30808/admin/model-info"
