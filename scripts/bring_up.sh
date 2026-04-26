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

# --- 7. Substitution API ---
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
log "Ingest API    : polling mealie every 30s"
log "Feature Worker: polling feature_jobs every 5s"
log ""
log "--- Next Steps ---"
log "1. Open Mealie, create an account, add recipes"
log "2. Ingest service will auto-detect and store them"
log "3. Feature worker will compute embeddings into Qdrant"
