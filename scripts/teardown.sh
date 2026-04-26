#!/usr/bin/env bash
# ForkWise — full teardown.
# Backs up all state to S3, then deletes K8s namespaces.
# Runs on node1. After this, run terraform destroy from the provision notebook.

set -euo pipefail

GREEN=$'\e[32m' YELLOW=$'\e[33m' RED=$'\e[31m' RESET=$'\e[0m'
log()  { echo -e "${GREEN}[teardown]${RESET} $*"; }
warn() { echo -e "${YELLOW}[teardown]${RESET} $*"; }
die()  { echo -e "${RED}[teardown]${RESET} $*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- Confirmation ----
cat <<EOF
${RED}============================================${RESET}
  WARNING — full teardown
${RED}============================================${RESET}
This will:
  1. Backup all state to S3 (mealie-db, platform-db, qdrant, mealie-data)
  2. DELETE K8s namespaces: forkwise-app, forkwise-platform, monitoring

After this, run terraform destroy from the provision notebook to free VMs.

EOF
read -r -p "Type 'destroy' to continue: " ans
[[ "$ans" == "destroy" ]] || die "aborted"

# ---- Backup first ----
log "running backup before teardown..."
if bash "$REPO_ROOT/scripts/backup.sh"; then
    log "backup OK"
else
    warn "backup had errors — proceeding with teardown anyway"
fi

# ---- Delete namespaces ----
log "deleting K8s namespaces..."
for ns in forkwise-app forkwise-platform monitoring; do
    kubectl delete namespace "$ns" --wait=false 2>/dev/null || true
done

# ---- Wait for deletion ----
log "waiting for namespace deletion (up to 3m)..."
for i in {1..90}; do
    remaining=$(kubectl get ns forkwise-app forkwise-platform monitoring 2>&1 | grep -c "Active" || true)
    if [[ "$remaining" -eq 0 ]]; then
        break
    fi
    sleep 2
done

log ""
log "============================================"
log "  teardown complete"
log "============================================"
log ""
log "State backed up to: s3://data-proj01/backups/"
log "To destroy VMs: run terraform destroy from the provision notebook"
log "To rebuild: provision VMs, then run: bash scripts/bring_up.sh"
