#!/usr/bin/env bash
# ForkWise — backup persistent state to S3 before teardown.
# Runs on node1. Components: mealie-db, platform-db, qdrant, mealie-data PVC.

set -uo pipefail

GREEN=$'\e[32m' YELLOW=$'\e[33m' RED=$'\e[31m' RESET=$'\e[0m'
log()  { echo -e "${GREEN}[backup]${RESET} $*"; }
warn() { echo -e "${YELLOW}[backup]${RESET} $*"; }
die()  { echo -e "${RED}[backup]${RESET} $*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TS="$(date +%Y%m%d-%H%M)"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

source "$REPO_ROOT/scripts/_s3_common.sh"
ensure_aws_cli

FAILED=()

s3_upload() {
    local local_file="$1" comp="$2" ext="$3"
    local dest="$(s3_path "$comp")"
    log "  uploading to ${dest}/${TS}.${ext}"
    s3_cmd cp "$local_file" "${dest}/${TS}.${ext}" --quiet \
        || { warn "  upload of ${comp} timestamped copy FAILED"; return 1; }
    s3_cmd cp "$local_file" "${dest}/latest.${ext}" --quiet \
        || { warn "  upload of ${comp} latest pointer FAILED"; return 1; }
    return 0
}

# ---- 1. Mealie DB (postgres) ----
backup_mealie_db() {
    log "=== mealie-db ==="
    local ns=forkwise-app
    local pod
    pod=$(kubectl -n "$ns" get pod -l app=mealie-db -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    [[ -n "$pod" ]] || { warn "no mealie-db pod; skipping"; return 1; }
    local out="$WORKDIR/mealie-db.sql.gz"
    log "  dumping mealie database..."
    kubectl -n "$ns" exec "$pod" -- pg_dump -U mealie -d mealie 2>/dev/null \
        | gzip > "$out" \
        || { warn "pg_dump FAILED"; return 1; }
    log "  size: $(du -h "$out" | cut -f1)"
    s3_upload "$out" mealie-db sql.gz
}

# ---- 2. Platform DB (postgres) ----
backup_platform_db() {
    log "=== platform-db ==="
    local ns=forkwise-platform
    local pod="platform-db-0"
    kubectl -n "$ns" get pod "$pod" >/dev/null 2>&1 \
        || { warn "no platform-db pod; skipping"; return 1; }
    local out="$WORKDIR/platform-db.sql.gz"
    log "  dumping all databases (forkwise_mlops + mlflow)..."
    kubectl -n "$ns" exec "$pod" -- pg_dumpall --clean --if-exists -U forkwise 2>/dev/null \
        | gzip > "$out" \
        || { warn "pg_dumpall FAILED"; return 1; }
    log "  size: $(du -h "$out" | cut -f1)"
    s3_upload "$out" platform-db sql.gz
}

# ---- 3. Qdrant (snapshot API) ----
backup_qdrant() {
    log "=== qdrant ==="
    local ns=forkwise-platform
    local coll="ingredient_embeddings"

    kubectl -n "$ns" port-forward svc/qdrant 16333:6333 >/dev/null 2>&1 &
    local pf_pid=$!
    trap "kill $pf_pid 2>/dev/null" RETURN
    sleep 3

    log "  creating snapshot..."
    local snap_name
    snap_name=$(curl -sS -X POST "http://localhost:16333/collections/${coll}/snapshots" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['name'])" 2>/dev/null)
    [[ -n "$snap_name" && "$snap_name" != "None" ]] \
        || { warn "qdrant snapshot create FAILED"; kill $pf_pid 2>/dev/null; return 1; }
    log "  snapshot: $snap_name"

    local out="$WORKDIR/qdrant.snapshot"
    curl -sS -o "$out" "http://localhost:16333/collections/${coll}/snapshots/${snap_name}" \
        || { warn "qdrant download FAILED"; kill $pf_pid 2>/dev/null; return 1; }

    curl -sS -X DELETE "http://localhost:16333/collections/${coll}/snapshots/${snap_name}" >/dev/null 2>&1 || true
    kill $pf_pid 2>/dev/null

    log "  size: $(du -h "$out" | cut -f1)"
    s3_upload "$out" qdrant snapshot
}

# ---- 4. Mealie data PVC ----
backup_mealie_data() {
    log "=== mealie-data ==="
    local ns=forkwise-app
    local pod
    pod=$(kubectl -n "$ns" get pod -l app=mealie -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    [[ -n "$pod" ]] || { warn "no mealie pod; skipping"; return 1; }
    local out="$WORKDIR/mealie-data.tar.gz"
    log "  tarring /app/data..."
    kubectl -n "$ns" exec "$pod" -- tar -czf - -C /app data > "$out" \
        || { warn "tar FAILED"; return 1; }
    log "  size: $(du -h "$out" | cut -f1)"
    s3_upload "$out" mealie-data tar.gz
}

log "backup timestamp: $TS"
log "target: s3://${S3_BUCKET}/${BACKUP_PREFIX}/"
echo

for fn in backup_mealie_db backup_platform_db backup_qdrant backup_mealie_data; do
    if ! "$fn"; then FAILED+=("$fn"); fi
    echo
done

echo "==============================================="
if [[ ${#FAILED[@]} -eq 0 ]]; then
    log "backup complete — all 4 components OK"
else
    warn "backup finished with ${#FAILED[@]} FAILED component(s): ${FAILED[*]}"
fi
echo "==============================================="
