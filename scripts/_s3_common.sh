#!/usr/bin/env bash
# _s3_common.sh — shared S3 helpers for backup.sh and bring_up.sh restore steps.
# Sourced, not executed.

S3_BUCKET="${S3_BUCKET:-data-proj01}"
S3_ENDPOINT="${S3_ENDPOINT:-https://chi.tacc.chameleoncloud.org:7480}"
S3_ACCESS_KEY="${S3_ACCESS_KEY:-8921c48faf83433db2b1439a9b2889fd}"
S3_SECRET_KEY="${S3_SECRET_KEY:-7d1ce78efc5a48019888c9f3fa8ba2dd}"
BACKUP_PREFIX="${BACKUP_PREFIX:-backups}"

export AWS_ACCESS_KEY_ID="$S3_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$S3_SECRET_KEY"

s3_cmd() {
    aws --endpoint-url "$S3_ENDPOINT" s3 "$@"
}

s3_path() {
    local comp="$1"
    echo "s3://${S3_BUCKET}/${BACKUP_PREFIX}/${comp}"
}

s3_has_backup() {
    local comp="$1"
    local pattern="${2:-latest.*}"
    s3_cmd ls "$(s3_path "$comp")/" 2>/dev/null | grep -q "$pattern"
}

ensure_aws_cli() {
    if ! command -v aws >/dev/null 2>&1; then
        log "installing awscli..."
        pip install awscli --break-system-packages -q 2>/dev/null \
            || pip install awscli -q 2>/dev/null
    fi
}
