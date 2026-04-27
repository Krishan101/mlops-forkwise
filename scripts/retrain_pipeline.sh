#!/usr/bin/env bash
# =============================================================================
# ForkWise Retraining Pipeline
# =============================================================================
# Exports feedback from Postgres, merges with base training data,
# retrains GISMo on the GPU instance, evaluates, and promotes if improved.
#
# Usage:
#   From node1:
#     bash scripts/retrain_pipeline.sh <GPU_HOST_IP>
#
#   Example:
#     bash scripts/retrain_pipeline.sh 129.114.25.172
#
# Prerequisites:
#   - GPU instance is running with Docker + gismo-train image built
#   - Training data is already on the GPU instance at ~/training/data/
#   - SSH key access to GPU instance
# =============================================================================

set -euo pipefail

GREEN=$'\e[32m' YELLOW=$'\e[33m' RED=$'\e[31m' RESET=$'\e[0m'
log()  { echo -e "${GREEN}[retrain]${RESET} $*"; }
warn() { echo -e "${YELLOW}[retrain]${RESET} $*"; }
die()  { echo -e "${RED}[retrain]${RESET} $*" >&2; exit 1; }

GPU_HOST="${1:-}"
[[ -n "$GPU_HOST" ]] || die "Usage: bash scripts/retrain_pipeline.sh <GPU_HOST_IP>"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/forkwise-key}"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

S3_ENDPOINT="https://chi.tacc.chameleoncloud.org:7480"
S3_BUCKET="data-proj01"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-8921c48faf83433db2b1439a9b2889fd}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-7d1ce78efc5a48019888c9f3fa8ba2dd}"
export PATH="$HOME/.local/bin:$PATH"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MIN_FEEDBACK=5  # Minimum feedback events to trigger retraining

# =========================================================================
# Step 1: Export feedback from Postgres
# =========================================================================
log "Step 1: Exporting feedback from Postgres..."

FEEDBACK_FILE="$WORKDIR/feedback_export.json"

kubectl -n forkwise-platform exec platform-db-0 -- psql -U forkwise -d forkwise_mlops -t -A -c "
SELECT json_agg(row_to_json(t))
FROM (
    SELECT
        sq.original_ingredient AS original,
        sr.suggested_ingredient AS replacement,
        sq.recipe_id,
        sr.accepted,
        fe.event_type,
        fe.timestamp
    FROM feedback_events fe
    JOIN substitution_queries sq ON fe.query_id = sq.query_id
    JOIN substitution_results sr ON fe.query_id = sr.query_id
        AND fe.suggested_ingredient = sr.suggested_ingredient
    WHERE sr.trained_at IS NULL
    ORDER BY fe.timestamp
) t;
" > "$FEEDBACK_FILE"

# Check if we got valid JSON
FEEDBACK_COUNT=$(python3 -c "
import json, sys
try:
    with open('$FEEDBACK_FILE') as f:
        data = json.load(f)
    if data is None:
        print(0)
    else:
        print(len(data))
except:
    print(0)
")

log "  Exported $FEEDBACK_COUNT untrained feedback events"

if [[ "$FEEDBACK_COUNT" -lt "$MIN_FEEDBACK" ]]; then
    warn "  Only $FEEDBACK_COUNT feedback events (minimum: $MIN_FEEDBACK). Skipping retrain."
    warn "  To force retrain anyway, set MIN_FEEDBACK=0"
    exit 0
fi

# =========================================================================
# Step 2: Convert feedback to training format
# =========================================================================
log "Step 2: Converting feedback to training tuples..."

FEEDBACK_TUPLES="$WORKDIR/feedback_tuples.json"

python3 << CONVERT_EOF
import json

with open("$FEEDBACK_FILE") as f:
    feedback = json.load(f)

if feedback is None:
    feedback = []

# Convert accepted feedback to positive substitution tuples
tuples = []
for fb in feedback:
    if fb.get("event_type") == "accept" and fb.get("accepted") == 1:
        tuples.append({
            "recipe_id": fb.get("recipe_id", "feedback"),
            "original": fb["original"],
            "replacement": fb["replacement"],
        })

with open("$FEEDBACK_TUPLES", "w") as f:
    json.dump(tuples, f)

print(f"  {len(tuples)} positive tuples from feedback")
CONVERT_EOF

# =========================================================================
# Step 3: Merge feedback with base training data
# =========================================================================
log "Step 3: Merging feedback with base training data..."

MERGED_TRAIN="$WORKDIR/train_merged.json"

python3 << MERGE_EOF
import json

# Load base training data from S3 (cached on GPU, but we have a copy)
# We'll download it here for the merge, then send merged file to GPU
import subprocess
result = subprocess.run([
    "aws", "--endpoint-url", "$S3_ENDPOINT",
    "s3", "cp", "s3://$S3_BUCKET/data/raw/recipe1msubs/train.json",
    "$WORKDIR/base_train.json"
], capture_output=True, text=True)

with open("$WORKDIR/base_train.json") as f:
    base = json.load(f)

with open("$FEEDBACK_TUPLES") as f:
    feedback_tuples = json.load(f)

# Merge: base + feedback
merged = base + feedback_tuples

with open("$MERGED_TRAIN", "w") as f:
    json.dump(merged, f)

print(f"  Base: {len(base)}, Feedback: {len(feedback_tuples)}, Merged: {len(merged)}")
MERGE_EOF

# =========================================================================
# Step 4: Upload merged data to GPU instance
# =========================================================================
log "Step 4: Uploading merged training data to GPU instance..."

scp $SSH_OPTS "$MERGED_TRAIN" cc@"$GPU_HOST":~/training/data/recipe1msubs/train.json
log "  Uploaded merged train.json to GPU instance"

# =========================================================================
# Step 5: Run retraining on GPU
# =========================================================================
log "Step 5: Running retraining on GPU instance..."

# Get current model's test MRR for comparison
CURRENT_MRR=$(aws --endpoint-url "$S3_ENDPOINT" \
    s3 cp "s3://$S3_BUCKET/models/v2/metadata.json" - 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('test_mrr', 0))" 2>/dev/null \
    || echo "0")
log "  Current deployed model test MRR: $CURRENT_MRR"

MLFLOW_URI="http://$(curl -s --max-time 5 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo '129.114.26.133'):30500"

ssh $SSH_OPTS cc@"$GPU_HOST" << RETRAIN_SSH
set -e
export PATH="\$HOME/.local/bin:\$PATH"

# Clear previous output
rm -f ~/training/output/*

# Run training with warm start settings (fewer epochs since we're fine-tuning)
docker run --rm --gpus all \
  -v ~/training/data:/app/data \
  -v ~/training/output:/app/output \
  gismo-train \
  --data-dir /app/data \
  --output-dir /app/output \
  --max-epochs 30 \
  --patience 8 \
  --eval-candidates 100 \
  --batch-size 512 \
  --num-neg 32 \
  --emb-dim 300 \
  --lr 2e-4 \
  --mlflow-uri $MLFLOW_URI

echo "RETRAIN_DONE"
RETRAIN_SSH

log "  Retraining completed on GPU"

# =========================================================================
# Step 6: Generate missing artifacts (embeddings, vocab, metadata)
# =========================================================================
log "Step 6: Generating artifacts on GPU..."

scp $SSH_OPTS "$REPO_ROOT/scripts/fixup_artifacts.py" cc@"$GPU_HOST":~/training/src/fixup_artifacts.py 2>/dev/null || true

ssh $SSH_OPTS cc@"$GPU_HOST" << FIXUP_SSH
set -e
docker run --rm --gpus all \
  --entrypoint python \
  -v ~/training/data:/home/cc/training/data \
  -v ~/training/output:/home/cc/training/output \
  -v ~/training/src:/home/cc/training/src \
  gismo-train \
  /home/cc/training/src/fixup_artifacts.py

echo "FIXUP_DONE"
FIXUP_SSH

log "  Artifacts generated"

# =========================================================================
# Step 7: Get new model's test MRR and compare
# =========================================================================
log "Step 7: Evaluating new model..."

NEW_MRR=$(ssh $SSH_OPTS cc@"$GPU_HOST" "cat ~/training/output/metadata.json" \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('test_mrr', 0))")

log "  New model test MRR: $NEW_MRR"
log "  Current model test MRR: $CURRENT_MRR"

PROMOTE=$(python3 -c "print('yes' if float('$NEW_MRR') >= float('$CURRENT_MRR') else 'no')")

if [[ "$PROMOTE" == "no" ]]; then
    warn "  New model ($NEW_MRR) did not improve over current ($CURRENT_MRR)"
    warn "  Keeping current model. Retrain artifacts saved on GPU instance."
    exit 0
fi

# =========================================================================
# Step 8: Promote new model
# =========================================================================
log "Step 8: Promoting new model..."

# Backup current model as previous
log "  Backing up current model as previous..."
for f in gismo_decoder.onnx ingredient_embeddings.npy vocab.json metadata.json; do
    base="${f%.*}"
    ext="${f##*.}"
    aws --endpoint-url "$S3_ENDPOINT" s3 cp \
        "s3://$S3_BUCKET/models/v2/$f" \
        "s3://$S3_BUCKET/models/v2/${base}_previous.${ext}" \
        --quiet 2>/dev/null || true
done

# Upload new model
log "  Uploading new model artifacts..."
for f in gismo_decoder.onnx ingredient_embeddings.npy vocab.json metadata.json gismo_best.pth; do
    ssh $SSH_OPTS cc@"$GPU_HOST" "cat ~/training/output/$f" \
        | aws --endpoint-url "$S3_ENDPOINT" s3 cp - "s3://$S3_BUCKET/models/v2/$f" --quiet
done

log "  Model uploaded to S3"

# =========================================================================
# Step 9: Restart substitution API to pick up new model
# =========================================================================
log "Step 9: Restarting substitution API..."

kubectl -n forkwise-platform rollout restart deployment/substitution-api
kubectl -n forkwise-platform rollout status deployment/substitution-api --timeout=3m

log "  Substitution API restarted"

# =========================================================================
# Step 10: Mark feedback as consumed
# =========================================================================
log "Step 10: Marking feedback as consumed..."

kubectl -n forkwise-platform exec platform-db-0 -- psql -U forkwise -d forkwise_mlops -c \
    "UPDATE substitution_results SET trained_at = NOW() WHERE trained_at IS NULL AND accepted IS NOT NULL;" \
    >/dev/null

log "  Feedback marked as trained"

# =========================================================================
# Summary
# =========================================================================
log ""
log "============================================"
log "  Retraining pipeline complete!"
log "============================================"
log "  Previous MRR: $CURRENT_MRR"
log "  New MRR:      $NEW_MRR"
log "  Feedback used: $FEEDBACK_COUNT events"
log "  Model promoted and deployed"
log "  Timestamp: $TIMESTAMP"
log "============================================"
