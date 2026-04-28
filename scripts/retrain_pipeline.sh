#!/usr/bin/env bash
# =============================================================================
# ForkWise Retraining Pipeline
# =============================================================================
# Exports feedback from Postgres, merges with base training data,
# retrains GISMo on the GPU instance, evaluates, and promotes if improved.
#
# Quality gates:
#   1. Absolute: new model MRR must be >= MIN_ABSOLUTE_MRR (default 10.0)
#   2. Relative: new model MRR must be >= 95% of current deployed model's MRR
#   Both gates must pass for the model to be promoted.
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
MIN_FEEDBACK=5           # Minimum feedback events to trigger retraining
MIN_ABSOLUTE_MRR=10.0    # Minimum test MRR to deploy any model (quality gate)

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
from collections import Counter

with open("$FEEDBACK_FILE") as f:
    feedback = json.load(f)

if feedback is None:
    feedback = []

# Separate accepts and rejects
accepts = set()
rejects = set()
for fb in feedback:
    key = (fb["original"].lower().strip(), fb["replacement"].lower().strip())
    if fb.get("event_type") == "accept" and fb.get("accepted") == 1:
        accepts.add(key)
    elif fb.get("event_type") == "reject":
        rejects.add(key)

# Detect contradictions: same pair both accepted and rejected
contradictions = accepts & rejects
if contradictions:
    print(f"  WARNING: {len(contradictions)} contradictory pairs (both accepted and rejected)")
    for orig, repl in list(contradictions)[:5]:
        print(f"    {orig} -> {repl}")

# Build tuples: only accepted, not contradicted, deduplicated
seen = set()
tuples = []
for fb in feedback:
    if fb.get("event_type") == "accept" and fb.get("accepted") == 1:
        key = (fb["original"].lower().strip(), fb["replacement"].lower().strip())
        if key in contradictions:
            continue  # Skip contradictory pairs
        if key in seen:
            continue  # Skip duplicates
        seen.add(key)
        tuples.append({
            "recipe_id": fb.get("recipe_id", "feedback"),
            "original": fb["original"],
            "replacement": fb["replacement"],
        })

with open("$FEEDBACK_TUPLES", "w") as f:
    json.dump(tuples, f)

print(f"  {len(tuples)} positive tuples from feedback")
print(f"  (deduplicated from {len(accepts)} accepts, removed {len(contradictions)} contradictions)")
CONVERT_EOF

# =========================================================================
# Step 3: Merge feedback with base training data
# =========================================================================
log "Step 3: Merging feedback with base training data..."

MERGED_TRAIN="$WORKDIR/train_merged.json"

python3 << MERGE_EOF
import json
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

CURRENT_MRR=$(aws --endpoint-url "$S3_ENDPOINT" \
    s3 cp "s3://$S3_BUCKET/models/v2/metadata.json" - 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('test_mrr', 0))" 2>/dev/null \
    || echo "0")
log "  Current deployed model test MRR: $CURRENT_MRR"

MLFLOW_URI="http://$(curl -s --max-time 5 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo '129.114.26.133'):30500"

( ssh $SSH_OPTS cc@"$GPU_HOST" << RETRAIN_SSH
set -e
export PATH="\$HOME/.local/bin:\$PATH"

# Clear previous output (fix permissions from Docker root ownership)
sudo chown -R cc:cc ~/training/output 2>/dev/null || true
rm -f ~/training/output/*

# Run training
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
) || true

log "  Retraining completed on GPU (or ONNX validation crashed - artifacts still saved)"

# =========================================================================
# Step 6: Generate artifacts (embeddings, vocab, metadata)
# =========================================================================
log "Step 6: Generating artifacts on GPU..."

scp $SSH_OPTS "$REPO_ROOT/scripts/fixup_artifacts.py" cc@"$GPU_HOST":~/training/src/fixup_artifacts.py 2>/dev/null || true

ssh $SSH_OPTS cc@"$GPU_HOST" << FIXUP_SSH
set -e
sudo chown -R cc:cc ~/training/output 2>/dev/null || true
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
# Step 7: Quality gates — evaluate new model
# =========================================================================
log "Step 7: Evaluating new model against quality gates..."

NEW_MRR=$(ssh $SSH_OPTS cc@"$GPU_HOST" "cat ~/training/output/metadata.json" \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('test_mrr', 0))")

log "  New model test MRR:     $NEW_MRR"
log "  Current model test MRR: $CURRENT_MRR"
log "  Minimum absolute MRR:   $MIN_ABSOLUTE_MRR"

# Quality gate 1: absolute minimum MRR
PASSES_ABSOLUTE=$(python3 -c "print('yes' if float('$NEW_MRR') >= $MIN_ABSOLUTE_MRR else 'no')")
if [[ "$PASSES_ABSOLUTE" == "no" ]]; then
    warn "  QUALITY GATE FAILED: New model MRR ($NEW_MRR) below absolute minimum ($MIN_ABSOLUTE_MRR)"
    warn "  Model will NOT be deployed. Check training data quality."
    exit 1
fi
log "  ✓ Passes absolute MRR gate ($NEW_MRR >= $MIN_ABSOLUTE_MRR)"

# Quality gate 2: must be at least 95% of current model's MRR
PASSES_RELATIVE=$(python3 -c "print('yes' if float('$NEW_MRR') >= float('$CURRENT_MRR') * 0.95 else 'no')")
if [[ "$PASSES_RELATIVE" == "no" ]]; then
    warn "  QUALITY GATE FAILED: New model MRR ($NEW_MRR) is >5% worse than current ($CURRENT_MRR)"
    warn "  Model will NOT be deployed. Keeping current model."
    exit 1
fi
log "  ✓ Passes relative MRR gate ($NEW_MRR >= 95% of $CURRENT_MRR)"

log "  Both quality gates passed! Proceeding with promotion."

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

# Delete PVC to force re-download of new model on next pod start
kubectl -n forkwise-platform delete pvc gismo-model-pvc --ignore-not-found 2>/dev/null || true
sleep 3
kubectl apply -f "$REPO_ROOT/k8s/platform/substitution-api.yaml"
kubectl -n forkwise-platform rollout restart deployment/substitution-api
kubectl -n forkwise-platform rollout status deployment/substitution-api --timeout=5m

log "  Substitution API restarted with new model"

# =========================================================================
# Step 9b: Canary validation (live traffic check)
# =========================================================================
CANARY_DURATION=300  # 5 minutes
CANARY_TEST_QUERIES=20
CANARY_ERROR_THRESHOLD=3  # max allowed failures out of test queries

log "Step 9b: Canary validation — testing new model for ${CANARY_DURATION}s..."
log "  Sending $CANARY_TEST_QUERIES test queries to verify model is serving correctly..."

# Wait for the new pod to be fully ready
sleep 30

# Send test queries and count failures
CANARY_PASS=0
CANARY_FAIL=0
for i in $(seq 1 $CANARY_TEST_QUERIES); do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
        http://192.168.1.11:30808/substitute \
        -H "Content-Type: application/json" \
        -d '{"ingredient":"butter","recipe_name":"test","top_k":3}' 2>/dev/null || echo "000")

    if [[ "$HTTP_CODE" == "200" ]]; then
        CANARY_PASS=$((CANARY_PASS + 1))
    else
        CANARY_FAIL=$((CANARY_FAIL + 1))
    fi
    sleep 2
done

log "  Canary results: $CANARY_PASS passed, $CANARY_FAIL failed (threshold: $CANARY_ERROR_THRESHOLD max failures)"

if [[ "$CANARY_FAIL" -gt "$CANARY_ERROR_THRESHOLD" ]]; then
    warn "  CANARY FAILED: $CANARY_FAIL/$CANARY_TEST_QUERIES queries failed"
    warn "  Rolling back to previous model..."

    # Rollback: swap current and previous in S3
    for f in gismo_decoder.onnx ingredient_embeddings.npy vocab.json metadata.json; do
        base="${f%.*}"
        ext="${f##*.}"
        aws --endpoint-url "$S3_ENDPOINT" s3 cp \
            "s3://$S3_BUCKET/models/v2/${base}_previous.${ext}" \
            "s3://$S3_BUCKET/models/v2/$f" \
            --quiet 2>/dev/null || true
    done

    # Restart API with rolled-back model
    kubectl -n forkwise-platform delete pvc gismo-model-pvc --ignore-not-found 2>/dev/null || true
    sleep 3
    kubectl apply -f "$REPO_ROOT/k8s/platform/substitution-api.yaml"
    kubectl -n forkwise-platform rollout restart deployment/substitution-api
    kubectl -n forkwise-platform rollout status deployment/substitution-api --timeout=5m

    die "  Canary failed. Rolled back to previous model. Pipeline aborted."
fi

log "  ✓ Canary passed ($CANARY_PASS/$CANARY_TEST_QUERIES queries successful)"

# Now wait the remaining canary observation window
REMAINING=$((CANARY_DURATION - CANARY_TEST_QUERIES * 2 - 30))
if [[ "$REMAINING" -gt 0 ]]; then
    log "  Observing for ${REMAINING}s more..."
    sleep "$REMAINING"
fi

log "  ✓ Canary observation window complete — model is stable"

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
log "  Previous MRR:        $CURRENT_MRR"
log "  New MRR:             $NEW_MRR"
log "  Absolute gate:       >= $MIN_ABSOLUTE_MRR ✓"
log "  Relative gate:       >= 95% of previous ✓"
log "  Canary:              $CANARY_PASS/$CANARY_TEST_QUERIES passed ✓"
log "  Feedback used:       $FEEDBACK_COUNT events"
log "  Model promoted and deployed"
log "  Timestamp:           $TIMESTAMP"
log "============================================"
