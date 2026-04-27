#!/usr/bin/env bash
# Upload trained GISMo model artifacts to Chameleon object storage
# Run from the GPU training instance after training completes

set -euo pipefail

OUTPUT_DIR="${1:-/home/cc/training/output}"
S3_ENDPOINT="https://chi.tacc.chameleoncloud.org:7480"
S3_BUCKET="data-proj01"
S3_PREFIX="models/v2"

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-8921c48faf83433db2b1439a9b2889fd}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-7d1ce78efc5a48019888c9f3fa8ba2dd}"
export PATH="$HOME/.local/bin:$PATH"

echo "=== Uploading GISMo model artifacts ==="
echo "Source: $OUTPUT_DIR"
echo "Dest:   s3://$S3_BUCKET/$S3_PREFIX/"

for f in gismo_decoder.onnx ingredient_embeddings.npy vocab.json metadata.json gismo_best.pth; do
    if [ -f "$OUTPUT_DIR/$f" ]; then
        echo "  Uploading $f..."
        aws --endpoint-url "$S3_ENDPOINT" s3 cp "$OUTPUT_DIR/$f" "s3://$S3_BUCKET/$S3_PREFIX/$f"
    else
        echo "  SKIP $f (not found)"
    fi
done

echo ""
echo "=== Verifying upload ==="
aws --endpoint-url "$S3_ENDPOINT" s3 ls "s3://$S3_BUCKET/$S3_PREFIX/"

echo ""
echo "Done! Model artifacts available at s3://$S3_BUCKET/$S3_PREFIX/"
