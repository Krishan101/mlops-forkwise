# GISMo Retraining Loop — Guide

## Overview

The retraining loop improves ForkWise's ingredient substitution quality over time by learning from user feedback (Accept/Reject clicks in Mealie) combined with the Recipe1MSubs academic dataset and FlavorGraph ingredient relationships.

The model is **GISMo** (Graph-based Ingredient Substitution Module), a GNN architecture from the paper *"Learning to Substitute Ingredients in Recipes"* (Fatemi et al., 2023). It uses Graph Isomorphism Network (GIN) layers over FlavorGraph to learn context-aware ingredient embeddings, then scores substitution candidates with a learned MLP decoder.

## Architecture

```
FlavorGraph (6,653 ingredients + flavor molecules)
    ↓ GIN layers (message passing)
Ingredient Embeddings (300-dim per ingredient)
    ↓
Context Encoder (average recipe ingredient embeddings)
    ↓
Substitution Decoder (MLP: source || candidate || context → score)
    ↓
Contrastive Loss (maximize true substitution score, minimize random negatives)
```

## Data Sources

| Source | Location | Purpose |
|--------|----------|---------|
| FlavorGraph | `s3://data-proj01/data/raw/flavorgraph/` | Graph structure: ingredient co-occurrences + flavor molecules |
| Recipe1MSubs | `s3://data-proj01/data/raw/recipe1msubs/` | 49K/10K/10K train/val/test substitution pairs |
| Recipe1M | `s3://data-proj01/data/raw/recipe1m/layer1.json` | Recipe context (ingredient lists per recipe) |
| User Feedback | PostgreSQL `feedback_events` table | Accept/Reject signals from Mealie UI |

## Files

```
services/retraining-loop/
├── gismo_model.py         # GISMo model: GIN + Context Encoder + MLP Decoder
├── data_loader.py         # Loads FlavorGraph, Recipe1MSubs, feedback from S3/Postgres
├── train.py               # Standalone training script (logs to MLflow)
├── feedback_train.py      # FastAPI server for triggered retraining
├── config.yaml            # Training hyperparameters
├── requirements.txt       # Python dependencies
├── Dockerfile             # Feedback trainer API image
└── Dockerfile.train       # Standalone training image
```

## Metrics Tracked in MLflow

### Training Metrics
- `train_loss` — contrastive loss per epoch
- `val_mrr` — Mean Reciprocal Rank on validation set
- `val_hit_at_1`, `val_hit_at_3`, `val_hit_at_10` — Hit@K metrics
- `test_mrr`, `test_hit_at_*` — Final test metrics
- `best_val_mrr` — Best validation MRR across epochs
- `best_epoch` — Epoch that achieved best MRR

### Inference Metrics
- `avg_inference_ms` — Average time to score all candidates for one query
- `p50_inference_ms`, `p95_inference_ms`, `p99_inference_ms` — Latency percentiles

### Training Metadata
- `total_train_time_sec` / `total_train_time_min`
- `train_samples`, `feedback_pairs`, `feedback_accepted`
- All hyperparameters (embed_dim, lr, epochs, etc.)

### Artifacts Saved to MLflow
- `model/gismo_best.pt` — Best model weights
- `model/gismo_vocab.json` — Ingredient vocabulary
- `embeddings/gismo_embeddings.json` — All ingredient embeddings
- `config/config.yaml` — Training config used

## Deployment Steps

### Step 1: Build and Push the Image

On node1:
```bash
cd ~/mlops-forkwise/services/retraining-loop
sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1
sudo sysctl -w net.ipv6.conf.lo.disable_ipv6=0
docker build --no-cache -t ghcr.io/krishan101/forkwise-feedback-trainer:0.1.0 .
docker push ghcr.io/krishan101/forkwise-feedback-trainer:0.1.0
```

Make the package public on GitHub (Packages → forkwise-feedback-trainer → Package settings → Public).

### Step 2: Deploy

```bash
kubectl apply -f ~/mlops-forkwise/k8s/platform/feedback-trainer.yaml
kubectl -n forkwise-platform rollout status deployment/feedback-trainer --timeout=3m
```

### Step 3: Verify

```bash
# Check health
curl -s http://feedback-trainer.forkwise-platform:8001/health | python3 -m json.tool

# Check training status
curl -s http://feedback-trainer.forkwise-platform:8001/training/status | python3 -m json.tool
```

## How to Trigger Retraining

### Manual Trigger (from node1)
```bash
curl -s -X POST http://feedback-trainer.forkwise-platform:8001/train \
  -H "Content-Type: application/json" \
  -d '{"epochs": 20, "lr": 0.00005}' | python3 -m json.tool
```

### With Custom Parameters
```bash
curl -s -X POST http://feedback-trainer.forkwise-platform:8001/train \
  -H "Content-Type: application/json" \
  -d '{
    "min_samples": 5,
    "epochs": 30,
    "lr": 0.0001,
    "run_name": "feedback-retrain-v2"
  }' | python3 -m json.tool
```

### Monitor Progress
```bash
# Poll status
watch -n 10 'curl -s http://feedback-trainer.forkwise-platform:8001/training/status | python3 -m json.tool'

# Check logs
kubectl -n forkwise-platform logs deploy/feedback-trainer -f
```

## How It Connects to Existing Services

### Data Flow for Retraining

```
1. User clicks "Suggest Substitute" in Mealie
   → Mealie calls substitution-api /predict
   → Results shown to user with Accept/Reject buttons

2. User clicks Accept or Reject
   → Mealie calls substitution-api /feedback
   → Stored in PostgreSQL: feedback_events + substitution_results.accepted

3. Feedback accumulates in PostgreSQL

4. Retraining triggered (manual POST /train or scheduled)
   → feedback_train.py loads:
     - FlavorGraph from S3
     - Recipe1MSubs from S3
     - Feedback pairs from PostgreSQL
   → Trains GISMo model
   → Logs metrics + artifacts to MLflow
   → Exports new embeddings to Qdrant (replaces old collection)
   → Stamps trained rows in PostgreSQL (trained_at = NOW())

5. Substitution API now uses updated embeddings from Qdrant
   → Better substitution suggestions
```

### Endpoint Wiring

| From | To | Purpose |
|------|----|---------|
| Mealie | `substitution-api:8080/predict` | Get substitution suggestions |
| Mealie | `substitution-api:8080/feedback` | Log accept/reject |
| feedback-trainer | `platform-db:5432` | Read feedback, stamp trained rows |
| feedback-trainer | S3 (`data-proj01`) | Read FlavorGraph + Recipe1MSubs |
| feedback-trainer | `mlflow:5000` | Log experiments, save artifacts |
| feedback-trainer | `qdrant:6333` | Update ingredient embeddings |

## Initial Training (First Time)

Before the feedback loop can work, you need to run one initial training:

### Option A: Run Standalone Training as a K8s Job

```bash
# Build the training image
cd ~/mlops-forkwise/services/retraining-loop
docker build -f Dockerfile.train -t ghcr.io/krishan101/forkwise-gismo-train:0.1.0 .
docker push ghcr.io/krishan101/forkwise-gismo-train:0.1.0
```

Then create a K8s Job:
```bash
kubectl -n forkwise-platform apply -f - <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: gismo-initial-train
  namespace: forkwise-platform
spec:
  backoffLimit: 1
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: trainer
        image: ghcr.io/krishan101/forkwise-gismo-train:0.1.0
        args: ["--config", "config.yaml"]
        env:
        - { name: MLFLOW_TRACKING_URI, value: "http://mlflow:5000" }
        - { name: AWS_ACCESS_KEY_ID, value: "8921c48faf83433db2b1439a9b2889fd" }
        - { name: AWS_SECRET_ACCESS_KEY, value: "7d1ce78efc5a48019888c9f3fa8ba2dd" }
        - { name: S3_ENDPOINT, value: "https://chi.tacc.chameleoncloud.org:7480" }
        - { name: S3_BUCKET, value: "data-proj01" }
        - { name: MLFLOW_S3_ENDPOINT_URL, value: "https://chi.tacc.chameleoncloud.org:7480" }
        resources:
          requests: { cpu: "1", memory: 4Gi }
          limits:   { cpu: "2", memory: 8Gi }
EOF
```

Monitor:
```bash
kubectl -n forkwise-platform logs job/gismo-initial-train -f
```

### Option B: Trigger via Feedback Trainer API

Set `min_samples` to 0 to force training even without feedback:
```bash
curl -s -X POST http://feedback-trainer.forkwise-platform:8001/train \
  -H "Content-Type: application/json" \
  -d '{"min_samples": 0, "epochs": 50, "run_name": "initial-train-v1"}'
```

## After Training Completes

1. **Check MLflow** at `http://<FLOATING_IP>:30500` — you should see the training run with all metrics
2. **Verify Qdrant** was updated — the `ingredient_embeddings` collection should have 6,653 vectors (one per FlavorGraph ingredient) instead of the previous 33 (from Mealie recipes)
3. **Test substitution** — the results should now be much better since they're based on the trained GISMo model instead of generic sentence-transformer similarity:

```bash
curl -s -X POST http://substitution-api.forkwise-platform:8080/substitute \
  -H "Content-Type: application/json" \
  -d '{"ingredient":"butter","recipe_name":"Classic Pancakes","top_k":5}' | python3 -m json.tool
```

The substitution API automatically uses whatever embeddings are in Qdrant — no restart needed after retraining.

## Scheduling Retraining

For automated retraining on a schedule, create a CronJob:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: gismo-retrain
  namespace: forkwise-platform
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: Never
          containers:
          - name: trigger
            image: curlimages/curl:latest
            command: ["curl", "-sS", "-X", "POST",
              "http://feedback-trainer:8001/train",
              "-H", "Content-Type: application/json",
              "-d", '{"min_samples": 10, "epochs": 20}']
```

## Comparison: Before vs After GISMo

| Aspect | Before (SBERT base) | After (GISMo trained) |
|--------|---------------------|----------------------|
| Model | all-MiniLM-L6-v2 (generic) | GISMo GNN (specialized) |
| Embeddings | 384-dim text similarity | 300-dim graph-learned ingredient relations |
| Context-aware | Partial (recipe name in query) | Yes (recipe ingredient context encoder) |
| Data | Generic English NLP | FlavorGraph + Recipe1MSubs + user feedback |
| Results | "melted butter" for "butter" | "margarine", "coconut oil", "applesauce" |
