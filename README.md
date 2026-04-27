# ForkWise MLOps Infrastructure

ML-powered ingredient substitution system built on top of [Mealie](https://github.com/HivanshD/mealie), a self-hosted recipe manager. Users browse recipes in Mealie and get smart substitution suggestions for any ingredient — powered by GISMo (Graph-based Ingredient Substitution Module) with ONNX serving and sentence-transformer embeddings.

## Architecture

```
User → Mealie (recipe app, port 30900)
         ↓ (polls every 30s)
       Ingest Service → Platform Postgres + Feature Job Queue
                           ↓
                        Feature Worker → Sentence-Transformer Embeddings → Qdrant
                           ↓
User clicks "Suggest Substitute" → Mealie → Substitution API
                                               ↓
                                   Stage 1: Qdrant retrieval (top-50 by cosine sim)
                                   Stage 2: GISMo ONNX reranking (graph-based scoring)
                                               ↓
                                   Ranked suggestions returned to user
                                               ↓
                     Accept/Reject feedback → Postgres → Retraining Pipeline

Retraining CronJob (every 6h):
  Feedback threshold met? → Export → Merge with base data → Train on GPU
    → Quality gates (MRR ≥ 10.0 AND ≥ 95% of current) → Promote to S3 → Restart API

MLflow (port 30500) — experiment tracking for all training runs
Grafana (port 30300) — serving metrics, latency, errors, feedback, pod resources
```

## GISMo Model

The substitution model follows the [GISMo paper](https://arxiv.org/abs/2302.07960) (Fatemi et al., 2023):

- **Encoder:** 2-layer Graph Isomorphism Network (GIN) over FlavorGraph (6,653 ingredient + 1,645 chemical compound nodes, 147K edges)
- **Decoder:** 3-layer MLP scoring (source, candidate, context) triples
- **Training data:** Recipe1MSubs (49,044 train / 10,729 val / 10,747 test substitution pairs)
- **Serving:** ONNX export of decoder + precomputed GIN embeddings (no GPU needed at inference)
- **Performance:** Val MRR 74.08, Test MRR 24.26, H@1=14.16, H@3=26.93, H@10=45.93

## Infrastructure

- **Cloud:** Chameleon Cloud, KVM@TACC site
- **VMs:** 3x `m1.xlarge` on Ubuntu 24.04 (1 control plane + 2 workers)
- **GPU:** NVIDIA H100 on KVM@TACC (for training/retraining, not needed for serving)
- **Networking:** dual-NIC — `sharednet1` (public) + private `192.168.1.0/24`
- **K8s:** kubespray (release-2.26), single control plane on node1
- **Storage:** local-path-provisioner for PVCs
- **Object Store:** `s3://data-proj01` on Chameleon (model artifacts, training data, backups)

## Credentials

**Chameleon OpenStack (KVM@TACC):**
- Auth URL: `https://kvm.tacc.chameleoncloud.org:5000`
- SSH key on Chameleon: `forkwise-key`
- SSH key on local machine: `C:\Users\Krishan Guta\.ssh\forkwise_key`
- Project prefix: `proj01`

**Object Store (CHI@TACC):**
- Endpoint: `https://chi.tacc.chameleoncloud.org:7480`
- Access Key: `8921c48faf83433db2b1439a9b2889fd`
- Secret Key: `7d1ce78efc5a48019888c9f3fa8ba2dd`
- Bucket: `data-proj01` (training data, model artifacts, backups)

**Mealie:**
- Email: `krishankumargupta101@gmail.com`
- Password: `mynameiskrishan`

**Grafana:**
- User: `admin`
- Password: `forkwise-admin`

**GHCR (GitHub Container Registry):**
- User: `Krishan101`
- All images are public under `ghcr.io/krishan101/`

## Docker Images

| Service | Image | Built From |
|---------|-------|------------|
| Mealie (fork) | `ghcr.io/krishan101/forkwise-mealie:0.1.0` | `github.com/HivanshD/mealie` |
| Ingest API | `ghcr.io/krishan101/forkwise-ingest:0.1.0` | `services/ingest-api/` |
| Feature Worker | `ghcr.io/krishan101/forkwise-feature-worker:0.1.3` | `services/feature-worker/` |
| Substitution API | `ghcr.io/krishan101/forkwise-substitution-api:0.2.2` | `services/substitution-api/` |
| MLflow | `ghcr.io/krishan101/forkwise-mlflow:0.1.0` | `services/mlflow/` |
| GISMo Training | Built on GPU instance | `services/training/` |

## Model Artifacts in Object Store

```
s3://data-proj01/models/v2/
├── gismo_decoder.onnx              (2.9 MB — ONNX decoder for scoring)
├── ingredient_embeddings.npy       (10 MB — precomputed GIN embeddings)
├── vocab.json                      (170 KB — ingredient name → index)
├── metadata.json                   (model metrics + training info)
├── gismo_best.pth                  (43 MB — PyTorch checkpoint)
├── *_previous.*                    (rollback copies of each file)
```

## Repository Structure

```
mlops-forkwise/
├── provision/
│   └── provision.ipynb              # Jupyter notebook — run on Chameleon JupyterHub
├── tf/kvm/                          # Terraform configs for 3 VMs + network + floating IP
├── db/
│   └── init.sql                     # Platform postgres schema
├── docs/
│   └── SAFEGUARDING.md              # Safeguarding plan (fairness, privacy, etc.)
├── k8s/
│   ├── mealie/                      # Mealie app + its postgres
│   ├── platform/
│   │   ├── substitution-api.yaml    # API with GISMo init container + PVC
│   │   ├── retrain-cronjob.yaml     # CronJob + RBAC for automated retraining
│   │   └── ...                      # postgres, qdrant, mlflow, ingest, feature-worker
│   └── monitoring/
│       ├── servicemonitor.yaml      # Prometheus scraping for substitution API
│       └── grafana-dashboard.json   # Dashboard with alerts
├── services/
│   ├── ingest-api/                  # Polls Mealie for recipes, stores in platform DB
│   ├── feature-worker/              # Computes sentence-transformer embeddings → Qdrant
│   ├── substitution-api/            # Two-stage GISMo+Qdrant serving + Prometheus metrics
│   ├── training/                    # GISMo training script + Dockerfile + fixup artifacts
│   └── mlflow/                      # MLflow with psycopg2 + boto3
├── scripts/
│   ├── bring_up.sh                  # Full deploy with S3 restore + GISMo resources
│   ├── retrain_pipeline.sh          # End-to-end: feedback → merge → train → quality gates → promote
│   ├── load_generator.sh            # Emulated user traffic for operation period
│   ├── seed_feedback.py             # Generate test feedback data
│   ├── fixup_artifacts.py           # Generate embeddings/vocab/metadata post-training
│   ├── upload_gismo_model.sh        # Upload model to object store
│   ├── backup.sh                    # Snapshot state to S3
│   ├── teardown.sh                  # Backup then delete K8s resources
│   ├── setup_monitoring.sh          # Prometheus + Grafana via Helm
│   └── _s3_common.sh               # Shared S3 helpers
└── clouds.yaml.example              # Template for OpenStack credentials
```

## Full Setup From Scratch

### Step 1: Provision VMs (Chameleon JupyterHub)

1. Go to [Chameleon JupyterHub](https://jupyter.chameleoncloud.org/)
2. Place `clouds.yaml` at `/work/clouds.yaml` with your KVM@TACC application credentials
3. Upload `provision/provision.ipynb` or clone the repo at `/work/`
4. Run all cells top to bottom — creates lease, provisions 3 VMs with floating IP

### Step 2: Copy SSH Key to node1 (from Windows PowerShell)

```powershell
scp -i "C:\Users\Krishan Guta\.ssh\forkwise_key" "C:\Users\Krishan Guta\.ssh\forkwise_key" cc@<FLOATING_IP>:~/.ssh/forkwise-key
```

### Step 3: SSH into node1 and set up

```powershell
ssh -i "C:\Users\Krishan Guta\.ssh\forkwise_key" cc@<FLOATING_IP>
```

Then on node1, follow Steps D through K from the provision notebook output.

### Step 4: Deploy everything

```bash
cd ~
git clone -b gismo-integration https://github.com/Krishan101/mlops-forkwise.git
cd mlops-forkwise
bash scripts/bring_up.sh
```

This deploys all K8s resources, restores from S3 backups if they exist, sets up monitoring, deploys the GISMo ServiceMonitor and retraining CronJob.

### Step 5: Build substitution API image (first time only)

```bash
cd ~/mlops-forkwise/services/substitution-api
sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1
sudo sysctl -w net.ipv6.conf.lo.disable_ipv6=0
docker build --no-cache -t ghcr.io/krishan101/forkwise-substitution-api:0.2.2 .
docker push ghcr.io/krishan101/forkwise-substitution-api:0.2.2
```

### Step 6: Verify

| Service | URL | Credentials |
|---------|-----|-------------|
| Mealie | `http://<FLOATING_IP>:30900` | Create account on first visit |
| Substitution API | `http://<FLOATING_IP>:30808/admin/model-info` | — |
| MLflow | `http://<FLOATING_IP>:30500` | — |
| Grafana | `http://<FLOATING_IP>:30300` | admin / forkwise-admin |

## Retraining Pipeline

The retraining pipeline runs either manually or via the CronJob (every 6 hours):

```
Feedback in Postgres (≥ 50 events)
  → Export accepted feedback
  → Validate: deduplicate + remove contradictory pairs
  → Merge with base Recipe1MSubs training data (49K samples)
  → Upload to GPU instance
  → Train GISMo (30 epochs, early stopping)
  → Generate ONNX + embeddings + vocab
  → STAGING: Quality Gate 1 — test MRR ≥ 10.0 (absolute minimum)
  → STAGING: Quality Gate 2 — test MRR ≥ 95% of current model
  → Backup current model as _previous
  → Upload new model to S3
  → Restart substitution API (init container re-downloads)
  → CANARY: Send 20 test queries, verify ≤ 3 failures
    → FAIL: auto-rollback to previous model
    → PASS: mark feedback consumed, model is live in PRODUCTION
```

Manual trigger:
```bash
bash scripts/retrain_pipeline.sh <GPU_IP>
```

## Monitoring

**Prometheus** scrapes the substitution API's `/metrics` endpoint every 15s via ServiceMonitor.

**Custom drift detection metrics:**
- `forkwise_gismo_top_score` — GISMo score distribution (histogram)
- `forkwise_unknown_ingredient_total` — queries with unknown ingredients
- `forkwise_qdrant_fallback_total` — queries falling back to Qdrant-only
- `forkwise_feedback_accept_total` / `forkwise_feedback_reject_total` — accept/reject ratio

**Grafana dashboard** (imported from `k8s/monitoring/grafana-dashboard.json`) includes:
- Substitution requests per second
- Request latency (p50/p95/p99)
- Error rate (%) with alert: fires if > 10% for 5 minutes
- Total requests and feedback events (last 1h)
- Feedback rate
- Pod resource usage (memory)
- Latency alert: fires if p95 > 1 second for 5 minutes
- Pod restarts (last 24h)
- GISMo model active status

**Autoscaling:** HPA scales the substitution API from 1 to 3 replicas at 70% CPU utilization.

## Rollback

If the deployed model is performing poorly:

**Automated:** The retrain pipeline's quality gates prevent deploying a model with MRR below 10.0 or more than 5% worse than the current model.

**Manual:**
```bash
curl -X POST http://<FLOATING_IP>:30808/admin/rollback
```
This swaps the current and previous model files on disk and reloads the ONNX session.

## Safeguarding

See `docs/SAFEGUARDING.md` for the full safeguarding plan covering:
- **Fairness:** Western cuisine bias acknowledgment, feedback correction loop
- **Explainability:** Score transparency, model info endpoint, MLflow lineage
- **Transparency:** Visible ML labels, model version in responses, open source
- **Privacy:** Self-hosted, no PII in training, minimal data collection
- **Accountability:** Full audit trail, rollback, quality gates, canary validation
- **Robustness:** Graceful fallback, input normalization, probes, HPA, backup/restore
- **Threshold justifications:** All numeric thresholds (MRR gates, alert rules, HPA targets) explicitly justified
- **Multi-environment strategy:** Logical staging/canary/production flow documented
- **Data quality:** Three-stage validation (ingestion, training data construction, production drift detection)

## Known Issues

- **Mealie `reference_id` bug:** `bring_up.sh` fixes this automatically. For new recipes added after deployment: `kubectl -n forkwise-app exec deploy/mealie-db -- psql -U mealie -d mealie -c "UPDATE recipes_ingredients SET reference_id = gen_random_uuid() WHERE reference_id IS NULL"`
- **IPv6 on Chameleon:** `bring_up.sh` handles this. For manual builds: `sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1`
- **Disk space:** Run `docker system prune -a -f` before building if disk is low on node1.
- **GPU dependency:** The retraining CronJob requires an active GPU lease with SSH access. If no GPU is available, the CronJob exits gracefully.
- **ONNX validation crash:** The training script's ONNX validation step may crash due to a cuda/cpu device mismatch. The ONNX file is saved correctly before the crash; `fixup_artifacts.py` handles the remaining artifacts.
