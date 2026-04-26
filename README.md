# ForkWise MLOps Infrastructure

ML-powered ingredient substitution system built on top of [Mealie](https://github.com/HivanshD/mealie), a self-hosted recipe manager. Users browse recipes in Mealie and get smart substitution suggestions for any ingredient — powered by **GISMo** (Graph-based Ingredient Substitution Module), a GNN model trained on FlavorGraph and Recipe1MSubs, with continuous improvement from user feedback.

Based on the paper: *"Learning to Substitute Ingredients in Recipes"* (Fatemi et al., 2023, arXiv:2302.07960).

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │         Mealie (port 30900)         │
                        │    Recipe app with substitution UI  │
                        └──────┬──────────────┬───────────────┘
                               │              │
                    polls every 30s    "Suggest Substitute"
                               │              │
                               ▼              ▼
                        ┌────────────┐  ┌─────────────────────┐
                        │ Ingest API │  │  Substitution API   │
                        └─────┬──────┘  │  /predict /feedback │
                              │         └────┬───────────┬────┘
                              ▼              │           │
                     ┌─────────────┐    Qdrant search   logs to
                     │ Platform DB │         │        PostgreSQL
                     │ (Postgres)  │◄────────┘           │
                     └─────┬───────┘                     │
                           │                             ▼
                    feature jobs              ┌──────────────────┐
                           │                  │ Feedback Trainer  │
                           ▼                  │ POST /train       │
                    ┌──────────────┐          │ GISMo retraining  │
                    │Feature Worker│          └────────┬──────────┘
                    │  SBERT embed │                   │
                    └──────┬───────┘          trains GISMo (GNN)
                           │                  logs to MLflow
                           ▼                  updates Qdrant
                    ┌──────────────┐                   │
                    │    Qdrant    │◄──────────────────┘
                    │ Vector DB   │
                    └─────────────┘

MLflow (port 30500) — experiment tracking
Grafana (port 30300) — cluster monitoring (Prometheus + kube-prometheus-stack)
```

## Infrastructure

- **Cloud:** Chameleon Cloud, KVM@TACC site
- **VMs:** 3x `m1.xlarge` on Ubuntu 24.04 (1 control plane + 2 workers)
- **Networking:** dual-NIC — `sharednet1` (public) + private `192.168.1.0/24`
- **K8s:** kubespray (release-2.26), single control plane on node1
- **Storage:** local-path-provisioner for PVCs
- **Object Store:** `s3://data-proj01` on Chameleon (backups + ML training data)
- **Monitoring:** Prometheus + Grafana via kube-prometheus-stack Helm chart

## ML Model: GISMo

GISMo uses Graph Isomorphism Network (GIN) layers over FlavorGraph (6,653 ingredient nodes + flavor molecule edges) to learn context-aware ingredient embeddings. Given a recipe context and a source ingredient, it scores all candidate ingredients as potential substitutions using a learned MLP decoder trained with contrastive loss.

**Training data:**
- FlavorGraph — ingredient co-occurrence graph from Recipe1M
- Recipe1MSubs — 49K/10K/10K train/val/test substitution pairs extracted from user comments
- User feedback — Accept/Reject signals from Mealie UI (continuous improvement)

**Metrics tracked in MLflow:**
- MRR (Mean Reciprocal Rank), Hit@1, Hit@3, Hit@10
- Training loss per epoch, inference latency (avg/p50/p95/p99)
- Model weights, vocabulary, embeddings as artifacts

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
- Buckets: `data-proj01` (training data + backups), `models-proj01` (model checkpoints)

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
| Substitution API | `ghcr.io/krishan101/forkwise-substitution-api:0.1.3` | `services/substitution-api/` |
| MLflow | `ghcr.io/krishan101/forkwise-mlflow:0.1.0` | `services/mlflow/` |
| Feedback Trainer | `ghcr.io/krishan101/forkwise-feedback-trainer:0.1.0` | `services/retraining-loop/` |
| GISMo Train (Job) | `ghcr.io/krishan101/forkwise-gismo-train:0.1.0` | `services/retraining-loop/Dockerfile.train` |

## Repository Structure

```
mlops-forkwise/
├── provision/
│   └── provision.ipynb              # Jupyter notebook — run on Chameleon JupyterHub
├── tf/kvm/                          # Terraform configs for 3 VMs + network + floating IP
├── db/
│   └── init.sql                     # Platform postgres schema
├── k8s/
│   ├── mealie/                      # Mealie app + its postgres
│   ├── platform/                    # Platform services (all in forkwise-platform namespace)
│   │   ├── postgres.yaml            # Platform DB (recipe metadata, feedback, jobs, mlflow DB)
│   │   ├── qdrant.yaml              # Qdrant vector DB
│   │   ├── ingest-api.yaml          # Polls Mealie → postgres + feature jobs
│   │   ├── feature-worker.yaml      # Computes SBERT embeddings → Qdrant
│   │   ├── substitution-api.yaml    # /predict + /feedback + /substitute endpoints
│   │   ├── mlflow.yaml              # MLflow tracking server
│   │   └── feedback-trainer.yaml    # GISMo retraining API
│   └── monitoring/                  # Prometheus + Grafana namespace
├── services/
│   ├── ingest-api/                  # Polls Mealie for recipes
│   ├── feature-worker/              # SBERT embeddings → Qdrant
│   ├── substitution-api/            # Searches Qdrant for substitutes
│   ├── mlflow/                      # MLflow with psycopg2 + boto3
│   └── retraining-loop/             # GISMo model + training + feedback API
│       ├── gismo_model.py           # GISMo: GIN + Context Encoder + MLP Decoder
│       ├── data_loader.py           # Loads FlavorGraph, Recipe1MSubs, feedback
│       ├── train.py                 # Standalone training (MLflow logging)
│       ├── feedback_train.py        # FastAPI retraining server
│       ├── config.yaml              # Training hyperparameters
│       ├── GUIDE.md                 # Detailed retraining guide
│       ├── Dockerfile               # Feedback trainer image
│       └── Dockerfile.train         # Standalone training image
├── scripts/
│   ├── bring_up.sh                  # Deploys everything + S3 restore
│   ├── backup.sh                    # Snapshots all state to S3
│   ├── teardown.sh                  # Backs up then deletes K8s resources
│   ├── setup_monitoring.sh          # Installs kube-prometheus-stack
│   └── _s3_common.sh               # Shared S3 helpers
└── clouds.yaml.example             # Template for OpenStack credentials
```

## Full Setup From Scratch

### Step 1: Provision VMs (Chameleon JupyterHub)

1. Go to [Chameleon JupyterHub](https://jupyter.chameleoncloud.org/)
2. Clone the repo: `cd /work && git clone https://github.com/Krishan101/mlops-forkwise.git`
3. Place `clouds.yaml` at `/work/clouds.yaml` with your KVM@TACC application credentials
4. Open `provision/provision.ipynb` and run all cells top to bottom

**Note:** If Terraform fails with "More than one Security Group found", hardcode SG IDs in `data.tf` and `main.tf` (see Known Issues).

### Step 2: Copy SSH Key and clouds.yaml to node1 (from Windows PowerShell)

```powershell
scp -i "C:\Users\Krishan Guta\.ssh\forkwise_key" "C:\Users\Krishan Guta\.ssh\forkwise_key" cc@<FLOATING_IP>:~/.ssh/forkwise-key
ssh -i "C:\Users\Krishan Guta\.ssh\forkwise_key" cc@<FLOATING_IP> "mkdir -p ~/.config/openstack"
scp -i "C:\Users\Krishan Guta\.ssh\forkwise_key" "C:\Users\Krishan Guta\path\to\clouds.yaml" cc@<FLOATING_IP>:~/.config/openstack/clouds.yaml
```

### Step 3: SSH into node1

```powershell
ssh -i "C:\Users\Krishan Guta\.ssh\forkwise_key" cc@<FLOATING_IP>
```

### Step 4: Set up SSH keys on node1

```bash
chmod 600 ~/.ssh/*
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -q -N ""
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
cat ~/.ssh/id_rsa.pub | ssh -i ~/.ssh/forkwise-key -o StrictHostKeyChecking=no cc@192.168.1.12 "cat >> ~/.ssh/authorized_keys"
cat ~/.ssh/id_rsa.pub | ssh -i ~/.ssh/forkwise-key -o StrictHostKeyChecking=no cc@192.168.1.13 "cat >> ~/.ssh/authorized_keys"
```

### Step 5: Disable IPv6 on all nodes

```bash
for ip in 192.168.1.11 192.168.1.12 192.168.1.13; do ssh -o StrictHostKeyChecking=no cc@$ip 'sudo sysctl -w net.ipv6.conf.ens3.disable_ipv6=1'; done
```

### Step 6: Install kubespray and deploy K8s

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git tmux
git clone -b release-2.26 https://github.com/kubernetes-sigs/kubespray.git
cd kubespray
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt ruamel.yaml
```

Build inventory:

```bash
cp -rfp inventory/sample inventory/mycluster
CONFIG_FILE=inventory/mycluster/hosts.yaml python3 contrib/inventory_builder/inventory.py 192.168.1.11 192.168.1.12 192.168.1.13
python3 -c "import yaml; p='inventory/mycluster/hosts.yaml'; d=yaml.safe_load(open(p)); d['all']['children']['kube_control_plane']['hosts']={'node1': None}; open(p,'w').write(yaml.safe_dump(d, default_flow_style=False))"
```

Configure ansible:

```bash
cat > inventory/mycluster/group_vars/all/all.yml << 'EOF'
ansible_user: cc
ansible_ssh_private_key_file: /home/cc/.ssh/forkwise-key
ansible_become: true
ansible_become_method: sudo
disable_ipv6_dns: true
EOF

sed -i 's/^helm_enabled: false/helm_enabled: true/' inventory/mycluster/group_vars/k8s_cluster/addons.yml
```

Test and run (in tmux):

```bash
ansible -i inventory/mycluster/hosts.yaml all -m ping
tmux new -s kubespray
cd ~/kubespray && source .venv/bin/activate
ansible-playbook -i inventory/mycluster/hosts.yaml cluster.yml -b 2>&1 | tee /tmp/kubespray.log
# Detach: Ctrl+B then D   Reattach: tmux attach -t kubespray
```

### Step 7: Post-kubespray setup

```bash
mkdir -p ~/.kube && sudo cp /etc/kubernetes/admin.conf ~/.kube/config && sudo chown $(id -u):$(id -g) ~/.kube/config
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/master/deploy/local-path-storage.yaml
kubectl patch storageclass local-path -p '{"metadata":{"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl -n kube-system patch deployment metrics-server --type='json' -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
kubectl -n kube-system get configmap coredns -o yaml | sed 's|forward . /etc/resolv.conf|forward . 8.8.8.8 1.1.1.1|' | kubectl apply -f -
kubectl -n kube-system rollout restart deployment coredns
```

### Step 8: Clone repo and deploy everything

```bash
cd ~
git clone https://github.com/Krishan101/mlops-forkwise.git
cd mlops-forkwise
bash scripts/bring_up.sh
```

This single command deploys all services, restores from S3 backups if they exist, fixes known bugs, and sets up monitoring.

### Step 9: Upload Training Data to S3

Before running the GISMo model, upload FlavorGraph and Recipe1MSubs to S3:

```bash
pip install awscli --break-system-packages
export AWS_ACCESS_KEY_ID=8921c48faf83433db2b1439a9b2889fd
export AWS_SECRET_ACCESS_KEY=7d1ce78efc5a48019888c9f3fa8ba2dd

# Download FlavorGraph
git clone https://github.com/lamypark/FlavorGraph.git /tmp/flavorgraph
aws --endpoint-url https://chi.tacc.chameleoncloud.org:7480 s3 cp /tmp/flavorgraph/input/nodes_191120.csv s3://data-proj01/data/raw/flavorgraph/nodes_191120.csv
aws --endpoint-url https://chi.tacc.chameleoncloud.org:7480 s3 cp /tmp/flavorgraph/input/edges_191120.csv s3://data-proj01/data/raw/flavorgraph/edges_191120.csv

# Download Recipe1MSubs (from GISMo repo)
git clone https://github.com/facebookresearch/gismo.git /tmp/gismo
aws --endpoint-url https://chi.tacc.chameleoncloud.org:7480 s3 cp /tmp/gismo/data/recipe1msubs/train.json s3://data-proj01/data/raw/recipe1msubs/train.json
aws --endpoint-url https://chi.tacc.chameleoncloud.org:7480 s3 cp /tmp/gismo/data/recipe1msubs/val.json s3://data-proj01/data/raw/recipe1msubs/val.json
aws --endpoint-url https://chi.tacc.chameleoncloud.org:7480 s3 cp /tmp/gismo/data/recipe1msubs/test.json s3://data-proj01/data/raw/recipe1msubs/test.json
```

### Step 10: Build and Deploy Feedback Trainer

```bash
cd ~/mlops-forkwise/services/retraining-loop
sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1
sudo sysctl -w net.ipv6.conf.lo.disable_ipv6=0
docker system prune -a -f
docker build --no-cache -t ghcr.io/krishan101/forkwise-feedback-trainer:0.1.0 .
docker push ghcr.io/krishan101/forkwise-feedback-trainer:0.1.0
kubectl apply -f ~/mlops-forkwise/k8s/platform/feedback-trainer.yaml
```

### Step 11: Run Initial GISMo Training

```bash
curl -s -X POST http://feedback-trainer.forkwise-platform:8001/train \
  -H "Content-Type: application/json" \
  -d '{"min_samples": 0, "epochs": 50, "run_name": "initial-gismo-v1"}' | python3 -m json.tool
```

Monitor progress:

```bash
kubectl -n forkwise-platform logs deploy/feedback-trainer -f
```

### Step 12: Verify

| Service | URL | Credentials |
|---------|-----|-------------|
| Mealie | `http://<FLOATING_IP>:30900` | Create account on first visit |
| Substitution API | `http://<FLOATING_IP>:30808/health` | — |
| MLflow | `http://<FLOATING_IP>:30500` | — |
| Grafana | `http://<FLOATING_IP>:30300` | admin / forkwise-admin |
| Feedback Trainer | `http://feedback-trainer.forkwise-platform:8001/health` | cluster-internal only |

Test substitution:

```bash
curl -s -X POST http://substitution-api.forkwise-platform:8080/substitute \
  -H "Content-Type: application/json" \
  -d '{"ingredient":"butter","recipe_name":"Classic Pancakes","top_k":5}' | python3 -m json.tool
```

## Building Docker Images

If you need to rebuild any service image (after code changes), run on node1:

```bash
# Fix IPv6 first
sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1
sudo sysctl -w net.ipv6.conf.lo.disable_ipv6=0

# Clean disk space if needed (node1 has only 37GB)
docker system prune -a -f

# Login to GHCR
echo "<GHCR_TOKEN>" | docker login ghcr.io -u Krishan101 --password-stdin

# Build and push (example: substitution-api)
cd ~/mlops-forkwise/services/substitution-api
docker build --no-cache -t ghcr.io/krishan101/forkwise-substitution-api:0.1.4 .
docker push ghcr.io/krishan101/forkwise-substitution-api:0.1.4

# Make package public: github.com → Packages → Package settings → Public

# Update the running deployment
kubectl -n forkwise-platform set image deployment/substitution-api substitution-api=ghcr.io/krishan101/forkwise-substitution-api:0.1.4
```

## Backup & Restore

**Backup** (run before teardown or anytime):

```bash
bash scripts/backup.sh
```

Snapshots to `s3://data-proj01/backups/`: mealie-db, platform-db, qdrant, mealie-data.

**Restore** happens automatically during `bring_up.sh` — if backups exist in S3, they are restored after each service deploys.

## Teardown

```bash
bash scripts/teardown.sh
```

Backs up everything to S3, then deletes all K8s namespaces. After teardown, destroy VMs from the provision notebook's teardown cells.

## Data Pipeline Flow

1. **User adds recipe in Mealie** (via UI or API)
2. **Ingest service** polls Mealie every 30s, stores metadata + ingredients in platform postgres, queues a feature job
3. **Feature worker** computes a 384-dim SBERT embedding for each ingredient, upserts into Qdrant
4. **User clicks "Suggest Substitute"** → Mealie → Substitution API → Qdrant vector search → ranked results
5. **User clicks Accept/Reject** → feedback logged to postgres
6. **Feedback trainer** (triggered manually or on schedule) retrains GISMo on feedback + Recipe1MSubs, updates Qdrant with improved embeddings

## Retraining Loop

See [services/retraining-loop/GUIDE.md](services/retraining-loop/GUIDE.md) for detailed instructions on the GISMo retraining loop, including how to trigger retraining, monitor progress, and schedule automated retraining.

## S3 Data Layout

```
s3://data-proj01/
├── backups/                         # Automated backups from backup.sh
│   ├── mealie-db/latest.sql.gz
│   ├── platform-db/latest.sql.gz
│   ├── qdrant/latest.snapshot
│   └── mealie-data/latest.tar.gz
├── data/raw/                        # Training data
│   ├── flavorgraph/
│   │   ├── nodes_191120.csv
│   │   └── edges_191120.csv
│   ├── recipe1msubs/
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   └── recipe1m/
│       └── layer1.json (optional)
└── mlflow-artifacts/                # MLflow experiment artifacts
```

## Known Issues

- **Mealie `reference_id` bug:** The Mealie fork generates a new UUID for each ingredient on every recipe load. `bring_up.sh` works around this with a SQL UPDATE.
- **IPv6 on Chameleon:** IPv6 doesn't route on Chameleon KVM@TACC. Must disable for Docker builds and pip installs. Keep loopback enabled (`net.ipv6.conf.lo.disable_ipv6=0`) or kubectl breaks.
- **Disk space:** node1 has 37GB. Run `docker system prune -a -f` before building images.
- **Security groups:** Chameleon's shared project has duplicate SG names. Terraform `data` lookups fail. Workaround: hardcode SG IDs in `data.tf` and `main.tf`.
- **SG attachment:** Creating SGs is not enough — must also attach them to node1's sharednet1 port. `bring_up.sh` handles this automatically.
