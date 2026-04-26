# ForkWise MLOps Infrastructure

ML-powered ingredient substitution system built on top of [Mealie](https://github.com/HivanshD/mealie), a self-hosted recipe manager. Users browse recipes in Mealie and get smart substitution suggestions for any ingredient — powered by sentence-transformer embeddings and a vector similarity search pipeline.

## Architecture

```
User → Mealie (recipe app, port 30900)
         ↓ (polls every 30s)
       Ingest Service → Platform Postgres + Feature Job Queue
                           ↓
                        Feature Worker → Sentence-Transformer Embeddings → Qdrant
                           ↓
User clicks "Suggest Substitute" → Mealie → Substitution API → Qdrant search → ranked results
                           ↓
                     Accept/Reject feedback → Postgres (for retraining)

MLflow (port 30500) — experiment tracking
Grafana (port 30300) — cluster monitoring (Prometheus)
```

## Infrastructure

- **Cloud:** Chameleon Cloud, KVM@TACC site
- **VMs:** 3x `m1.xlarge` on Ubuntu 24.04 (1 control plane + 2 workers)
- **Networking:** dual-NIC — `sharednet1` (public) + private `192.168.1.0/24`
- **K8s:** kubespray (release-2.26), single control plane on node1
- **Storage:** local-path-provisioner for PVCs
- **Object Store:** `s3://data-proj01` on Chameleon (backups + ML training data)

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

## Repository Structure

```
mlops-forkwise/
├── provision/
│   └── provision.ipynb          # Jupyter notebook — run on Chameleon JupyterHub
├── tf/kvm/                      # Terraform configs for 3 VMs + network + floating IP
├── db/
│   └── init.sql                 # Platform postgres schema (recipe metadata, feedback, jobs)
├── k8s/
│   ├── mealie/                  # Mealie app + its postgres
│   ├── platform/                # Platform services (ingest, feature-worker, substitution, mlflow, qdrant, postgres)
│   └── monitoring/              # Prometheus + Grafana namespace
├── services/
│   ├── ingest-api/              # Polls Mealie for recipes, stores in platform DB
│   ├── feature-worker/          # Computes sentence-transformer embeddings → Qdrant
│   ├── substitution-api/        # Searches Qdrant for ingredient substitutes
│   └── mlflow/                  # MLflow with psycopg2 + boto3
├── scripts/
│   ├── bring_up.sh              # Deploys everything, restores from S3 if backups exist
│   ├── backup.sh                # Snapshots all state to S3
│   ├── teardown.sh              # Backs up then deletes all K8s resources
│   ├── setup_monitoring.sh      # Installs kube-prometheus-stack via Helm
│   └── _s3_common.sh            # Shared S3 helpers
└── clouds.yaml.example          # Template for OpenStack credentials
```

## Full Setup From Scratch

### Step 1: Provision VMs (Chameleon JupyterHub)

1. Go to [Chameleon JupyterHub](https://jupyter.chameleoncloud.org/)
2. Place `clouds.yaml` at `/work/clouds.yaml` with your KVM@TACC application credentials
3. Upload `provision/provision.ipynb` or clone the repo at `/work/`
4. Run all cells top to bottom — this creates a 24-hour lease, installs Terraform, and provisions 3 VMs with a floating IP

### Step 2: Copy SSH Key to node1 (from Windows PowerShell)

```powershell
scp -i "C:\Users\Krishan Guta\.ssh\forkwise_key" "C:\Users\Krishan Guta\.ssh\forkwise_key" cc@<FLOATING_IP>:~/.ssh/forkwise-key
```

### Step 3: Copy clouds.yaml to node1 (from Windows PowerShell)

```powershell
ssh -i "C:\Users\Krishan Guta\.ssh\forkwise_key" cc@<FLOATING_IP> "mkdir -p ~/.config/openstack"
scp -i "C:\Users\Krishan Guta\.ssh\forkwise_key" "C:\Users\Krishan Guta\path\to\clouds.yaml" cc@<FLOATING_IP>:~/.config/openstack/clouds.yaml
```

### Step 4: SSH into node1

```powershell
ssh -i "C:\Users\Krishan Guta\.ssh\forkwise_key" cc@<FLOATING_IP>
```

### Step 5: Set up SSH keys on node1

```bash
chmod 600 ~/.ssh/*
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -q -N ""
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
cat ~/.ssh/id_rsa.pub | ssh -i ~/.ssh/forkwise-key -o StrictHostKeyChecking=no cc@192.168.1.12 "cat >> ~/.ssh/authorized_keys"
cat ~/.ssh/id_rsa.pub | ssh -i ~/.ssh/forkwise-key -o StrictHostKeyChecking=no cc@192.168.1.13 "cat >> ~/.ssh/authorized_keys"
```

### Step 6: Disable IPv6 on all nodes

```bash
for ip in 192.168.1.11 192.168.1.12 192.168.1.13; do ssh -o StrictHostKeyChecking=no cc@$ip 'sudo sysctl -w net.ipv6.conf.ens3.disable_ipv6=1'; done
```

### Step 7: Install kubespray and deploy K8s

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

### Step 8: Post-kubespray setup

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

Verify:

```bash
kubectl get nodes
kubectl get pods -A
```

### Step 9: Clone repo and deploy everything

```bash
cd ~
git clone https://github.com/Krishan101/mlops-forkwise.git
cd mlops-forkwise
bash scripts/bring_up.sh
```

This single command:
- Disables IPv6 (Chameleon fix)
- Creates and attaches security groups for all NodePorts
- Deploys all K8s resources in order
- Checks S3 for backups and restores if found (mealie-db, platform-db, qdrant, mealie-data)
- Fixes the Mealie ingredient reference_id bug
- Installs Prometheus + Grafana monitoring

### Step 10: Verify

After `bring_up.sh` completes, all services are accessible:

| Service | URL | Credentials |
|---------|-----|-------------|
| Mealie | `http://<FLOATING_IP>:30900` | Create account on first visit |
| Substitution API | `http://<FLOATING_IP>:30808/health` | — |
| MLflow | `http://<FLOATING_IP>:30500` | — |
| Grafana | `http://<FLOATING_IP>:30300` | admin / forkwise-admin |

Test substitution from node1:

```bash
curl -s -X POST http://substitution-api.forkwise-platform:8080/substitute -H "Content-Type: application/json" -d '{"ingredient":"butter","recipe_name":"Classic Pancakes","top_k":5}' | python3 -m json.tool
```

## Building Docker Images

If you need to rebuild any service image (after code changes), run on node1:

```bash
# Fix IPv6 first (required for pip/HuggingFace downloads during build)
sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1
sudo sysctl -w net.ipv6.conf.lo.disable_ipv6=0

# Clean disk space if needed
docker system prune -a -f

# Login to GHCR
echo "<GHCR_TOKEN>" | docker login ghcr.io -u Krishan101 --password-stdin

# Build and push (example: substitution-api)
cd ~/mlops-forkwise/services/substitution-api
docker build --no-cache -t ghcr.io/krishan101/forkwise-substitution-api:0.1.4 .
docker push ghcr.io/krishan101/forkwise-substitution-api:0.1.4

# Make package public: github.com → Packages → Package settings → Change visibility → Public

# Update the running deployment
kubectl -n forkwise-platform set image deployment/substitution-api substitution-api=ghcr.io/krishan101/forkwise-substitution-api:0.1.4
```

For the Mealie fork image:

```bash
cd ~
git clone https://github.com/HivanshD/mealie.git mealie-fork
cd mealie-fork
docker build -f docker/Dockerfile -t ghcr.io/krishan101/forkwise-mealie:0.1.1 .
docker push ghcr.io/krishan101/forkwise-mealie:0.1.1
kubectl -n forkwise-app set image deployment/mealie mealie=ghcr.io/krishan101/forkwise-mealie:0.1.1
```

## Backup & Restore

**Backup** (run before teardown or anytime):

```bash
bash scripts/backup.sh
```

Snapshots to `s3://data-proj01/backups/`:
- `mealie-db/latest.sql.gz` — Mealie postgres dump
- `platform-db/latest.sql.gz` — Platform postgres dump (all DBs)
- `qdrant/latest.snapshot` — Qdrant vector snapshot
- `mealie-data/latest.tar.gz` — Mealie PVC data (uploads, config)

**Restore** happens automatically during `bring_up.sh` — if backups exist in S3, they are restored after each service deploys.

## Teardown

```bash
bash scripts/teardown.sh
```

This backs up everything to S3, then deletes all K8s namespaces. After teardown, destroy VMs from the provision notebook's teardown cells.

## Data Pipeline Flow

1. **User adds recipe in Mealie** (via UI or API)
2. **Ingest service** polls Mealie every 30s, detects new recipes, stores metadata + ingredients in platform postgres, queues a feature job
3. **Feature worker** picks up the job, computes a 384-dim sentence-transformer embedding for each ingredient (contextualized by recipe name), upserts into Qdrant
4. **User clicks "Suggest Substitute"** on any ingredient in Mealie's recipe page
5. **Mealie backend** calls the Substitution API's `/predict` endpoint
6. **Substitution API** embeds the query, searches Qdrant for similar ingredients from other recipes, returns ranked suggestions, logs the query to postgres
7. **User clicks Accept/Reject** — feedback is logged to postgres via the `/feedback` endpoint for future retraining

## Known Issues

- **Mealie `reference_id` bug:** The Mealie fork generates a new UUID for each ingredient on every recipe load instead of persisting it. `bring_up.sh` works around this with a SQL UPDATE. New recipes added after deployment may need the fix re-run: `kubectl -n forkwise-app exec deploy/mealie-db -- psql -U mealie -d mealie -c "UPDATE recipes_ingredients SET reference_id = gen_random_uuid() WHERE reference_id IS NULL"`
- **IPv6 on Chameleon:** IPv6 doesn't route on Chameleon KVM@TACC. Must disable it for Docker builds, pip installs, and HuggingFace model downloads. `bring_up.sh` handles this automatically. For manual Docker builds, run `sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1` first (keep loopback enabled for kubectl: `sudo sysctl -w net.ipv6.conf.lo.disable_ipv6=0`).
- **Disk space:** The node1 VM has 37GB disk. Docker images (especially Mealie and the sentence-transformer services) are large. Run `docker system prune -a -f` before building if disk is low.
- **Security groups:** Chameleon's shared project may have duplicate security group names. `bring_up.sh` creates and attaches by name; if duplicates exist, manual attachment by ID may be needed.
