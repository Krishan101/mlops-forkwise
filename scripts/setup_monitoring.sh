#!/usr/bin/env bash
# Install Prometheus + Grafana via kube-prometheus-stack helm chart.
# Run from node1 after bring_up.sh or standalone.

set -euo pipefail

GREEN=$'\e[32m' RESET=$'\e[0m'
log() { echo -e "${GREEN}[monitoring]${RESET} $*"; }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

log "adding prometheus-community helm repo..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo update

log "creating monitoring namespace..."
kubectl apply -f "$REPO_ROOT/k8s/monitoring/namespace.yaml"

log "installing kube-prometheus-stack..."
helm upgrade --install kube-prometheus prometheus-community/kube-prometheus-stack \
    --namespace monitoring \
    --set grafana.service.type=NodePort \
    --set grafana.service.nodePort=30300 \
    --set grafana.adminPassword=forkwise-admin \
    --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
    --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
    --wait --timeout 5m

log "waiting for grafana..."
kubectl -n monitoring rollout status deployment/kube-prometheus-grafana --timeout=3m

NODE1_IP="$(curl -s --max-time 5 https://api.ipify.org || echo '<unknown>')"

log ""
log "============================================"
log "  Monitoring deployed"
log "============================================"
log ""
log "Grafana    : http://${NODE1_IP}:30300"
log "             user: admin  pass: forkwise-admin"
log "Prometheus : kube-prometheus-prometheus.monitoring:9090 (cluster-internal)"
