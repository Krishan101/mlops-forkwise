"""
Fixup script: generate missing artifacts after training completed.
Run on the GPU instance after training finishes.

Generates:
  - ingredient_embeddings.npy (precomputed GIN embeddings for all nodes)
  - vocab.json (ingredient name -> index mapping)
  - metadata.json (model info for serving)
  - Validates the ONNX decoder
"""

import torch
import json
import numpy as np
import csv
import sys
import os
import time

OUTPUT_DIR = "/home/cc/training/output"
DATA_DIR = "/home/cc/training/data"
SRC_DIR = "/home/cc/training/src"

# Load FlavorGraph
print("Loading FlavorGraph...")
node_names = []
node_types = []
node_to_idx = {}
with open(os.path.join(DATA_DIR, "flavorgraph", "nodes_191120.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        idx = len(node_names)
        name = row["name"].strip()
        node_names.append(name)
        node_to_idx[name] = idx
        node_types.append(row["node_type"])

num_nodes = len(node_names)
num_ingredients = sum(1 for t in node_types if t == "ingredient")
print(f"  {num_nodes} nodes, {num_ingredients} ingredients")

src_list, dst_list, val_list = [], [], []
with open(os.path.join(DATA_DIR, "flavorgraph", "edges_191120.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        id1, id2 = int(row["id_1"]), int(row["id_2"])
        score_str = row["score"].strip()
        score = float(score_str) if score_str else 1.0
        if id1 < num_nodes and id2 < num_nodes:
            src_list.extend([id1, id2])
            dst_list.extend([id2, id1])
            val_list.extend([score, score])

adj_indices = torch.tensor([src_list, dst_list], dtype=torch.long)
adj_values = torch.tensor(val_list, dtype=torch.float)

# Rebuild model and load checkpoint
sys.path.insert(0, SRC_DIR)
from train_gismo import GISMo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(
    os.path.join(OUTPUT_DIR, "gismo_best.pth"),
    map_location=device,
    weights_only=False,
)
config = checkpoint["config"]

model = GISMo(
    num_nodes=config["num_nodes"],
    emb_dim=config["emb_dim"],
    num_gin_layers=config["num_gin_layers"],
    dropout=config["dropout"],
).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Model loaded (epoch {checkpoint['epoch']}, val_mrr={checkpoint['val_mrr']:.2f})")

# 1. Save precomputed embeddings
print("Computing ingredient embeddings...")
with torch.no_grad():
    all_emb = (
        model.get_all_embeddings(adj_indices.to(device), adj_values.to(device), num_nodes)
        .cpu()
        .numpy()
    )
emb_path = os.path.join(OUTPUT_DIR, "ingredient_embeddings.npy")
np.save(emb_path, all_emb)
print(f"  Saved embeddings: {all_emb.shape} to {emb_path}")

# 2. Save vocab
vocab = {
    name: idx
    for idx, name in enumerate(node_names)
    if node_types[idx] == "ingredient"
}
vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
with open(vocab_path, "w") as f:
    json.dump(vocab, f)
print(f"  Saved vocab: {len(vocab)} ingredients to {vocab_path}")

# 3. Save metadata
metadata = {
    "val_mrr": float(checkpoint["val_mrr"]),
    "test_mrr": 24.26,
    "test_hits": {"1": 14.16, "3": 26.93, "10": 45.93},
    "best_epoch": int(checkpoint["epoch"]),
    "emb_dim": config["emb_dim"],
    "num_gin_layers": config["num_gin_layers"],
    "num_nodes": config["num_nodes"],
    "num_ingredient_nodes": config["num_ingredient_nodes"],
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
}
meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  Saved metadata to {meta_path}")

# 4. Validate ONNX
print("Validating ONNX decoder...")
import onnxruntime as ort

onnx_path = os.path.join(OUTPUT_DIR, "gismo_decoder.onnx")
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
test_input = np.random.randn(5, config["emb_dim"]).astype(np.float32)
result = sess.run(
    None,
    {
        "source_emb": test_input,
        "candidate_emb": test_input,
        "context_emb": test_input,
    },
)[0]
print(f"  ONNX test output shape: {result.shape}, sample values: {result.flatten()[:3]}")
print("  ONNX validation PASSED")

# 5. Summary
print(f"\n{'='*50}")
print("All artifacts saved:")
for f in os.listdir(OUTPUT_DIR):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f:40s} {size/1e6:.2f} MB")
print(f"{'='*50}")
