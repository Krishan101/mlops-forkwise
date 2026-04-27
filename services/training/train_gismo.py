"""
GISMo Training Script for ForkWise
===================================
Graph-based Ingredient Substitution Module

Architecture (following the paper):
  - Ingredient Encoder: 2-layer GIN over FlavorGraph (6653 ingredient + 1645 compound nodes)
  - Context Encoder: averages ingredient embeddings from a recipe
  - Substitution Decoder: 3-layer MLP scoring (source, candidate, context) triples

Data:
  - Recipe1MSubs: train/val/test JSON with {recipe_id, original, replacement}
  - FlavorGraph: nodes CSV + edges CSV
  - context_map.json: recipe_id -> list of ingredient strings
  - merge_dict.pkl: ingredient name normalization

Training:
  - Contrastive loss with negative sampling
  - Early stopping on validation MRR
  - Logs to MLflow
  - Exports best model as ONNX
"""

import argparse
import csv
import json
import os
import pickle
import time
import uuid
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GINLayer(nn.Module):
    """Graph Isomorphism Network layer."""
    def __init__(self, in_dim, out_dim, eps_init=0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps_init))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, h, adj_indices, adj_values, num_nodes):
        # Sparse message passing
        # adj_indices: [2, num_edges], adj_values: [num_edges]
        src, dst = adj_indices
        messages = adj_values.unsqueeze(-1) * h[src]  # [num_edges, dim]

        # Aggregate messages for each node
        agg = torch.zeros(num_nodes, h.size(1), device=h.device)
        agg.index_add_(0, dst, messages)

        # GIN update
        out = (1 + self.eps) * h + agg
        return self.mlp(out)


class IngredientEncoder(nn.Module):
    """2-layer GIN encoder over FlavorGraph."""
    def __init__(self, num_nodes, emb_dim, num_layers=2, dropout=0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, emb_dim)
        self.layers = nn.ModuleList([
            GINLayer(emb_dim, emb_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, adj_indices, adj_values, num_nodes):
        h = self.embedding.weight  # [num_nodes, emb_dim]
        for layer in self.layers:
            h = layer(h, adj_indices, adj_values, num_nodes)
            h = F.relu(h)
            h = self.dropout(h)
        return h


class SubstitutionDecoder(nn.Module):
    """3-layer MLP that scores (source, candidate, context) triples."""
    def __init__(self, emb_dim, dropout=0.25):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, source_emb, candidate_emb, context_emb):
        # source_emb: [batch, dim], candidate_emb: [batch, dim], context_emb: [batch, dim]
        combined = torch.cat([source_emb, candidate_emb, context_emb], dim=-1)
        return self.mlp(combined).squeeze(-1)  # [batch]


class GISMo(nn.Module):
    def __init__(self, num_nodes, emb_dim, num_gin_layers=2, dropout=0.25):
        super().__init__()
        self.ingredient_encoder = IngredientEncoder(num_nodes, emb_dim, num_gin_layers, dropout)
        self.decoder = SubstitutionDecoder(emb_dim, dropout)
        self.emb_dim = emb_dim

    def get_all_embeddings(self, adj_indices, adj_values, num_nodes):
        """Run GIN forward pass, return all node embeddings."""
        return self.ingredient_encoder(adj_indices, adj_values, num_nodes)

    def score(self, all_emb, source_idx, candidate_idx, context_indices_list):
        """
        Score substitution candidates.
        source_idx: [batch] - index of source ingredient
        candidate_idx: [batch] - index of candidate ingredient
        context_indices_list: list of lists - ingredient indices for each recipe
        """
        source_emb = all_emb[source_idx]
        candidate_emb = all_emb[candidate_idx]

        # Context = average of recipe ingredient embeddings
        context_embs = []
        for indices in context_indices_list:
            if len(indices) > 0:
                ctx = all_emb[indices].mean(dim=0)
            else:
                ctx = torch.zeros(self.emb_dim, device=all_emb.device)
            context_embs.append(ctx)
        context_emb = torch.stack(context_embs)

        return self.decoder(source_emb, candidate_emb, context_emb)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class FlavorGraph:
    """Loads FlavorGraph from CSV files."""
    def __init__(self, nodes_path, edges_path):
        self.node_names = []  # index -> name
        self.node_to_idx = {}  # name -> index
        self.node_types = []  # index -> type
        self.num_ingredient_nodes = 0

        # Load nodes - reindex from 0
        with open(nodes_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = len(self.node_names)
                name = row['name'].strip()
                self.node_names.append(name)
                self.node_to_idx[name] = idx
                self.node_types.append(row['node_type'])
                if row['node_type'] == 'ingredient':
                    self.num_ingredient_nodes += 1

        self.num_nodes = len(self.node_names)

        # Load edges
        src_list, dst_list, val_list = [], [], []
        with open(edges_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                id1 = int(row['id_1'])
                id2 = int(row['id_2'])
                score_str = row['score'].strip()
                score = float(score_str) if score_str else 1.0
                # Map original node_ids to our reindexed ids
                # The CSV uses the original node_id from the nodes file
                # We need to map: original_id -> our sequential index
                if id1 < self.num_nodes and id2 < self.num_nodes:
                    src_list.append(id1)
                    dst_list.append(id2)
                    val_list.append(score)
                    # Undirected: add reverse edge
                    src_list.append(id2)
                    dst_list.append(id1)
                    val_list.append(score)

        self.adj_indices = torch.tensor([src_list, dst_list], dtype=torch.long)
        self.adj_values = torch.tensor(val_list, dtype=torch.float)
        print(f"FlavorGraph: {self.num_nodes} nodes ({self.num_ingredient_nodes} ingredients), "
              f"{len(val_list)} edges (bidirectional)")

    def get_ingredient_idx(self, name, merge_dict=None):
        """Look up ingredient index, applying merge_dict if needed."""
        if name in self.node_to_idx:
            return self.node_to_idx[name]
        if merge_dict and name in merge_dict:
            merged = merge_dict[name]
            if merged in self.node_to_idx:
                return self.node_to_idx[merged]
        return None


class SubstitutionDataset(Dataset):
    """Dataset of (source_idx, target_idx, context_indices) tuples."""
    def __init__(self, samples, graph, context_map, merge_dict, num_neg=64):
        self.graph = graph
        self.num_neg = num_neg
        self.num_ingredient_nodes = graph.num_ingredient_nodes
        self.valid_samples = []

        skipped = 0
        for sample in samples:
            src_idx = graph.get_ingredient_idx(sample['original'], merge_dict)
            tgt_idx = graph.get_ingredient_idx(sample['replacement'], merge_dict)
            if src_idx is None or tgt_idx is None:
                skipped += 1
                continue

            # Get context ingredients for this recipe
            recipe_id = sample['recipe_id']
            context_indices = []
            if recipe_id in context_map:
                for ing_str in context_map[recipe_id]:
                    # Normalize: lowercase, replace spaces with underscores
                    ing_name = ing_str.lower().strip().replace(' ', '_')
                    idx = graph.get_ingredient_idx(ing_name, merge_dict)
                    if idx is not None:
                        context_indices.append(idx)

            self.valid_samples.append({
                'source_idx': src_idx,
                'target_idx': tgt_idx,
                'context_indices': context_indices,
            })

        print(f"  Loaded {len(self.valid_samples)} valid samples ({skipped} skipped)")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        src = sample['source_idx']
        tgt = sample['target_idx']
        ctx = sample['context_indices']

        # Generate negative samples
        negatives = []
        while len(negatives) < self.num_neg:
            neg = torch.randint(0, self.num_ingredient_nodes, (1,)).item()
            if neg != src and neg != tgt:
                negatives.append(neg)

        return {
            'source': src,
            'target': tgt,
            'negatives': negatives,
            'context': ctx,
        }


def collate_fn(batch):
    """Custom collate for variable-length context lists."""
    return {
        'source': torch.tensor([b['source'] for b in batch]),
        'target': torch.tensor([b['target'] for b in batch]),
        'negatives': torch.tensor([b['negatives'] for b in batch]),
        'context': [b['context'] for b in batch],
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataset, graph, device, max_candidates=None):
    """Compute MRR and Hit@k on a dataset."""
    model.eval()

    all_emb = model.get_all_embeddings(
        graph.adj_indices.to(device),
        graph.adj_values.to(device),
        graph.num_nodes
    )

    num_ingredients = graph.num_ingredient_nodes
    if max_candidates is None:
        max_candidates = num_ingredients

    reciprocal_ranks = []
    hits_at = {1: 0, 3: 0, 10: 0}
    total = 0

    for sample in dataset.valid_samples:
        src_idx = sample['source_idx']
        tgt_idx = sample['target_idx']
        ctx_indices = sample['context_indices']

        # Score all ingredient candidates (excluding source)
        candidates = [i for i in range(min(num_ingredients, max_candidates)) if i != src_idx]
        if tgt_idx not in candidates:
            candidates.append(tgt_idx)

        cand_tensor = torch.tensor(candidates, device=device)
        src_tensor = torch.tensor([src_idx], device=device).expand(len(candidates))

        # Build context for each candidate (same context repeated)
        ctx_list = [ctx_indices] * len(candidates)

        # Score in chunks to avoid OOM
        chunk_size = 512
        all_scores = []
        for i in range(0, len(candidates), chunk_size):
            chunk_cand = cand_tensor[i:i+chunk_size]
            chunk_src = src_tensor[i:i+chunk_size]
            chunk_ctx = ctx_list[i:i+chunk_size]
            scores = model.score(all_emb, chunk_src, chunk_cand, chunk_ctx)
            all_scores.append(scores)

        all_scores = torch.cat(all_scores)

        # Find rank of target
        tgt_pos = candidates.index(tgt_idx)
        tgt_score = all_scores[tgt_pos]
        rank = (all_scores > tgt_score).sum().item() + 1

        reciprocal_ranks.append(1.0 / rank)
        for k in hits_at:
            if rank <= k:
                hits_at[k] += 1
        total += 1

    mrr = np.mean(reciprocal_ranks) * 100
    hits = {k: v / total * 100 for k, v in hits_at.items()}

    return mrr, hits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, graph, device):
    model.train()
    total_loss = 0
    num_batches = 0

    all_emb = model.get_all_embeddings(
        graph.adj_indices.to(device),
        graph.adj_values.to(device),
        graph.num_nodes
    )

    for batch in dataloader:
        source = batch['source'].to(device)
        target = batch['target'].to(device)
        negatives = batch['negatives'].to(device)  # [batch, num_neg]
        context_list = batch['context']

        batch_size = source.size(0)
        num_neg = negatives.size(1)

        # Score positive pairs
        pos_scores = model.score(all_emb, source, target, context_list)

        # Score negative pairs
        # Expand source and context for each negative
        source_exp = source.unsqueeze(1).expand(-1, num_neg).reshape(-1)
        neg_flat = negatives.reshape(-1)
        ctx_exp = []
        for ctx in context_list:
            ctx_exp.extend([ctx] * num_neg)

        neg_scores = model.score(all_emb, source_exp, neg_flat, ctx_exp)
        neg_scores = neg_scores.view(batch_size, num_neg)

        # Contrastive loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # [batch, 1+num_neg]
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)  # positive is at index 0
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Recompute embeddings after parameter update
        all_emb = model.get_all_embeddings(
            graph.adj_indices.to(device),
            graph.adj_values.to(device),
            graph.num_nodes
        )

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='training/data')
    parser.add_argument('--output-dir', default='training/output')
    parser.add_argument('--emb-dim', type=int, default=300)
    parser.add_argument('--num-gin-layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--num-neg', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--eval-candidates', type=int, default=500,
                        help='Number of candidates to score during eval (for speed). Use 0 for all.')
    parser.add_argument('--mlflow-uri', default=None)
    parser.add_argument('--no-mlflow', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Load FlavorGraph ---
    print("Loading FlavorGraph...")
    graph = FlavorGraph(
        os.path.join(args.data_dir, 'flavorgraph', 'nodes_191120.csv'),
        os.path.join(args.data_dir, 'flavorgraph', 'edges_191120.csv'),
    )

    # --- Load merge dict ---
    print("Loading merge dictionary...")
    with open(os.path.join(args.data_dir, 'recipe1msubs', 'merge_dict.pkl'), 'rb') as f:
        merge_dict = pickle.load(f)
    print(f"  {len(merge_dict)} merge entries")

    # --- Load context map ---
    print("Loading context map...")
    with open(os.path.join(args.data_dir, 'recipe1m', 'context_map.json')) as f:
        context_map_raw = json.load(f)
    print(f"  {len(context_map_raw)} recipes")

    # --- Load substitution data ---
    print("Loading substitution datasets...")
    for split in ['train', 'val', 'test']:
        with open(os.path.join(args.data_dir, 'recipe1msubs', f'{split}.json')) as f:
            data = json.load(f)
        if split == 'train':
            train_data = data
        elif split == 'val':
            val_data = data
        else:
            test_data = data

    eval_candidates = args.eval_candidates if args.eval_candidates > 0 else None

    print(f"\nBuilding datasets (num_neg={args.num_neg})...")
    print("Train:")
    train_dataset = SubstitutionDataset(train_data, graph, context_map_raw, merge_dict, args.num_neg)
    print("Val:")
    val_dataset = SubstitutionDataset(val_data, graph, context_map_raw, merge_dict, args.num_neg)
    print("Test:")
    test_dataset = SubstitutionDataset(test_data, graph, context_map_raw, merge_dict, args.num_neg)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # --- Model ---
    print(f"\nModel: GISMo (emb_dim={args.emb_dim}, layers={args.num_gin_layers}, dropout={args.dropout})")
    model = GISMo(
        num_nodes=graph.num_nodes,
        emb_dim=args.emb_dim,
        num_gin_layers=args.num_gin_layers,
        dropout=args.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- MLflow setup ---
    use_mlflow = not args.no_mlflow
    if use_mlflow:
        try:
            import mlflow
            import mlflow.pytorch
            if args.mlflow_uri:
                mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment("gismo-ingredient-substitution")
            mlflow.start_run(log_system_metrics=True)
            mlflow.log_params({
                'emb_dim': args.emb_dim,
                'num_gin_layers': args.num_gin_layers,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'dropout': args.dropout,
                'num_neg': args.num_neg,
                'batch_size': args.batch_size,
                'max_epochs': args.max_epochs,
                'patience': args.patience,
                'num_graph_nodes': graph.num_nodes,
                'num_ingredient_nodes': graph.num_ingredient_nodes,
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'test_samples': len(test_dataset),
                'num_params': num_params,
                'eval_candidates': str(eval_candidates),
            })
            print("MLflow logging enabled")
        except Exception as e:
            print(f"MLflow setup failed: {e}, continuing without MLflow")
            use_mlflow = False

    # --- Training loop ---
    best_val_mrr = 0
    patience_counter = 0
    best_model_path = os.path.join(args.output_dir, 'gismo_best.pth')

    config = {
        'emb_dim': args.emb_dim,
        'num_gin_layers': args.num_gin_layers,
        'dropout': args.dropout,
        'num_nodes': graph.num_nodes,
        'num_ingredient_nodes': graph.num_ingredient_nodes,
    }

    print(f"\n{'='*60}")
    print(f"Training for up to {args.max_epochs} epochs (patience={args.patience})")
    print(f"{'='*60}\n")

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, graph, device)

        # Evaluate on validation set
        val_mrr, val_hits = evaluate(model, val_dataset, graph, device, eval_candidates)

        epoch_time = time.time() - t0

        print(f"Epoch {epoch:3d} | loss={train_loss:.4f} | "
              f"val_MRR={val_mrr:.2f} H@1={val_hits[1]:.2f} H@3={val_hits[3]:.2f} H@10={val_hits[10]:.2f} | "
              f"{epoch_time:.1f}s")

        if use_mlflow:
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_mrr': val_mrr,
                'val_hit_at_1': val_hits[1],
                'val_hit_at_3': val_hits[3],
                'val_hit_at_10': val_hits[10],
                'epoch_time_s': epoch_time,
            }, step=epoch)

        # Early stopping
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'val_mrr': val_mrr,
                'val_hits': val_hits,
                'optimizer_state_dict': optimizer.state_dict(),
            }, best_model_path)
            print(f"  -> New best! Saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # --- Load best model and evaluate on test set ---
    print(f"\n{'='*60}")
    print("Loading best model for final evaluation...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Best validation MRR: {checkpoint['val_mrr']:.2f} (epoch {checkpoint['epoch']})")

    # Full evaluation on test set (all candidates)
    print("Evaluating on test set (all ingredient candidates)...")
    test_mrr, test_hits = evaluate(model, test_dataset, graph, device, max_candidates=None)
    print(f"Test MRR={test_mrr:.2f} H@1={test_hits[1]:.2f} H@3={test_hits[3]:.2f} H@10={test_hits[10]:.2f}")

    if use_mlflow:
        mlflow.log_metrics({
            'test_mrr': test_mrr,
            'test_hit_at_1': test_hits[1],
            'test_hit_at_3': test_hits[3],
            'test_hit_at_10': test_hits[10],
            'best_epoch': checkpoint['epoch'],
        })

    # --- Export ONNX ---
    print("\nExporting ONNX model (decoder only)...")
    model.eval()
    onnx_path = os.path.join(args.output_dir, 'gismo_decoder.onnx')
    dummy_input = torch.randn(1, args.emb_dim * 3, device=device)
    torch.onnx.export(
        model.decoder,
        (dummy_input[:, :args.emb_dim], dummy_input[:, args.emb_dim:2*args.emb_dim], dummy_input[:, 2*args.emb_dim:]),
        onnx_path,
        input_names=['source_emb', 'candidate_emb', 'context_emb'],
        output_names=['score'],
        dynamic_axes={
            'source_emb': {0: 'batch'},
            'candidate_emb': {0: 'batch'},
            'context_emb': {0: 'batch'},
            'score': {0: 'batch'},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  ONNX decoder saved to {onnx_path}")

    # Validate ONNX
    try:
        import onnxruntime as ort
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        test_src = torch.randn(5, args.emb_dim).numpy()
        test_cand = torch.randn(5, args.emb_dim).numpy()
        test_ctx = torch.randn(5, args.emb_dim).numpy()
        onnx_out = ort_session.run(None, {
            'source_emb': test_src,
            'candidate_emb': test_cand,
            'context_emb': test_ctx,
        })[0]

        torch_out = model.decoder(
            torch.tensor(test_src), torch.tensor(test_cand), torch.tensor(test_ctx)
        ).detach().cpu().numpy()

        max_diff = np.max(np.abs(onnx_out - torch_out))
        print(f"  ONNX validation: max diff = {max_diff:.8f} (should be < 1e-5)")
        assert max_diff < 1e-4, f"ONNX validation failed: max diff {max_diff}"
        print("  ONNX validation PASSED")
    except ImportError:
        print("  onnxruntime not available, skipping ONNX validation")

    # --- Save precomputed embeddings ---
    print("Saving precomputed ingredient embeddings...")
    with torch.no_grad():
        all_emb = model.get_all_embeddings(
            graph.adj_indices.to(device),
            graph.adj_values.to(device),
            graph.num_nodes
        ).cpu().numpy()
    emb_path = os.path.join(args.output_dir, 'ingredient_embeddings.npy')
    np.save(emb_path, all_emb)
    print(f"  Saved {all_emb.shape} to {emb_path}")

    # --- Save vocab mapping ---
    vocab = {name: idx for idx, name in enumerate(graph.node_names)
             if graph.node_types[idx] == 'ingredient'}
    vocab_path = os.path.join(args.output_dir, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    print(f"  Vocab ({len(vocab)} ingredients) saved to {vocab_path}")

    # --- Save metadata ---
    metadata = {
        'train_mrr': float(checkpoint['val_mrr']),
        'test_mrr': float(test_mrr),
        'test_hits': {str(k): float(v) for k, v in test_hits.items()},
        'best_epoch': int(checkpoint['epoch']),
        'emb_dim': args.emb_dim,
        'num_gin_layers': args.num_gin_layers,
        'num_nodes': graph.num_nodes,
        'num_ingredient_nodes': graph.num_ingredient_nodes,
        'timestamp': time.strftime('%Y%m%d_%H%M%S'),
    }
    meta_path = os.path.join(args.output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {meta_path}")

    # --- MLflow artifacts ---
    if use_mlflow:
        mlflow.log_artifact(best_model_path)
        mlflow.log_artifact(onnx_path)
        mlflow.log_artifact(emb_path)
        mlflow.log_artifact(vocab_path)
        mlflow.log_artifact(meta_path)
        mlflow.end_run()
        print("MLflow run completed")

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best val MRR: {checkpoint['val_mrr']:.2f}")
    print(f"  Test MRR:     {test_mrr:.2f}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
