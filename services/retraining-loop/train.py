"""
GISMo Training Script
Trains the GISMo model on Recipe1MSubs + FlavorGraph.
Logs all metrics, params, and artifacts to MLflow.
Exports trained embeddings to Qdrant for the substitution API.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --feedback  # include user feedback data
"""

import argparse
import json
import logging
import os
import time

import mlflow
import numpy as np
import torch
import yaml
from torch.optim import Adam

from gismo_model import GISMo
from data_loader import (
    load_flavorgraph,
    load_recipe1msubs,
    load_recipe1m_ingredients,
    SubstitutionDataset,
    normalize_ingredient,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gismo-train")


def compute_mrr(model, dataset, node_ids, edge_index, edge_weight, device, max_eval=500):
    """
    Compute Mean Reciprocal Rank on a dataset.
    For each (source, target) pair, rank the target among all ingredients.
    """
    model.eval()
    node_ids_t = node_ids.to(device)
    edge_index_t = edge_index.to(device)
    edge_weight_t = edge_weight.to(device) if edge_weight is not None else None

    with torch.no_grad():
        all_emb = model.encode_graph(node_ids_t, edge_index_t, edge_weight_t)

    reciprocal_ranks = []
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_10 = 0
    num_eval = min(len(dataset.samples), max_eval)

    for i in range(num_eval):
        sample = dataset.samples[i]
        src_idx = sample["source"]
        tgt_idx = sample["positive"]
        recipe_ings = sample["recipe_ingredients"]

        with torch.no_grad():
            context = model.get_context(all_emb, [recipe_ings])
            # Score all ingredients
            all_indices = torch.arange(model.num_ingredients, device=device).unsqueeze(0)
            source_t = torch.tensor([src_idx], device=device)
            scores = model.score(all_emb, source_t, all_indices, context).squeeze(0)

            # Mask out the source ingredient
            scores[src_idx] = float("-inf")

            # Find rank of target
            sorted_indices = torch.argsort(scores, descending=True)
            rank = (sorted_indices == tgt_idx).nonzero(as_tuple=True)[0]
            if len(rank) > 0:
                rank_val = rank[0].item() + 1
                reciprocal_ranks.append(1.0 / rank_val)
                if rank_val <= 1:
                    hits_at_1 += 1
                if rank_val <= 3:
                    hits_at_3 += 1
                if rank_val <= 10:
                    hits_at_10 += 1
            else:
                reciprocal_ranks.append(0.0)

    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    metrics = {
        "mrr": mrr,
        "hit_at_1": hits_at_1 / num_eval if num_eval > 0 else 0,
        "hit_at_3": hits_at_3 / num_eval if num_eval > 0 else 0,
        "hit_at_10": hits_at_10 / num_eval if num_eval > 0 else 0,
        "num_evaluated": num_eval,
    }
    return metrics


def export_embeddings(model, vocab, node_ids, edge_index, edge_weight, device, output_path):
    """Export trained ingredient embeddings as JSON for Qdrant indexing."""
    model.eval()
    with torch.no_grad():
        all_emb = model.encode_graph(
            node_ids.to(device),
            edge_index.to(device),
            edge_weight.to(device) if edge_weight is not None else None,
        )
    emb_np = all_emb.cpu().numpy()

    # Reverse vocab: index → name
    idx_to_name = {v: k for k, v in vocab.items()}

    embeddings = []
    for idx in range(len(vocab)):
        embeddings.append({
            "ingredient": idx_to_name.get(idx, f"unk_{idx}"),
            "index": idx,
            "embedding": emb_np[idx].tolist(),
        })

    with open(output_path, "w") as f:
        json.dump(embeddings, f)

    log.info(f"exported {len(embeddings)} embeddings to {output_path}")
    return output_path


def measure_inference_latency(model, vocab, node_ids, edge_index, edge_weight, device, num_runs=50):
    """Measure average inference latency for a single substitution query."""
    model.eval()
    node_ids_t = node_ids.to(device)
    edge_index_t = edge_index.to(device)
    edge_weight_t = edge_weight.to(device) if edge_weight is not None else None

    with torch.no_grad():
        all_emb = model.encode_graph(node_ids_t, edge_index_t, edge_weight_t)

    sample_ids = list(range(min(100, len(vocab))))
    latencies = []

    for _ in range(num_runs):
        src = torch.tensor([sample_ids[0]], device=device)
        recipe = [sample_ids[:5]]
        all_cands = torch.arange(len(vocab), device=device).unsqueeze(0)

        start = time.perf_counter()
        with torch.no_grad():
            ctx = model.get_context(all_emb, recipe)
            _ = model.score(all_emb, src, all_cands, ctx)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "avg_inference_ms": np.mean(latencies),
        "p50_inference_ms": np.percentile(latencies, 50),
        "p95_inference_ms": np.percentile(latencies, 95),
        "p99_inference_ms": np.percentile(latencies, 99),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--feedback", action="store_true", help="Include user feedback data")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device: {device}")

    # MLflow setup
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg.get("mlflow", {}).get("tracking_uri", "http://mlflow:5000"))
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = cfg.get("mlflow", {}).get("experiment_name", "forkwise-gismo")
    mlflow.set_experiment(experiment_name)

    # Load data
    vocab, edge_index, edge_weight = load_flavorgraph()
    num_ingredients = len(vocab)

    train_pairs = load_recipe1msubs("train")
    val_pairs = load_recipe1msubs("val")
    test_pairs = load_recipe1msubs("test")

    # Optionally load Recipe1M for recipe context
    recipes = None
    if cfg.get("use_recipe_context", True):
        recipes = load_recipe1m_ingredients(max_recipes=cfg.get("max_recipes", 100000))

    # Optionally include feedback data
    feedback_pairs = []
    if args.feedback:
        try:
            import psycopg2
            from data_loader import load_feedback_pairs
            conn = psycopg2.connect(
                host=os.environ.get("POSTGRES_HOST", "platform-db"),
                port=int(os.environ.get("POSTGRES_PORT", "5432")),
                dbname=os.environ.get("POSTGRES_DB", "forkwise_mlops"),
                user=os.environ.get("POSTGRES_USER", "forkwise"),
                password=os.environ.get("POSTGRES_PASSWORD", "forkwise-secret-pw"),
            )
            feedback_pairs = load_feedback_pairs(conn)
            conn.close()
            # Add accepted feedback as additional positive pairs
            accepted = [p for p in feedback_pairs if p["accepted"]]
            train_pairs.extend(accepted)
            log.info(f"added {len(accepted)} accepted feedback pairs to training data")
        except Exception as e:
            log.warning(f"could not load feedback: {e}")

    # Build datasets
    num_neg = cfg.get("num_negatives", 10)
    train_dataset = SubstitutionDataset(train_pairs, vocab, recipes, num_neg)
    val_dataset = SubstitutionDataset(val_pairs, vocab, recipes, num_neg)
    test_dataset = SubstitutionDataset(test_pairs, vocab, recipes, num_neg)

    # Model
    embed_dim = cfg.get("embed_dim", 300)
    num_gin_layers = cfg.get("num_gin_layers", 2)
    model = GISMo(
        num_ingredients=num_ingredients,
        embed_dim=embed_dim,
        num_gin_layers=num_gin_layers,
        decoder_hidden=cfg.get("decoder_hidden", 256),
        dropout=cfg.get("dropout", 0.25),
    ).to(device)

    node_ids = torch.arange(num_ingredients, dtype=torch.long)
    optimizer = Adam(model.parameters(), lr=float(cfg.get("lr", 5e-5)), weight_decay=float(cfg.get("weight_decay", 1e-4)))

    epochs = cfg.get("epochs", 50)
    batch_size = cfg.get("batch_size", 64)
    eval_every = cfg.get("eval_every", 5)

    with mlflow.start_run(run_name=cfg.get("run_name", f"gismo-{int(time.time())}")):
        # Log all params
        mlflow.log_params({
            "model": "GISMo",
            "num_ingredients": num_ingredients,
            "embed_dim": embed_dim,
            "num_gin_layers": num_gin_layers,
            "decoder_hidden": cfg.get("decoder_hidden", 256),
            "dropout": cfg.get("dropout", 0.25),
            "lr": cfg.get("lr", 5e-5),
            "weight_decay": cfg.get("weight_decay", 1e-4),
            "epochs": epochs,
            "batch_size": batch_size,
            "num_negatives": num_neg,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "feedback_pairs": len(feedback_pairs),
            "use_recipe_context": cfg.get("use_recipe_context", True),
            "device": str(device),
            "graph_nodes": num_ingredients,
            "graph_edges": edge_index.shape[1],
        })

        best_mrr = 0
        best_epoch = 0
        t0 = time.time()

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            num_batches = max(len(train_dataset) // batch_size, 1)

            for step in range(num_batches):
                batch = train_dataset.get_batch(batch_size)
                optimizer.zero_grad()

                loss = model(
                    node_ids.to(device),
                    edge_index.to(device),
                    edge_weight.to(device),
                    batch["source_ids"].to(device),
                    batch["positive_ids"].to(device),
                    batch["negative_ids"].to(device),
                    batch["recipe_ingredient_ids"],
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            log.info(f"epoch {epoch+1}/{epochs} loss={avg_loss:.4f}")

            # Evaluate
            if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
                val_metrics = compute_mrr(
                    model, val_dataset, node_ids, edge_index, edge_weight, device,
                    max_eval=cfg.get("max_eval", 500),
                )
                for k, v in val_metrics.items():
                    mlflow.log_metric(f"val_{k}", v, step=epoch)
                log.info(f"  val MRR={val_metrics['mrr']:.4f} Hit@1={val_metrics['hit_at_1']:.4f} "
                         f"Hit@3={val_metrics['hit_at_3']:.4f} Hit@10={val_metrics['hit_at_10']:.4f}")

                if val_metrics["mrr"] > best_mrr:
                    best_mrr = val_metrics["mrr"]
                    best_epoch = epoch + 1
                    # Save best model
                    torch.save(model.state_dict(), "/tmp/gismo_best.pt")
                    mlflow.log_artifact("/tmp/gismo_best.pt", artifact_path="model")

        train_time = time.time() - t0
        mlflow.log_metric("total_train_time_sec", train_time)
        mlflow.log_metric("total_train_time_min", train_time / 60)
        mlflow.log_metric("best_val_mrr", best_mrr)
        mlflow.log_metric("best_epoch", best_epoch)

        # Load best model for final eval
        model.load_state_dict(torch.load("/tmp/gismo_best.pt", map_location=device))

        # Test evaluation
        test_metrics = compute_mrr(
            model, test_dataset, node_ids, edge_index, edge_weight, device,
            max_eval=cfg.get("max_eval", 500),
        )
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        log.info(f"TEST MRR={test_metrics['mrr']:.4f} Hit@1={test_metrics['hit_at_1']:.4f} "
                 f"Hit@3={test_metrics['hit_at_3']:.4f} Hit@10={test_metrics['hit_at_10']:.4f}")

        # Inference latency
        latency_metrics = measure_inference_latency(
            model, vocab, node_ids, edge_index, edge_weight, device
        )
        for k, v in latency_metrics.items():
            mlflow.log_metric(k, v)
        log.info(f"inference latency: avg={latency_metrics['avg_inference_ms']:.1f}ms "
                 f"p95={latency_metrics['p95_inference_ms']:.1f}ms")

        # Export embeddings
        emb_path = export_embeddings(
            model, vocab, node_ids, edge_index, edge_weight, device,
            "/tmp/gismo_embeddings.json"
        )
        mlflow.log_artifact(emb_path, artifact_path="embeddings")

        # Save vocab
        vocab_path = "/tmp/gismo_vocab.json"
        with open(vocab_path, "w") as f:
            json.dump(vocab, f)
        mlflow.log_artifact(vocab_path, artifact_path="model")

        # Save config
        mlflow.log_artifact(args.config, artifact_path="config")

        log.info(f"training complete. best MRR={best_mrr:.4f} at epoch {best_epoch}")
        log.info(f"run_id: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
