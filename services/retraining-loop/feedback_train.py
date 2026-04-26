"""
ForkWise Feedback Training API
Triggered by schedule or manually via POST /train.
Pulls feedback from PostgreSQL, retrains GISMo, logs to MLflow,
updates Qdrant with new embeddings.

Similar pattern to photoprism's feedback_train.py but for GISMo/GNN.
"""

import logging
import os
import json
import time
import threading

import mlflow
import torch
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional

from gismo_model import GISMo
from data_loader import (
    load_flavorgraph,
    load_recipe1msubs,
    load_recipe1m_ingredients,
    load_feedback_pairs,
    SubstitutionDataset,
    normalize_ingredient,
)
from train import compute_mrr, export_embeddings, measure_inference_latency

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("feedback-trainer")

app = FastAPI(title="ForkWise Feedback Training API")

# --- Config ---
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow.forkwise-platform:5000")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "forkwise-gismo")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "platform-db")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "forkwise_mlops")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "forkwise")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "forkwise-secret-pw")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
MIN_FEEDBACK_SAMPLES = int(os.environ.get("MIN_FEEDBACK_SAMPLES", "10"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Training state
training_status = {
    "is_training": False,
    "last_trained": None,
    "status": "idle",
    "run_id": None,
    "metrics": {},
}


class TrainRequest(BaseModel):
    min_samples: Optional[int] = None
    epochs: Optional[int] = 20
    lr: Optional[float] = 5e-5
    run_name: Optional[str] = None


class TrainResponse(BaseModel):
    status: str
    message: str
    run_id: Optional[str] = None


def get_pg_conn():
    import psycopg2
    return psycopg2.connect(
        host=POSTGRES_HOST, port=POSTGRES_PORT, dbname=POSTGRES_DB,
        user=POSTGRES_USER, password=POSTGRES_PASSWORD,
    )


def update_qdrant_embeddings(embeddings_path):
    """Upload new GISMo embeddings to Qdrant, replacing the old collection."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    import hashlib

    with open(embeddings_path) as f:
        embeddings = json.load(f)

    if not embeddings:
        log.warning("no embeddings to upload")
        return

    embed_dim = len(embeddings[0]["embedding"])
    client = QdrantClient(host=QDRANT_HOST, port=int(QDRANT_PORT))
    collection = "ingredient_embeddings"

    # Recreate collection with new embeddings
    try:
        client.delete_collection(collection)
    except Exception:
        pass

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
    )

    points = []
    for emb in embeddings:
        h = hashlib.sha256(emb["ingredient"].encode()).hexdigest()
        point_id = int(h[:15], 16)
        points.append(PointStruct(
            id=point_id,
            vector=emb["embedding"],
            payload={
                "ingredient": emb["ingredient"],
                "ingredient_index": emb["index"],
                "source": "gismo",
            },
        ))

    # Batch upsert
    batch_size = 500
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=collection, points=points[i:i+batch_size])

    log.info(f"uploaded {len(points)} GISMo embeddings to Qdrant ({embed_dim}-d)")


def run_training(config):
    """Background training task."""
    global training_status

    training_status["is_training"] = True
    training_status["status"] = "training"

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        # Load data
        vocab, edge_index, edge_weight = load_flavorgraph()
        num_ingredients = len(vocab)
        train_pairs = load_recipe1msubs("train")
        val_pairs = load_recipe1msubs("val")

        # Load feedback
        conn = get_pg_conn()
        feedback_pairs = load_feedback_pairs(conn)
        accepted = [p for p in feedback_pairs if p["accepted"]]
        train_pairs.extend(accepted)
        log.info(f"training with {len(train_pairs)} pairs ({len(accepted)} from feedback)")

        recipes = load_recipe1m_ingredients(max_recipes=50000)
        train_dataset = SubstitutionDataset(train_pairs, vocab, recipes, num_negatives=10)
        val_dataset = SubstitutionDataset(val_pairs, vocab, recipes, num_negatives=10)

        model = GISMo(
            num_ingredients=num_ingredients,
            embed_dim=config.get("embed_dim", 300),
            num_gin_layers=2,
            decoder_hidden=256,
            dropout=0.25,
        ).to(device)

        node_ids = torch.arange(num_ingredients, dtype=torch.long)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)

        with mlflow.start_run(run_name=config.get("run_name", f"feedback-retrain-{int(time.time())}")):
            mlflow.log_params({
                "model": "GISMo",
                "training_type": "feedback_retrain",
                "num_ingredients": num_ingredients,
                "train_samples": len(train_dataset),
                "feedback_accepted": len(accepted),
                "feedback_total": len(feedback_pairs),
                "epochs": config["epochs"],
                "lr": config["lr"],
                "device": str(device),
            })

            training_status["run_id"] = mlflow.active_run().info.run_id

            best_mrr = 0
            t0 = time.time()

            for epoch in range(config["epochs"]):
                model.train()
                epoch_loss = 0
                num_batches = max(len(train_dataset) // 64, 1)

                for step in range(num_batches):
                    batch = train_dataset.get_batch(64)
                    optimizer.zero_grad()
                    loss = model(
                        node_ids.to(device), edge_index.to(device), edge_weight.to(device),
                        batch["source_ids"].to(device), batch["positive_ids"].to(device),
                        batch["negative_ids"].to(device), batch["recipe_ingredient_ids"],
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / num_batches
                mlflow.log_metric("train_loss", avg_loss, step=epoch)

                if (epoch + 1) % 5 == 0:
                    val_metrics = compute_mrr(model, val_dataset, node_ids, edge_index, edge_weight, device, max_eval=200)
                    for k, v in val_metrics.items():
                        mlflow.log_metric(f"val_{k}", v, step=epoch)
                    log.info(f"epoch {epoch+1} loss={avg_loss:.4f} val_mrr={val_metrics['mrr']:.4f}")

                    if val_metrics["mrr"] > best_mrr:
                        best_mrr = val_metrics["mrr"]
                        torch.save(model.state_dict(), "/tmp/gismo_best.pt")

            train_time = time.time() - t0
            mlflow.log_metric("total_train_time_sec", train_time)
            mlflow.log_metric("best_val_mrr", best_mrr)

            # Load best and export
            if os.path.exists("/tmp/gismo_best.pt"):
                model.load_state_dict(torch.load("/tmp/gismo_best.pt", map_location=device))

            # Inference latency
            latency = measure_inference_latency(model, vocab, node_ids, edge_index, edge_weight, device)
            for k, v in latency.items():
                mlflow.log_metric(k, v)

            # Export and update Qdrant
            emb_path = export_embeddings(model, vocab, node_ids, edge_index, edge_weight, device, "/tmp/gismo_embeddings.json")
            mlflow.log_artifact(emb_path, artifact_path="embeddings")
            mlflow.log_artifact("/tmp/gismo_best.pt", artifact_path="model")

            update_qdrant_embeddings(emb_path)

            # Stamp feedback rows as trained
            try:
                conn2 = get_pg_conn()
                with conn2.cursor() as cur:
                    cur.execute("UPDATE substitution_results SET trained_at = NOW() WHERE trained_at IS NULL")
                    conn2.commit()
                    log.info(f"stamped trained_at on {cur.rowcount} rows")
                conn2.close()
            except Exception as e:
                log.warning(f"could not stamp trained_at: {e}")

            training_status.update({
                "is_training": False,
                "last_trained": time.time(),
                "status": "completed",
                "run_id": mlflow.active_run().info.run_id,
                "metrics": {
                    "best_val_mrr": best_mrr,
                    "train_time_min": train_time / 60,
                    "avg_inference_ms": latency["avg_inference_ms"],
                },
            })

            log.info(f"training complete. run_id={mlflow.active_run().info.run_id} mrr={best_mrr:.4f}")

    except Exception as e:
        log.error(f"training failed: {e}", exc_info=True)
        training_status.update({
            "is_training": False,
            "status": f"failed: {str(e)}",
        })


@app.get("/health")
def health():
    return {"status": "ok", "training_status": training_status}


@app.get("/training/status")
def get_status():
    return training_status


@app.post("/train", response_model=TrainResponse)
def trigger_train(request: TrainRequest, background_tasks: BackgroundTasks):
    if training_status["is_training"]:
        raise HTTPException(409, "training already in progress")

    min_samples = request.min_samples or MIN_FEEDBACK_SAMPLES

    # Check feedback count
    try:
        conn = get_pg_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM feedback_events")
            count = cur.fetchone()[0]
        conn.close()
    except Exception as e:
        raise HTTPException(500, f"postgres error: {e}")

    if count < min_samples:
        return TrainResponse(
            status="skipped",
            message=f"only {count} feedback events (need {min_samples}). skipped.",
        )

    config = {
        "epochs": request.epochs or 20,
        "lr": request.lr or 5e-5,
        "embed_dim": 300,
        "run_name": request.run_name or f"feedback-retrain-{int(time.time())}",
    }

    background_tasks.add_task(run_training, config)

    return TrainResponse(
        status="started",
        message=f"training started with {count} feedback events. check /training/status",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
