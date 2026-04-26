"""
ForkWise Feature Worker
Polls the feature_jobs table for pending jobs. For each recipe, computes
a sentence-transformer embedding for every ingredient (contextualized by
the recipe name), and upserts the vectors into Qdrant.

Job lifecycle: pending → processing → done | failed
"""

import logging
import os
import time
import hashlib

import psycopg2
import psycopg2.extras
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","svc":"feature-worker","msg":"%(message)s"}',
)
log = logging.getLogger("feature-worker")

# --- Config ---
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "platform-db")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "forkwise_mlops")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "forkwise")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "forkwise-secret-pw")

QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "ingredient_embeddings"
EMBEDDING_DIM = 384
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "5"))


def get_pg_conn():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def ensure_collection(qdrant: QdrantClient):
    existing = {c.name for c in qdrant.get_collections().collections}
    if COLLECTION_NAME not in existing:
        log.info(f"creating Qdrant collection '{COLLECTION_NAME}' ({EMBEDDING_DIM}-d cosine)")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def stable_point_id(recipe_id: str, ingredient: str) -> int:
    """Deterministic integer ID from recipe_id + ingredient text."""
    h = hashlib.sha256(f"{recipe_id}:{ingredient}".encode()).hexdigest()
    return int(h[:15], 16)


def process_job(cur, job_id, recipe_id, model, qdrant):
    # Fetch recipe name and ingredients
    cur.execute("SELECT name FROM recipe_metadata WHERE recipe_id = %s", (recipe_id,))
    row = cur.fetchone()
    if not row:
        raise Exception(f"recipe {recipe_id} not found in recipe_metadata")
    recipe_name = row["name"]

    cur.execute(
        "SELECT ingredient, position FROM recipe_ingredients WHERE recipe_id = %s ORDER BY position",
        (recipe_id,),
    )
    ingredients = cur.fetchall()
    if not ingredients:
        raise Exception(f"no ingredients for recipe {recipe_id}")

    # Compute embeddings — contextualize each ingredient with the recipe name
    points = []
    for ing in ingredients:
        text = f"{ing['ingredient']} (in {recipe_name})"
        embedding = model.encode(text).tolist()

        point_id = stable_point_id(recipe_id, ing["ingredient"])
        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "recipe_id": recipe_id,
                "recipe_name": recipe_name,
                "ingredient": ing["ingredient"],
                "position": ing["position"],
            },
        ))

    # Upsert batch into Qdrant
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    # Mark done
    cur.execute(
        "UPDATE feature_jobs SET status='done', completed_at=NOW() WHERE job_id=%s",
        (job_id,),
    )
    log.info(f"  [done] {recipe_name}: {len(points)} ingredient embeddings")


def main():
    log.info(f"loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    log.info("model loaded")

    log.info(f"connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection(qdrant)

    log.info(f"polling feature_jobs every {POLL_INTERVAL}s")
    while True:
        try:
            conn = get_pg_conn()
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute("""
                        SELECT job_id, recipe_id
                        FROM feature_jobs
                        WHERE status = 'pending'
                        ORDER BY created_at
                        LIMIT 1
                        FOR UPDATE SKIP LOCKED
                    """)
                    row = cur.fetchone()

                    if row is None:
                        time.sleep(POLL_INTERVAL)
                        conn.close()
                        continue

                    job_id = row["job_id"]
                    recipe_id = row["recipe_id"]
                    cur.execute(
                        "UPDATE feature_jobs SET status='processing', started_at=NOW() WHERE job_id=%s",
                        (job_id,),
                    )
                    log.info(f"processing job {job_id}: recipe {recipe_id}")

                    try:
                        process_job(cur, job_id, recipe_id, model, qdrant)
                    except Exception as exc:
                        cur.execute(
                            "UPDATE feature_jobs SET status='failed', completed_at=NOW(), error=%s WHERE job_id=%s",
                            (str(exc), job_id),
                        )
                        log.error(f"  [failed] {recipe_id}: {exc}")

            conn.close()

        except psycopg2.OperationalError as exc:
            log.error(f"postgres connection error: {exc}. retrying in {POLL_INTERVAL}s")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
