"""
ForkWise Substitution API
Accepts a query like "substitute for butter in Classic Pancakes",
searches Qdrant for similar ingredients across all recipes,
and returns ranked substitution suggestions.

Logs every query + results to postgres for the feedback/retraining loop.
"""

import logging
import os
import uuid

import psycopg2
import psycopg2.extras
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","svc":"substitution-api","msg":"%(message)s"}',
)
log = logging.getLogger("substitution-api")

# --- Config ---
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "platform-db")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "forkwise_mlops")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "forkwise")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "forkwise-secret-pw")

QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION_NAME = "ingredient_embeddings"

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Globals ---
model = None
qdrant = None


def get_pg_conn():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


app = FastAPI(title="ForkWise Substitution API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    global model, qdrant
    log.info(f"loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    log.info("model loaded")
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    log.info(f"connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")


# --- Request/Response Models ---

class SubstitutionRequest(BaseModel):
    ingredient: str
    recipe_name: str = ""
    top_k: int = 5


class Suggestion(BaseModel):
    ingredient: str
    recipe_name: str
    score: float


class SubstitutionResponse(BaseModel):
    query_id: str
    original: str
    suggestions: list[Suggestion]


class FeedbackRequest(BaseModel):
    query_id: str
    suggested_ingredient: str
    accepted: bool


# --- Endpoints ---

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/ready")
def ready():
    if model is None:
        raise HTTPException(503, "model not loaded")
    try:
        qdrant.get_collection(COLLECTION_NAME)
        return {"ready": True}
    except Exception as e:
        raise HTTPException(503, f"not ready: {e}")


@app.post("/substitute", response_model=SubstitutionResponse)
def substitute(body: SubstitutionRequest):
    # Build query text with optional recipe context
    if body.recipe_name:
        query_text = f"{body.ingredient} (in {body.recipe_name})"
    else:
        query_text = body.ingredient

    # Embed the query
    vector = model.encode(query_text).tolist()

    # Search Qdrant — fetch extra results so we can filter out the original ingredient
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=body.top_k + 10,
    )

    # Filter: exclude hits that are the same ingredient text
    original_lower = body.ingredient.lower().strip()
    suggestions = []
    seen = set()
    for hit in results:
        ing = hit.payload.get("ingredient", "")
        ing_clean = ing.lower().strip()

        # Skip the original ingredient itself
        if ing_clean == original_lower:
            continue
        # Skip duplicates
        if ing_clean in seen:
            continue
        seen.add(ing_clean)

        suggestions.append(Suggestion(
            ingredient=ing,
            recipe_name=hit.payload.get("recipe_name", ""),
            score=round(float(hit.score), 4),
        ))
        if len(suggestions) >= body.top_k:
            break

    # Log query + results to postgres
    query_id = str(uuid.uuid4())
    try:
        conn = get_pg_conn()
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO substitution_queries (query_id, recipe_id, original_ingredient, query_context)
                   VALUES (%s, %s, %s, %s)""",
                (query_id, None, body.ingredient, body.recipe_name or None),
            )
            for rank, s in enumerate(suggestions, 1):
                cur.execute(
                    """INSERT INTO substitution_results (query_id, suggested_ingredient, rank, score)
                       VALUES (%s, %s, %s, %s)""",
                    (query_id, s.ingredient, rank, s.score),
                )
            conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"failed to log query: {e}")

    log.info(f"query_id={query_id} ingredient={body.ingredient!r} recipe={body.recipe_name!r} results={len(suggestions)}")
    return SubstitutionResponse(
        query_id=query_id,
        original=body.ingredient,
        suggestions=suggestions,
    )


@app.post("/feedback")
def feedback(body: FeedbackRequest):
    try:
        conn = get_pg_conn()
        with conn.cursor() as cur:
            # Update accepted flag on the result
            cur.execute(
                """UPDATE substitution_results SET accepted = %s
                   WHERE query_id = %s::uuid AND suggested_ingredient = %s""",
                (1 if body.accepted else 0, body.query_id, body.suggested_ingredient),
            )
            # Log feedback event
            cur.execute(
                """INSERT INTO feedback_events (query_id, suggested_ingredient, event_type, rank)
                   SELECT %s::uuid, %s, %s, rank FROM substitution_results
                   WHERE query_id = %s::uuid AND suggested_ingredient = %s
                   LIMIT 1""",
                (body.query_id, body.suggested_ingredient,
                 "accept" if body.accepted else "reject",
                 body.query_id, body.suggested_ingredient),
            )
            conn.commit()
        conn.close()
        log.info(f"feedback query_id={body.query_id} ingredient={body.suggested_ingredient} accepted={body.accepted}")
        return {"ok": True}
    except Exception as e:
        log.error(f"feedback failed: {e}")
        raise HTTPException(500, f"feedback failed: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
