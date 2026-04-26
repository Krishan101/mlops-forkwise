"""
ForkWise Substitution API
Two interface modes:
  1. /substitute — standalone curl-friendly endpoint
  2. /predict + /feedback — Mealie-compatible endpoints (called by Mealie's SubstitutionService)
"""

import logging
import os
import uuid
import time

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
MODEL_VERSION = os.environ.get("MODEL_VERSION", "all-MiniLM-L6-v2-base")

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


def search_substitutions(ingredient_text: str, recipe_name: str, top_k: int) -> list[dict]:
    """Core search logic shared by both /substitute and /predict."""
    if recipe_name:
        query_text = f"{ingredient_text} (in {recipe_name})"
    else:
        query_text = ingredient_text

    vector = model.encode(query_text).tolist()

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k + 10,
    )

    original_lower = ingredient_text.lower().strip()
    suggestions = []
    seen = set()
    for hit in results:
        ing = hit.payload.get("ingredient", "")
        ing_clean = ing.lower().strip()
        if ing_clean == original_lower:
            continue
        if ing_clean in seen:
            continue
        seen.add(ing_clean)
        suggestions.append({
            "ingredient": ing,
            "recipe_name": hit.payload.get("recipe_name", ""),
            "score": round(float(hit.score), 4),
        })
        if len(suggestions) >= top_k:
            break

    return suggestions


def log_query_to_db(query_id: str, recipe_id: str | None, ingredient: str,
                    recipe_name: str | None, suggestions: list[dict]):
    """Log query + results to postgres for feedback loop."""
    try:
        conn = get_pg_conn()
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO substitution_queries (query_id, recipe_id, original_ingredient, query_context)
                   VALUES (%s, %s, %s, %s)""",
                (query_id, recipe_id, ingredient, recipe_name),
            )
            for rank, s in enumerate(suggestions, 1):
                cur.execute(
                    """INSERT INTO substitution_results (query_id, suggested_ingredient, rank, score)
                       VALUES (%s, %s, %s, %s)""",
                    (query_id, s["ingredient"], rank, s["score"]),
                )
            conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"failed to log query: {e}")


# =========================================================================
# Standalone endpoint (curl-friendly)
# =========================================================================

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
    suggestions = search_substitutions(body.ingredient, body.recipe_name, body.top_k)
    query_id = str(uuid.uuid4())
    log_query_to_db(query_id, None, body.ingredient, body.recipe_name or None, suggestions)
    log.info(f"query_id={query_id} ingredient={body.ingredient!r} recipe={body.recipe_name!r} results={len(suggestions)}")
    return SubstitutionResponse(
        query_id=query_id,
        original=body.ingredient,
        suggestions=[Suggestion(**s) for s in suggestions],
    )


# =========================================================================
# Mealie-compatible endpoints
# =========================================================================

class MealieIngredient(BaseModel):
    raw: str
    normalized: str


class MealiePredictRequest(BaseModel):
    recipe_id: str
    ingredients: list[MealieIngredient] = []
    missing_ingredient: MealieIngredient
    request_id: str | None = None
    top_k: int = 3
    recipe_title: str | None = None
    instructions: list[str] | None = None
    timestamp: str | None = None


class MealiePrediction(BaseModel):
    ingredient: str
    rank: int
    embedding_score: float


class MealiePredictResponse(BaseModel):
    recipe_id: str
    missing_ingredient: str
    request_id: str | None = None
    substitutions: list[MealiePrediction] = []
    model_version: str | None = None
    serving_version: str | None = None
    latency_ms: int | None = None


class MealieFeedbackRequest(BaseModel):
    request_id: str
    recipe_id: str
    missing_ingredient: str
    suggested_substitution: str
    user_accepted: bool
    model_version: str | None = None


class MealieFeedbackResponse(BaseModel):
    status: str
    key: str | None = None


@app.post("/predict", response_model=MealiePredictResponse)
def predict(body: MealiePredictRequest):
    start = time.time()

    ingredient_text = body.missing_ingredient.normalized or body.missing_ingredient.raw
    recipe_name = body.recipe_title or ""

    suggestions = search_substitutions(ingredient_text, recipe_name, body.top_k)

    request_id = body.request_id or f"req_{uuid.uuid4().hex}"
    log_query_to_db(request_id, body.recipe_id, ingredient_text, recipe_name, suggestions)

    latency = int((time.time() - start) * 1000)

    log.info(f"[mealie] request_id={request_id} ingredient={ingredient_text!r} recipe={recipe_name!r} results={len(suggestions)} latency={latency}ms")

    return MealiePredictResponse(
        recipe_id=body.recipe_id,
        missing_ingredient=ingredient_text,
        request_id=request_id,
        substitutions=[
            MealiePrediction(ingredient=s["ingredient"], rank=i + 1, embedding_score=s["score"])
            for i, s in enumerate(suggestions)
        ],
        model_version=MODEL_VERSION,
        serving_version="0.1.0",
        latency_ms=latency,
    )


@app.post("/feedback", response_model=MealieFeedbackResponse)
def feedback(body: MealieFeedbackRequest):
    try:
        conn = get_pg_conn()
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE substitution_results SET accepted = %s
                   WHERE query_id = %s::uuid AND suggested_ingredient = %s""",
                (1 if body.user_accepted else 0, body.request_id, body.suggested_substitution),
            )
            cur.execute(
                """INSERT INTO feedback_events (query_id, suggested_ingredient, event_type, rank)
                   SELECT %s::uuid, %s, %s, rank FROM substitution_results
                   WHERE query_id = %s::uuid AND suggested_ingredient = %s
                   LIMIT 1""",
                (body.request_id, body.suggested_substitution,
                 "accept" if body.user_accepted else "reject",
                 body.request_id, body.suggested_substitution),
            )
            conn.commit()
        conn.close()
        key = f"{body.request_id}:{body.suggested_substitution}"
        log.info(f"[mealie] feedback request_id={body.request_id} ingredient={body.suggested_substitution} accepted={body.user_accepted}")
        return MealieFeedbackResponse(status="logged", key=key)
    except Exception as e:
        log.error(f"feedback failed: {e}")
        raise HTTPException(500, f"feedback failed: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
