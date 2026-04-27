"""
ForkWise Substitution API (v2 - with GISMo reranking)
=====================================================
Two-stage architecture:
  1. Retrieve: sentence-transformer + Qdrant → top-50 candidates (fast approximate)
  2. Rerank: GISMo ONNX decoder scores candidates using precomputed graph embeddings

Falls back to Qdrant-only mode if GISMo model is not available.

Interface modes:
  1. /substitute — standalone curl-friendly endpoint
  2. /predict + /feedback — Mealie-compatible endpoints
  3. /admin/model-info — shows current model status
  4. /admin/rollback — swaps to previous model
"""

import logging
import os
import json
import uuid
import time

import numpy as np
import psycopg2
import psycopg2.extras
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter
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

# GISMo config
GISMO_ENABLED = os.environ.get("GISMO_ENABLED", "false").lower() == "true"
GISMO_MODEL_DIR = os.environ.get("GISMO_MODEL_DIR", "/app/gismo_model")
GISMO_RERANK_CANDIDATES = int(os.environ.get("GISMO_RERANK_CANDIDATES", "50"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "current")  # "current" or "previous"

# --- Globals ---
sentence_model = None
qdrant = None
gismo_scorer = None

# --- Custom Prometheus metrics for drift detection ---
GISMO_SCORE_HISTOGRAM = Histogram(
    'forkwise_gismo_top_score',
    'Distribution of top GISMo reranking score per query',
    buckets=[-5, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, 10]
)

UNKNOWN_INGREDIENT_COUNTER = Counter(
    'forkwise_unknown_ingredient_total',
    'Number of query ingredients not found in GISMo vocabulary'
)

KNOWN_INGREDIENT_COUNTER = Counter(
    'forkwise_known_ingredient_total',
    'Number of query ingredients found in GISMo vocabulary'
)

QDRANT_FALLBACK_COUNTER = Counter(
    'forkwise_qdrant_fallback_total',
    'Queries where GISMo scoring failed and fell back to Qdrant-only'
)

FEEDBACK_ACCEPT_COUNTER = Counter(
    'forkwise_feedback_accept_total',
    'Total accepted substitution suggestions'
)

FEEDBACK_REJECT_COUNTER = Counter(
    'forkwise_feedback_reject_total',
    'Total rejected substitution suggestions'
)


# =========================================================================
# GISMo ONNX Scorer
# =========================================================================

class GISMoScorer:
    """Loads ONNX decoder + precomputed embeddings for reranking."""

    def __init__(self, model_dir, version="current"):
        self.model_dir = model_dir
        self.version = version
        self.session = None
        self.embeddings = None
        self.vocab = None
        self.vocab_reverse = None  # idx -> name
        self.metadata = None
        self.ready = False

        self._load(version)

    def _model_path(self, version, filename):
        if version == "current":
            return os.path.join(self.model_dir, filename)
        else:
            # Previous version has _previous suffix
            base, ext = os.path.splitext(filename)
            prev_path = os.path.join(self.model_dir, f"{base}_previous{ext}")
            if os.path.exists(prev_path):
                return prev_path
            return os.path.join(self.model_dir, filename)

    def _load(self, version):
        try:
            import onnxruntime as ort

            onnx_path = self._model_path(version, "gismo_decoder.onnx")
            emb_path = self._model_path(version, "ingredient_embeddings.npy")
            vocab_path = self._model_path(version, "vocab.json")
            meta_path = self._model_path(version, "metadata.json")

            if not os.path.exists(onnx_path):
                log.warning(f"GISMo ONNX model not found at {onnx_path}")
                return

            # Load ONNX session
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                onnx_path, sess_options=sess_opts,
                providers=['CPUExecutionProvider']
            )

            # Load precomputed embeddings
            self.embeddings = np.load(emb_path).astype(np.float32)

            # Load vocab
            with open(vocab_path) as f:
                self.vocab = json.load(f)
            self.vocab_reverse = {v: k for k, v in self.vocab.items()}

            # Load metadata
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    self.metadata = json.load(f)

            self.ready = True
            log.info(f"GISMo scorer loaded: {onnx_path} "
                     f"({self.embeddings.shape[0]} nodes, dim={self.embeddings.shape[1]})")
            if self.metadata:
                log.info(f"  Model test_mrr={self.metadata.get('test_mrr', '?')}, "
                         f"trained={self.metadata.get('timestamp', '?')}")

        except Exception as e:
            log.error(f"Failed to load GISMo scorer: {e}")
            self.ready = False

    def get_ingredient_idx(self, ingredient_name):
        """Look up ingredient index from name. Tries several normalizations."""
        name = ingredient_name.lower().strip()

        # Direct lookup
        if name in self.vocab:
            return self.vocab[name]

        # Replace spaces with underscores
        name_underscore = name.replace(' ', '_')
        if name_underscore in self.vocab:
            return self.vocab[name_underscore]

        # Try without common prefixes/quantities
        # Strip leading numbers and units
        import re
        cleaned = re.sub(r'^[\d\s/½¼¾⅓⅔]+', '', name).strip()
        cleaned = re.sub(r'^(cups?|tablespoons?|teaspoons?|tbsp|tsp|lb|oz|ounces?|pounds?|cloves?|sliced|minced|chopped|diced|fresh|large|small|medium)\s+', '', cleaned, flags=re.IGNORECASE).strip()
        cleaned_underscore = cleaned.replace(' ', '_')
        if cleaned_underscore in self.vocab:
            return self.vocab[cleaned_underscore]

        return None

    def score_candidates(self, source_name, candidate_names, recipe_ingredient_names):
        """
        Score candidates using the ONNX decoder.

        Args:
            source_name: name of ingredient to substitute
            candidate_names: list of candidate ingredient names
            recipe_ingredient_names: list of all ingredient names in the recipe (for context)

        Returns:
            list of scores (same order as candidate_names), or None if scoring fails
        """
        if not self.ready:
            return None

        source_idx = self.get_ingredient_idx(source_name)
        if source_idx is None:
            return None

        source_emb = self.embeddings[source_idx]

        # Compute context embedding (average of recipe ingredients)
        ctx_embs = []
        for ing_name in recipe_ingredient_names:
            idx = self.get_ingredient_idx(ing_name)
            if idx is not None:
                ctx_embs.append(self.embeddings[idx])
        if ctx_embs:
            context_emb = np.mean(ctx_embs, axis=0)
        else:
            context_emb = np.zeros(self.embeddings.shape[1], dtype=np.float32)

        # Look up candidate embeddings
        valid_indices = []
        valid_positions = []
        for i, cand_name in enumerate(candidate_names):
            idx = self.get_ingredient_idx(cand_name)
            if idx is not None:
                valid_indices.append(idx)
                valid_positions.append(i)

        if not valid_indices:
            return None

        candidate_embs = self.embeddings[np.array(valid_indices)]

        # Tile source and context to match candidates
        n = len(valid_indices)
        source_tiled = np.tile(source_emb, (n, 1))
        context_tiled = np.tile(context_emb, (n, 1))

        # Run ONNX inference
        try:
            scores = self.session.run(None, {
                'source_emb': source_tiled,
                'candidate_emb': candidate_embs,
                'context_emb': context_tiled,
            })[0]  # shape: [n]
        except Exception as e:
            log.warning(f"ONNX inference failed: {e}")
            return None

        # Map scores back to original positions
        all_scores = [None] * len(candidate_names)
        for pos, score in zip(valid_positions, scores.flatten()):
            all_scores[pos] = float(score)

        return all_scores

    def rollback(self):
        """Swap current and previous models."""
        import shutil
        files = ["gismo_decoder.onnx", "ingredient_embeddings.npy", "vocab.json", "metadata.json"]

        for f in files:
            current = os.path.join(self.model_dir, f)
            base, ext = os.path.splitext(f)
            previous = os.path.join(self.model_dir, f"{base}_previous{ext}")

            if os.path.exists(current) and os.path.exists(previous):
                temp = current + ".tmp"
                shutil.move(current, temp)
                shutil.move(previous, current)
                shutil.move(temp, previous)

        # Reload
        self._load("current")
        return self.ready


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

# Prometheus metrics endpoint at /metrics
Instrumentator().instrument(app).expose(app)


@app.on_event("startup")
def startup():
    global sentence_model, qdrant, gismo_scorer
    log.info(f"loading sentence model: {MODEL_NAME}")
    sentence_model = SentenceTransformer(MODEL_NAME)
    log.info("sentence model loaded")
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    log.info(f"connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")

    if GISMO_ENABLED:
        log.info(f"loading GISMo scorer from {GISMO_MODEL_DIR} (version={MODEL_VERSION})")
        gismo_scorer = GISMoScorer(GISMO_MODEL_DIR, MODEL_VERSION)
        if gismo_scorer.ready:
            log.info("GISMo reranking ENABLED")
        else:
            log.warning("GISMo model not found or failed to load — falling back to Qdrant-only mode")
    else:
        log.info("GISMo reranking DISABLED (set GISMO_ENABLED=true to enable)")


def search_substitutions(ingredient_text: str, recipe_name: str, top_k: int,
                         recipe_ingredients: list[str] = None) -> list[dict]:
    """
    Two-stage retrieve-then-rerank:
    1. Qdrant retrieval: get top candidates by sentence-transformer similarity
    2. GISMo rerank: score candidates with the trained substitution model
    """
    # Stage 1: Retrieve from Qdrant
    if recipe_name:
        query_text = f"{ingredient_text} (in {recipe_name})"
    else:
        query_text = ingredient_text

    vector = sentence_model.encode(query_text).tolist()

    # Get more candidates than needed if GISMo will rerank
    retrieve_k = GISMO_RERANK_CANDIDATES if (gismo_scorer and gismo_scorer.ready) else top_k + 10

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=retrieve_k + 10,
    )

    original_lower = ingredient_text.lower().strip()
    candidates = []
    seen = set()
    for hit in results:
        ing = hit.payload.get("ingredient", "")
        ing_clean = ing.lower().strip()
        if ing_clean == original_lower:
            continue
        if ing_clean in seen:
            continue
        seen.add(ing_clean)
        candidates.append({
            "ingredient": ing,
            "recipe_name": hit.payload.get("recipe_name", ""),
            "qdrant_score": round(float(hit.score), 4),
        })
        if len(candidates) >= retrieve_k:
            break

    # Stage 2: Rerank with GISMo (if available)
    if gismo_scorer and gismo_scorer.ready and candidates:
        candidate_names = [c["ingredient"] for c in candidates]
        context_ingredients = recipe_ingredients or []

        # Track whether query ingredient is in GISMo vocabulary
        source_idx = gismo_scorer.get_ingredient_idx(ingredient_text)
        if source_idx is not None:
            KNOWN_INGREDIENT_COUNTER.inc()
        else:
            UNKNOWN_INGREDIENT_COUNTER.inc()

        gismo_scores = gismo_scorer.score_candidates(
            ingredient_text, candidate_names, context_ingredients
        )

        if gismo_scores is not None:
            # Combine scores: use GISMo score as primary, Qdrant as tiebreaker
            for i, c in enumerate(candidates):
                if gismo_scores[i] is not None:
                    c["gismo_score"] = round(gismo_scores[i], 4)
                    c["score"] = round(gismo_scores[i], 4)
                else:
                    c["gismo_score"] = None
                    c["score"] = c["qdrant_score"]

            # Sort by GISMo score descending (None scores go to the end)
            candidates.sort(key=lambda x: (x["gismo_score"] is not None, x.get("gismo_score", -999)), reverse=True)

            # Record top score for drift detection
            top_gismo = [c.get("gismo_score") for c in candidates if c.get("gismo_score") is not None]
            if top_gismo:
                GISMO_SCORE_HISTOGRAM.observe(top_gismo[0])

            log.info(f"  GISMo reranked {len(candidates)} candidates")
        else:
            # GISMo scoring failed for this query, use Qdrant scores
            QDRANT_FALLBACK_COUNTER.inc()
            for c in candidates:
                c["score"] = c["qdrant_score"]
    else:
        for c in candidates:
            c["score"] = c["qdrant_score"]

    # Return top_k
    suggestions = []
    for c in candidates[:top_k]:
        suggestions.append({
            "ingredient": c["ingredient"],
            "recipe_name": c["recipe_name"],
            "score": c["score"],
        })

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
    recipe_ingredients: list[str] = []
    top_k: int = 5


class Suggestion(BaseModel):
    ingredient: str
    recipe_name: str
    score: float


class SubstitutionResponse(BaseModel):
    query_id: str
    original: str
    suggestions: list[Suggestion]
    model: str = "qdrant-only"


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/ready")
def ready():
    if sentence_model is None:
        raise HTTPException(503, "model not loaded")
    try:
        qdrant.get_collection(COLLECTION_NAME)
        return {"ready": True, "gismo_enabled": gismo_scorer.ready if gismo_scorer else False}
    except Exception as e:
        raise HTTPException(503, f"not ready: {e}")


@app.get("/admin/model-info")
def model_info():
    info = {
        "sentence_model": MODEL_NAME,
        "gismo_enabled": GISMO_ENABLED,
        "gismo_ready": gismo_scorer.ready if gismo_scorer else False,
        "gismo_metadata": gismo_scorer.metadata if (gismo_scorer and gismo_scorer.ready) else None,
        "model_version": MODEL_VERSION,
    }
    return info


@app.post("/admin/rollback")
def rollback():
    if not gismo_scorer:
        raise HTTPException(400, "GISMo not enabled")
    success = gismo_scorer.rollback()
    if success:
        return {"status": "rolled back", "metadata": gismo_scorer.metadata}
    else:
        raise HTTPException(500, "Rollback failed")


@app.post("/substitute", response_model=SubstitutionResponse)
def substitute(body: SubstitutionRequest):
    suggestions = search_substitutions(
        body.ingredient, body.recipe_name, body.top_k, body.recipe_ingredients
    )
    query_id = str(uuid.uuid4())
    log_query_to_db(query_id, None, body.ingredient, body.recipe_name or None, suggestions)

    model_type = "gismo+qdrant" if (gismo_scorer and gismo_scorer.ready) else "qdrant-only"
    log.info(f"query_id={query_id} ingredient={body.ingredient!r} recipe={body.recipe_name!r} "
             f"results={len(suggestions)} model={model_type}")

    return SubstitutionResponse(
        query_id=query_id,
        original=body.ingredient,
        suggestions=[Suggestion(**s) for s in suggestions],
        model=model_type,
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

    # Extract ingredient names for GISMo context
    recipe_ingredient_names = [ing.normalized or ing.raw for ing in body.ingredients]

    suggestions = search_substitutions(
        ingredient_text, recipe_name, body.top_k, recipe_ingredient_names
    )

    request_id = body.request_id or str(uuid.uuid4())
    log_query_to_db(request_id, body.recipe_id, ingredient_text, recipe_name, suggestions)

    latency = int((time.time() - start) * 1000)
    model_type = "gismo+qdrant" if (gismo_scorer and gismo_scorer.ready) else "qdrant-only"

    log.info(f"[mealie] request_id={request_id} ingredient={ingredient_text!r} "
             f"recipe={recipe_name!r} results={len(suggestions)} latency={latency}ms model={model_type}")

    return MealiePredictResponse(
        recipe_id=body.recipe_id,
        missing_ingredient=ingredient_text,
        request_id=request_id,
        substitutions=[
            MealiePrediction(ingredient=s["ingredient"], rank=i + 1, embedding_score=s["score"])
            for i, s in enumerate(suggestions)
        ],
        model_version=model_type,
        serving_version="0.2.0",
        latency_ms=latency,
    )


@app.post("/feedback", response_model=MealieFeedbackResponse)
def feedback(body: MealieFeedbackRequest):
    try:
        conn = get_pg_conn()
        with conn.cursor() as cur:
            # Normalize request_id: strip 'req_' prefix and format as UUID if needed
            rid = body.request_id
            if rid.startswith("req_"):
                hex_str = rid[4:]
                rid = f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"
            cur.execute(
                """UPDATE substitution_results SET accepted = %s
                   WHERE query_id = %s::uuid AND suggested_ingredient = %s""",
                (1 if body.user_accepted else 0, rid, body.suggested_substitution),
            )
            cur.execute(
                """INSERT INTO feedback_events (query_id, suggested_ingredient, event_type, rank)
                   SELECT %s::uuid, %s, %s, rank FROM substitution_results
                   WHERE query_id = %s::uuid AND suggested_ingredient = %s
                   LIMIT 1""",
                (rid, body.suggested_substitution,
                 "accept" if body.user_accepted else "reject",
                 rid, body.suggested_substitution),
            )
            conn.commit()
        conn.close()
        key = f"{body.request_id}:{body.suggested_substitution}"
        # Track accept/reject for drift monitoring
        if body.user_accepted:
            FEEDBACK_ACCEPT_COUNTER.inc()
        else:
            FEEDBACK_REJECT_COUNTER.inc()
        log.info(f"[mealie] feedback request_id={body.request_id} "
                 f"ingredient={body.suggested_substitution} accepted={body.user_accepted}")
        return MealieFeedbackResponse(status="logged", key=key)
    except Exception as e:
        log.error(f"feedback failed: {e}")
        raise HTTPException(500, f"feedback failed: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
