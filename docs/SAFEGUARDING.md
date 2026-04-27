# ForkWise Safeguarding Plan

## Overview

ForkWise is an ML-powered ingredient substitution system integrated into Mealie, a self-hosted recipe manager. Users receive substitution suggestions when they click "Suggest Substitute" on any ingredient in a recipe. This document describes the concrete mechanisms we implement to address fairness, explainability, transparency, privacy, accountability, and robustness.

---

## 1. Fairness

**Risk:** The GISMo model is trained on Recipe1MSubs, which is derived from Recipe1M — a dataset scraped from predominantly US-centric food websites. Substitution suggestions may be biased toward Western cuisines and may not work equally well for ingredients common in Asian, African, Middle Eastern, or Latin American cooking.

**Mechanisms implemented:**

- **Diverse evaluation:** We evaluate model performance using stratified test splits — in-distribution (ID) and out-of-distribution (OOD) — to measure how well the model generalizes to unseen ingredient pairs. OOD performance is tracked separately in MLflow so we can detect if the model disproportionately fails on less-common ingredients.

- **Context-aware scoring:** GISMo uses the full recipe context (all ingredients in the recipe) when scoring substitutions, rather than treating ingredients in isolation. This helps the model respect cuisine-specific patterns rather than defaulting to Western substitution norms.

- **Feedback loop for correction:** Users can reject suggestions and those rejections are stored. If the model consistently suggests inappropriate substitutions for certain ingredient categories, the feedback data used in retraining will correct this bias over time.

- **Known limitation:** We acknowledge in the README and in the model metadata that the training data has a Western cuisine bias. Users should treat suggestions as starting points, not authoritative recommendations.

---

## 2. Explainability

**Risk:** Users receive ranked substitution suggestions with numerical scores but may not understand why a particular ingredient was suggested.

**Mechanisms implemented:**

- **Score transparency:** Every substitution response includes a numerical score from the GISMo model. Higher scores indicate the model considers the substitution more plausible in the given recipe context. The `/substitute` endpoint returns both the GISMo score and the original Qdrant similarity score, so the two-stage ranking is visible.

- **Model info endpoint:** The `/admin/model-info` endpoint exposes which model version is serving, its training metrics (MRR, Hit@k), and when it was trained. This allows operators to understand the model's expected quality level.

- **Recipe context in scoring:** The substitution API accepts and uses `recipe_ingredients` as context, making it clear that suggestions depend on what else is in the recipe — not just the ingredient being replaced.

- **MLflow tracking:** All training runs are logged with full hyperparameters, per-epoch metrics, and test evaluation results. Any deployed model can be traced back to its training run, dataset, and evaluation scores.

---

## 3. Transparency

**Risk:** Users may not know that an ML model is making the substitution suggestions, or how their feedback is used.

**Mechanisms implemented:**

- **Visible ML integration:** The "Suggest Substitute" button in Mealie clearly labels the feature. Suggestions are shown with numerical confidence scores so users understand these are model-generated, not curated by humans.

- **Model version in responses:** Every API response from `/predict` and `/substitute` includes a `model` field ("gismo+qdrant" or "qdrant-only") and `model_version`, making it clear which system generated the suggestion.

- **Open source:** The full system code, training scripts, model architecture, and deployment manifests are in a public GitHub repository. The GISMo architecture follows the published paper (Fatemi et al., 2023) with clear attribution.

- **Training data provenance:** The Recipe1MSubs dataset and FlavorGraph are documented in the repository. The merge dictionary and vocabulary mapping are versioned and stored in the Chameleon object store alongside the model artifacts.

---

## 4. Privacy

**Risk:** The system processes user recipes, ingredient preferences, and feedback, which could reveal dietary restrictions, allergies, or health conditions.

**Mechanisms implemented:**

- **Self-hosted architecture:** Mealie and all ForkWise services run on the user's own Chameleon Cloud infrastructure. No data is sent to external services for inference. The GISMo ONNX model runs locally within the Kubernetes cluster.

- **No personal data in training:** The GISMo model is trained on Recipe1MSubs (publicly available recipe data from Recipe1M). User feedback is stored in the platform Postgres database on the cluster, not sent externally. Retraining uses feedback tuples (ingredient pairs) without any user identifiers.

- **Minimal data collection:** The feedback system stores only the ingredient substitution pair and accept/reject status. It does not store user identity, IP address, or session information. The `recipe_id` in feedback refers to the Mealie recipe, not a user.

- **Credential management:** All secrets (database passwords, S3 keys, GHCR tokens) are stored as Kubernetes Secrets, not hardcoded in application code. The `clouds.yaml` for OpenStack credentials is in `.gitignore`.

---

## 5. Accountability

**Risk:** If the model suggests a harmful substitution (e.g., suggesting a common allergen as a substitute), there must be mechanisms to trace and correct the issue.

**Mechanisms implemented:**

- **Full audit trail:** Every substitution query is logged to the `substitution_queries` table with a unique `query_id`, timestamp, the original ingredient, and recipe context. Every suggestion is logged to `substitution_results` with its rank and score. Every feedback event is logged to `feedback_events` with the event type and timestamp. This creates a complete audit trail from query to suggestion to user response.

- **Model versioning and rollback:** The object store maintains both the current model and the previous model (`_previous` suffix). The `/admin/rollback` endpoint can instantly swap to the previous model if issues are detected. The retrain pipeline backs up the current model before promoting a new one.

- **Quality gates:** The retraining pipeline enforces two quality gates before deploying any model:
  1. Absolute gate: test MRR must be ≥ 10.0 (prevents deploying a broken model)
  2. Relative gate: test MRR must be ≥ 95% of the current model's MRR (prevents significant regression)

- **MLflow lineage:** Every deployed model can be traced back to its MLflow run, which records the exact training data, hyperparameters, and evaluation metrics. This enables root cause analysis if a model produces poor suggestions.

- **Monitoring alerts:** Grafana alerts fire when the error rate exceeds 10% or p95 latency exceeds 1 second, signaling operators to investigate.

---

## 6. Robustness

**Risk:** The system must handle edge cases gracefully — unknown ingredients, malformed inputs, model loading failures, and infrastructure issues.

**Mechanisms implemented:**

- **Graceful fallback:** If the GISMo ONNX model fails to load at startup, the substitution API falls back to Qdrant-only mode (sentence-transformer cosine similarity). The `GISMO_ENABLED` flag and the fallback logic ensure the system never becomes completely non-functional due to model issues.

- **Input normalization:** The `GISMoScorer.get_ingredient_idx()` method tries multiple normalizations (lowercase, underscore replacement, stripping quantities and units) to match user-provided ingredient names to the FlavorGraph vocabulary. Unrecognized ingredients fall back to Qdrant similarity.

- **Liveness and readiness probes:** The substitution API has separate `/health` (liveness) and `/ready` (readiness) endpoints. Kubernetes will restart the pod if liveness fails and will stop routing traffic if readiness fails.

- **Backup and restore:** The `backup.sh` and `bring_up.sh` scripts implement full state backup to Chameleon object storage (Postgres dumps, Qdrant snapshots, Mealie data) and automatic restoration on fresh deployments.

- **Retraining safety:** The CronJob-based retraining pipeline includes a feedback count threshold (minimum 50 events) to avoid retraining on insufficient data, and the quality gates prevent deploying a degraded model.

- **Infrastructure monitoring:** Prometheus monitors node CPU, memory, disk, and pod resource usage. Pod restart counts are tracked in Grafana with alert thresholds.
