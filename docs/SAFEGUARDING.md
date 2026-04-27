# ForkWise Safeguarding Plan

## Overview

ForkWise is an ML-powered ingredient substitution system integrated into Mealie, a self-hosted recipe manager. Users receive substitution suggestions when they click "Suggest Substitute" on any ingredient in a recipe. This document describes the concrete mechanisms we implement to address fairness, explainability, transparency, privacy, accountability, and robustness.

---

## 1. Fairness

**Risk:** The GISMo model is trained on Recipe1MSubs, which is derived from Recipe1M — a dataset scraped from predominantly US-centric food websites. Substitution suggestions may be biased toward Western cuisines and may not work equally well for ingredients common in Asian, African, Middle Eastern, or Latin American cooking.

**Mechanisms implemented:**

- **Diverse evaluation:** We evaluate model performance using MRR and Hit@k on held-out test data from Recipe1MSubs. While the test set shares the same Western bias as the training data, we track these metrics across retraining runs in MLflow to detect if model quality degrades for any category of ingredients.

- **Context-aware scoring:** GISMo uses the full recipe context (all ingredients in the recipe) when scoring substitutions, rather than treating ingredients in isolation. This helps the model respect cuisine-specific patterns rather than defaulting to Western substitution norms.

- **Feedback loop for correction:** Users can reject suggestions and those rejections are stored. The retraining pipeline incorporates accepted feedback as positive training examples. Over time, this adapts the model to the actual user base's ingredient preferences.

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
  1. **Absolute gate:** test MRR must be ≥ 10.0
  2. **Relative gate:** test MRR must be ≥ 95% of the current model's MRR

- **Canary validation:** After a new model is deployed, the pipeline sends 20 test queries and verifies at least 85% succeed. If the canary fails, the pipeline automatically rolls back to the previous model without human intervention.

- **MLflow lineage:** Every deployed model can be traced back to its MLflow run, which records the exact training data, hyperparameters, and evaluation metrics. This enables root cause analysis if a model produces poor suggestions.

- **Monitoring alerts:** Grafana alerts fire when the error rate exceeds 10% or p95 latency exceeds 1 second, signaling operators to investigate.

---

## 6. Robustness

**Risk:** The system must handle edge cases gracefully — unknown ingredients, malformed inputs, model loading failures, and infrastructure issues.

**Mechanisms implemented:**

- **Graceful fallback:** If the GISMo ONNX model fails to load at startup, the substitution API falls back to Qdrant-only mode (sentence-transformer cosine similarity). The `GISMO_ENABLED` flag and the fallback logic ensure the system never becomes completely non-functional due to model issues.

- **Input normalization:** The `GISMoScorer.get_ingredient_idx()` method tries multiple normalizations (lowercase, underscore replacement, stripping quantities and units) to match user-provided ingredient names to the FlavorGraph vocabulary. Unrecognized ingredients fall back to Qdrant similarity.

- **Liveness and readiness probes:** The substitution API has separate `/health` (liveness) and `/ready` (readiness) endpoints. Kubernetes will restart the pod if liveness fails and will stop routing traffic if readiness fails.

- **Horizontal Pod Autoscaler:** The substitution API scales from 1 to 3 replicas when CPU usage exceeds 70%, ensuring the system handles traffic spikes without manual intervention.

- **Backup and restore:** The `backup.sh` and `bring_up.sh` scripts implement full state backup to Chameleon object storage (Postgres dumps, Qdrant snapshots, Mealie data) and automatic restoration on fresh deployments.

- **Retraining safety:** The CronJob-based retraining pipeline includes a feedback count threshold (minimum 50 events) to avoid retraining on insufficient data, and the quality gates prevent deploying a degraded model.

- **Infrastructure monitoring:** Prometheus monitors node CPU, memory, disk, and pod resource usage. Pod restart counts are tracked in Grafana with alert thresholds.

---

## 7. Threshold Justifications

All numeric thresholds used in quality gates, alerts, and scaling are explicitly justified:

### Model Quality Thresholds

| Threshold | Value | Justification |
|-----------|-------|---------------|
| Absolute MRR gate | ≥ 10.0 | Random ranking on 6,653 ingredients gives MRR ~0.015%. MRR 10.0 means the correct substitution typically appears in the top 10 results. Since the UI shows 3-5 suggestions, a model below MRR 10 would rarely show useful results. |
| Relative MRR gate | ≥ 95% of current | Allows minor regression (up to 5%) when retraining with feedback data, since feedback introduces distribution shift. A drop beyond 5% indicates the feedback data is degrading the model. |
| Feedback threshold | ≥ 50 events | Below 50 feedback events, the signal-to-noise ratio is too low for meaningful retraining. 50 events typically yield 15-20 positive training tuples after filtering rejects and contradictions. |
| Canary test queries | 20 queries, ≤ 3 failures | Tests basic model serving functionality. Allowing 3/20 failures (15%) accounts for transient network issues during pod startup. More than 3 failures indicates a systematic problem. |

### Serving Alert Thresholds

| Threshold | Value | Justification |
|-----------|-------|---------------|
| Error rate alert | > 10% for 5 min | Under normal operation, the API returns 0% errors. An error rate above 10% means more than 1 in 10 user requests fail, significantly degrading user experience. The 5-minute window avoids false alarms from brief network blips. |
| Latency alert (p95) | > 1 second for 5 min | Normal p95 latency is 300-500ms (ONNX inference + Qdrant search). 1 second indicates resource contention, a stuck model load, or infrastructure issues. The 5-minute window filters transient spikes. |
| Pod restart threshold | > 3 in 24h (yellow), > 10 (red) | Occasional restarts (1-2/day) can happen due to OOM or transient issues. More than 3 suggests a recurring problem; more than 10 indicates a critical stability issue. |

### Scaling Thresholds

| Threshold | Value | Justification |
|-----------|-------|---------------|
| HPA CPU target | 70% average utilization | Below 70%, a single replica handles the load efficiently. Above 70%, response times start to degrade as the CPU is context-switching. Scaling at 70% provides headroom before latency is impacted. |
| HPA min/max replicas | 1-3 | Minimum 1 to conserve resources during low traffic. Maximum 3 because the cluster has 3 nodes with limited memory (8GB each), and each substitution API pod uses ~1GB. |

---

## 8. Multi-Environment Deployment Strategy

We implement a **logical multi-environment flow** rather than physically separate namespaces. This is a deliberate architecture decision justified by our infrastructure constraints (3 VMs with 8GB RAM each cannot run 3 copies of the 1GB+ substitution API).

### Environment Mapping

| Logical Environment | Physical Implementation | Validation |
|---------------------|------------------------|------------|
| **Staging** | Offline evaluation during training (Step 7 in retrain pipeline) | MRR/Hit@k computed on held-out test set. Both absolute and relative quality gates must pass. |
| **Canary** | Time-boxed live validation (Step 9b in retrain pipeline) | 20 test queries sent to the newly deployed model. If >3 fail, automatic rollback to previous model. |
| **Production** | Running deployment after canary passes (Step 10 onward) | Continuous monitoring via Prometheus + Grafana with alert rules. |

### Promotion Flow

```
Training complete
  → Staging: test MRR ≥ 10.0 AND ≥ 95% of current model
    → FAIL: model rejected, current model stays
    → PASS: upload to S3, restart API
      → Canary: 20 test queries, ≤ 3 failures allowed
        → FAIL: auto-rollback to previous model
        → PASS: mark feedback consumed, model is live in production
          → Production monitoring: Grafana alerts for error rate, latency
            → DEGRADATION: operator calls /admin/rollback
```

This flow provides equivalent safety guarantees to physical environment separation: no model reaches production without passing offline evaluation (staging) AND live traffic validation (canary), with automated rollback if either fails.

---

## 9. Data Quality at Three Stages

### Stage 1: Ingestion

The ingest service (`services/ingest-api/app.py`) validates data at ingestion from Mealie:
- Rejects recipes with empty names
- Validates each ingredient: rejects empty, too short (<2 chars), and too long (>500 chars) entries
- Rejects recipes with zero valid ingredients
- Logs quality metrics per poll cycle: `quality: new=X rejected=Y ingredients_accepted=Z`

### Stage 2: Training Data Construction

The retrain pipeline (`scripts/retrain_pipeline.sh` Step 2) validates feedback before including it in training data:
- Only includes accepted feedback (rejects are excluded)
- Deduplicates: same (original, replacement) pair included only once
- Contradiction detection: if the same pair was both accepted AND rejected, it is excluded from training data
- Logs: `X positive tuples (deduplicated from Y accepts, removed Z contradictions)`

### Stage 3: Production Monitoring (Drift Detection)

Custom Prometheus metrics track distribution changes in production:
- `forkwise_gismo_top_score` histogram: tracks GISMo score distribution per query. Shifting distribution indicates model drift.
- `forkwise_unknown_ingredient_total` / `forkwise_known_ingredient_total`: tracks vocabulary coverage. Rising unknown rate indicates users are cooking with ingredients outside the training data.
- `forkwise_qdrant_fallback_total`: counts queries where GISMo scoring failed. Rising rate indicates model or data issues.
- `forkwise_feedback_accept_total` / `forkwise_feedback_reject_total`: tracks accept/reject ratio. Declining acceptance rate indicates model quality degradation.
