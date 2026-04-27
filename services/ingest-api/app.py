"""
ForkWise Ingest Service
Polls Mealie's API for recipes, stores metadata + ingredients in the platform
postgres, and queues feature jobs for the embedding worker.
Runs as a long-lived pod with a configurable poll interval.

Data quality checks at ingestion:
  - Rejects recipes with empty names
  - Rejects recipes with 0 valid ingredients
  - Validates ingredient text length (skips empty or excessively long)
  - Logs quality metrics per poll cycle
"""

import logging
import os
import time

import httpx
import psycopg2
import psycopg2.extras

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","svc":"ingest-api","msg":"%(message)s"}',
)
log = logging.getLogger("ingest-api")

# --- Config ---
MEALIE_URL = os.environ.get("MEALIE_URL", "http://mealie.forkwise-app:9000")
MEALIE_EMAIL = os.environ["MEALIE_EMAIL"]
MEALIE_PASSWORD = os.environ["MEALIE_PASSWORD"]

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "platform-db")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "forkwise_mlops")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "forkwise")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "forkwise-secret-pw")

POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "30"))

# Data quality thresholds
MIN_INGREDIENT_LENGTH = 2      # Minimum characters for a valid ingredient
MAX_INGREDIENT_LENGTH = 500    # Maximum characters (reject garbage data)
MIN_INGREDIENTS_PER_RECIPE = 1 # Minimum valid ingredients to accept a recipe


def get_pg_conn():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def mealie_login(client: httpx.Client) -> str:
    r = client.post(
        f"{MEALIE_URL}/api/auth/token",
        data={"username": MEALIE_EMAIL, "password": MEALIE_PASSWORD},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    r.raise_for_status()
    return r.json()["access_token"]


def fetch_all_recipes(client: httpx.Client, token: str) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}"}
    r = client.get(f"{MEALIE_URL}/api/recipes", headers=headers, params={"perPage": 100})
    r.raise_for_status()
    return r.json().get("items", [])


def fetch_recipe_detail(client: httpx.Client, token: str, slug: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    r = client.get(f"{MEALIE_URL}/api/recipes/{slug}", headers=headers)
    r.raise_for_status()
    return r.json()


def validate_ingredient(note: str) -> tuple[bool, str]:
    """Validate a single ingredient string. Returns (is_valid, reason)."""
    if not note or not note.strip():
        return False, "empty"
    note = note.strip()
    if len(note) < MIN_INGREDIENT_LENGTH:
        return False, "too_short"
    if len(note) > MAX_INGREDIENT_LENGTH:
        return False, "too_long"
    return True, "ok"


def ingest_recipe(conn, recipe: dict) -> dict:
    """
    Ingest a recipe with data quality validation.
    Returns a quality report dict with counts of accepted/rejected ingredients.
    """
    recipe_id = recipe.get("id", "")
    name = recipe.get("name", "")
    slug = recipe.get("slug", "")
    ingredients = recipe.get("recipeIngredient", [])

    report = {
        "recipe_id": recipe_id,
        "name": name,
        "status": "skipped",
        "reason": None,
        "total_ingredients": len(ingredients),
        "valid_ingredients": 0,
        "rejected_empty": 0,
        "rejected_too_short": 0,
        "rejected_too_long": 0,
    }

    with conn.cursor() as cur:
        # Check if already ingested
        cur.execute("SELECT 1 FROM recipe_metadata WHERE recipe_id = %s", (recipe_id,))
        if cur.fetchone():
            report["status"] = "already_ingested"
            return report

        # Quality check: recipe must have a name
        if not name or not name.strip():
            report["status"] = "rejected"
            report["reason"] = "empty_name"
            log.warning(f"rejected recipe {recipe_id}: empty name")
            return report

        # Validate each ingredient
        valid_ingredients = []
        for i, ing in enumerate(ingredients):
            note = ing.get("note", "") or ing.get("display", "") or str(ing)
            is_valid, reason = validate_ingredient(note)

            if is_valid:
                valid_ingredients.append((note.strip(), i))
                report["valid_ingredients"] += 1
            else:
                report[f"rejected_{reason}"] = report.get(f"rejected_{reason}", 0) + 1

        # Quality check: recipe must have at least MIN_INGREDIENTS_PER_RECIPE valid ingredients
        if len(valid_ingredients) < MIN_INGREDIENTS_PER_RECIPE:
            report["status"] = "rejected"
            report["reason"] = "insufficient_ingredients"
            log.warning(f"rejected recipe {recipe_id} ({name}): "
                        f"only {len(valid_ingredients)} valid ingredients "
                        f"(minimum: {MIN_INGREDIENTS_PER_RECIPE})")
            return report

        # Insert metadata
        cur.execute(
            """INSERT INTO recipe_metadata (recipe_id, name, slug, source, ingredient_count)
               VALUES (%s, %s, %s, 'mealie', %s)""",
            (recipe_id, name, slug, len(valid_ingredients)),
        )

        # Insert validated ingredients
        for note, position in valid_ingredients:
            cur.execute(
                """INSERT INTO recipe_ingredients (recipe_id, ingredient, position)
                   VALUES (%s, %s, %s)""",
                (recipe_id, note, position),
            )

        # Queue feature job
        cur.execute(
            """INSERT INTO feature_jobs (recipe_id, status)
               VALUES (%s, 'pending')""",
            (recipe_id,),
        )

        conn.commit()

    report["status"] = "ingested"
    return report


def poll_loop():
    log.info(f"starting ingest service, polling every {POLL_INTERVAL}s")
    log.info(f"mealie: {MEALIE_URL}")
    log.info(f"quality thresholds: min_ingredient_len={MIN_INGREDIENT_LENGTH}, "
             f"max_ingredient_len={MAX_INGREDIENT_LENGTH}, "
             f"min_ingredients_per_recipe={MIN_INGREDIENTS_PER_RECIPE}")

    while True:
        try:
            with httpx.Client(timeout=30) as client:
                token = mealie_login(client)
                recipes = fetch_all_recipes(client, token)
                log.info(f"found {len(recipes)} recipes in mealie")

                conn = get_pg_conn()

                # Quality metrics for this poll cycle
                cycle_stats = {
                    "total_recipes": len(recipes),
                    "new_ingested": 0,
                    "already_ingested": 0,
                    "rejected": 0,
                    "failed": 0,
                    "total_ingredients_accepted": 0,
                    "total_ingredients_rejected": 0,
                }

                for summary in recipes:
                    slug = summary.get("slug", "")
                    if not slug:
                        continue
                    try:
                        detail = fetch_recipe_detail(client, token, slug)
                        report = ingest_recipe(conn, detail)

                        if report["status"] == "ingested":
                            cycle_stats["new_ingested"] += 1
                            cycle_stats["total_ingredients_accepted"] += report["valid_ingredients"]
                            cycle_stats["total_ingredients_rejected"] += (
                                report["rejected_empty"] +
                                report.get("rejected_too_short", 0) +
                                report.get("rejected_too_long", 0)
                            )
                            log.info(f"ingested: {detail.get('name', slug)} "
                                     f"({report['valid_ingredients']}/{report['total_ingredients']} ingredients valid)")
                        elif report["status"] == "already_ingested":
                            cycle_stats["already_ingested"] += 1
                        elif report["status"] == "rejected":
                            cycle_stats["rejected"] += 1
                    except Exception as e:
                        cycle_stats["failed"] += 1
                        log.warning(f"failed to ingest {slug}: {e}")

                conn.close()

                # Log quality summary for this poll cycle
                if cycle_stats["new_ingested"] > 0 or cycle_stats["rejected"] > 0:
                    avg_ingredients = (cycle_stats["total_ingredients_accepted"] /
                                      max(cycle_stats["new_ingested"], 1))
                    log.info(f"quality: new={cycle_stats['new_ingested']} "
                             f"rejected={cycle_stats['rejected']} "
                             f"failed={cycle_stats['failed']} "
                             f"ingredients_accepted={cycle_stats['total_ingredients_accepted']} "
                             f"ingredients_rejected={cycle_stats['total_ingredients_rejected']} "
                             f"avg_ingredients={avg_ingredients:.1f}")

        except Exception as e:
            log.error(f"poll cycle failed: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    poll_loop()
