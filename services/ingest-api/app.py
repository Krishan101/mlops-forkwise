"""
ForkWise Ingest Service
Polls Mealie's API for recipes, stores metadata + ingredients in the platform
postgres, and queues feature jobs for the embedding worker.
Runs as a long-lived pod with a configurable poll interval.
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


def ingest_recipe(conn, recipe: dict):
    recipe_id = recipe.get("id", "")
    name = recipe.get("name", "")
    slug = recipe.get("slug", "")
    ingredients = recipe.get("recipeIngredient", [])

    with conn.cursor() as cur:
        # Check if already ingested
        cur.execute("SELECT 1 FROM recipe_metadata WHERE recipe_id = %s", (recipe_id,))
        if cur.fetchone():
            return False

        # Insert metadata
        cur.execute(
            """INSERT INTO recipe_metadata (recipe_id, name, slug, source, ingredient_count)
               VALUES (%s, %s, %s, 'mealie', %s)""",
            (recipe_id, name, slug, len(ingredients)),
        )

        # Insert ingredients
        for i, ing in enumerate(ingredients):
            note = ing.get("note", "") or ing.get("display", "") or str(ing)
            if not note.strip():
                continue
            cur.execute(
                """INSERT INTO recipe_ingredients (recipe_id, ingredient, position)
                   VALUES (%s, %s, %s)""",
                (recipe_id, note.strip(), i),
            )

        # Queue feature job
        cur.execute(
            """INSERT INTO feature_jobs (recipe_id, status)
               VALUES (%s, 'pending')""",
            (recipe_id,),
        )

        conn.commit()
    return True


def poll_loop():
    log.info(f"starting ingest service, polling every {POLL_INTERVAL}s")
    log.info(f"mealie: {MEALIE_URL}")

    while True:
        try:
            with httpx.Client(timeout=30) as client:
                token = mealie_login(client)
                recipes = fetch_all_recipes(client, token)
                log.info(f"found {len(recipes)} recipes in mealie")

                conn = get_pg_conn()
                new_count = 0
                for summary in recipes:
                    slug = summary.get("slug", "")
                    if not slug:
                        continue
                    try:
                        detail = fetch_recipe_detail(client, token, slug)
                        if ingest_recipe(conn, detail):
                            log.info(f"ingested: {detail.get('name', slug)}")
                            new_count += 1
                    except Exception as e:
                        log.warning(f"failed to ingest {slug}: {e}")

                conn.close()
                if new_count > 0:
                    log.info(f"ingested {new_count} new recipes")

        except Exception as e:
            log.error(f"poll cycle failed: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    poll_loop()
