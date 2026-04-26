"""
Data loading for GISMo training.
Loads:
  - FlavorGraph (nodes_191120.csv, edges_191120.csv) from S3
  - Recipe1MSubs (train.json, val.json, test.json) from S3
  - User feedback from PostgreSQL (for retraining)

Builds:
  - Ingredient vocabulary (ingredient name → index)
  - PyTorch Geometric graph (edge_index, edge_weight)
  - Training batches: (source_id, positive_id, negative_ids, recipe_ingredient_ids)
"""

import csv
import io
import json
import logging
import os
import pickle
import random
from collections import defaultdict

import boto3
import torch
from botocore.client import Config

log = logging.getLogger("gismo-data")

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://chi.tacc.chameleoncloud.org:7480")
S3_BUCKET = os.environ.get("S3_BUCKET", "data-proj01")
S3_ACCESS = os.environ.get("AWS_ACCESS_KEY_ID", "8921c48faf83433db2b1439a9b2889fd")
S3_SECRET = os.environ.get("AWS_SECRET_ACCESS_KEY", "7d1ce78efc5a48019888c9f3fa8ba2dd")


def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS,
        aws_secret_access_key=S3_SECRET,
        config=Config(signature_version="s3"),
    )


def s3_read(key):
    """Read an S3 object and return bytes."""
    s3 = get_s3()
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return obj["Body"].read()


def load_flavorgraph():
    """
    Load FlavorGraph from S3.
    Returns:
        vocab: dict[str, int] — ingredient name → index
        edge_index: (2, num_edges) tensor
        edge_weight: (num_edges,) tensor
    """
    log.info("loading FlavorGraph nodes from S3...")
    nodes_bytes = s3_read("data/raw/flavorgraph/nodes_191120.csv")
    nodes_csv = csv.DictReader(io.StringIO(nodes_bytes.decode("utf-8")))

    vocab = {}
    node_types = {}
    for row in nodes_csv:
        name = row.get("name", row.get("node_id", "")).strip().lower()
        ntype = row.get("node_type", "ingredient")
        idx = len(vocab)
        if name not in vocab:
            vocab[name] = idx
            node_types[idx] = ntype

    log.info(f"  {len(vocab)} nodes loaded")

    log.info("loading FlavorGraph edges from S3...")
    edges_bytes = s3_read("data/raw/flavorgraph/edges_191120.csv")
    edges_csv = csv.DictReader(io.StringIO(edges_bytes.decode("utf-8")))

    src_list, dst_list, weight_list = [], [], []
    for row in edges_csv:
        s = row.get("source", row.get("node_id_1", "")).strip().lower()
        d = row.get("target", row.get("node_id_2", "")).strip().lower()
        w = float(row.get("weight", row.get("score", 1.0)))
        if s in vocab and d in vocab:
            src_list.append(vocab[s])
            dst_list.append(vocab[d])
            weight_list.append(w)
            # Undirected: add reverse edge
            src_list.append(vocab[d])
            dst_list.append(vocab[s])
            weight_list.append(w)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float32)
    log.info(f"  {edge_index.shape[1]} edges loaded (undirected)")

    return vocab, edge_index, edge_weight


def load_recipe1msubs(split="train"):
    """
    Load Recipe1MSubs substitution pairs from S3.
    Returns list of dicts: [{source, target, recipe_id}, ...]
    """
    key = f"data/raw/recipe1msubs/{split}.json"
    log.info(f"loading Recipe1MSubs {split} from S3 ({key})...")
    data = json.loads(s3_read(key))
    log.info(f"  {len(data)} substitution pairs loaded")
    return data


def load_recipe1m_ingredients(max_recipes=None):
    """
    Load Recipe1M layer1.json to get recipe ingredient lists.
    Returns dict: {recipe_id: [ingredient_name, ...]}
    """
    log.info("loading Recipe1M layer1.json from S3 (may take a moment)...")
    data = json.loads(s3_read("data/raw/recipe1m/layer1.json"))
    recipes = {}
    for i, recipe in enumerate(data):
        if max_recipes and i >= max_recipes:
            break
        rid = recipe.get("id", "")
        ings = []
        for ing in recipe.get("ingredients", []):
            text = ing.get("text", "").strip().lower()
            if text:
                ings.append(text)
        if ings:
            recipes[rid] = ings
    log.info(f"  {len(recipes)} recipes loaded")
    return recipes


def load_feedback_pairs(pg_conn):
    """
    Load accept/reject feedback from PostgreSQL for retraining.
    Returns list of dicts similar to Recipe1MSubs format.
    """
    log.info("loading feedback pairs from PostgreSQL...")
    import psycopg2.extras
    with pg_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute("""
            SELECT sq.original_ingredient AS source,
                   sr.suggested_ingredient AS target,
                   sq.recipe_id,
                   sq.query_context AS recipe_name,
                   fe.event_type,
                   sr.accepted
            FROM feedback_events fe
            JOIN substitution_queries sq ON sq.query_id = fe.query_id
            JOIN substitution_results sr ON sr.query_id = fe.query_id
                AND sr.suggested_ingredient = fe.suggested_ingredient
            WHERE fe.event_type IN ('accept', 'reject')
        """)
        rows = cur.fetchall()

    pairs = []
    for row in rows:
        pairs.append({
            "source": row["source"],
            "target": row["target"],
            "recipe_id": row["recipe_id"] or "",
            "accepted": row["event_type"] == "accept",
        })
    log.info(f"  {len(pairs)} feedback pairs loaded ({sum(1 for p in pairs if p['accepted'])} accepted)")
    return pairs


def normalize_ingredient(name, vocab):
    """Try to find an ingredient in the vocabulary, with fallback normalization."""
    name = name.strip().lower()
    if name in vocab:
        return vocab[name]
    # Try without numbers/quantities
    words = [w for w in name.split() if not any(c.isdigit() for c in w)]
    cleaned = " ".join(words[-2:]) if len(words) > 2 else " ".join(words)
    if cleaned in vocab:
        return vocab[cleaned]
    # Try last word only (the actual ingredient)
    if words and words[-1] in vocab:
        return vocab[words[-1]]
    return None


class SubstitutionDataset:
    """
    Dataset for GISMo training.
    Each sample: (source_idx, positive_idx, [negative_idxs], [recipe_ingredient_idxs])
    """

    def __init__(self, pairs, vocab, recipes=None, num_negatives=10):
        self.vocab = vocab
        self.num_negatives = num_negatives
        self.all_indices = list(range(len(vocab)))
        self.samples = []

        for pair in pairs:
            src_idx = normalize_ingredient(pair["source"], vocab)
            tgt_idx = normalize_ingredient(pair["target"], vocab)
            if src_idx is None or tgt_idx is None:
                continue

            # Get recipe ingredient indices
            recipe_ings = []
            if recipes and pair.get("recipe_id") in recipes:
                for ing_name in recipes[pair["recipe_id"]]:
                    idx = normalize_ingredient(ing_name, vocab)
                    if idx is not None:
                        recipe_ings.append(idx)

            if not recipe_ings:
                recipe_ings = [src_idx]  # Fallback: just the source ingredient

            self.samples.append({
                "source": src_idx,
                "positive": tgt_idx,
                "recipe_ingredients": recipe_ings,
                "accepted": pair.get("accepted", True),
            })

        log.info(f"  dataset: {len(self.samples)} valid samples from {len(pairs)} pairs")

    def __len__(self):
        return len(self.samples)

    def get_batch(self, batch_size):
        """Get a random batch with negative sampling."""
        batch = random.sample(self.samples, min(batch_size, len(self.samples)))

        source_ids = []
        positive_ids = []
        negative_ids = []
        recipe_ingredient_ids = []

        for sample in batch:
            source_ids.append(sample["source"])
            positive_ids.append(sample["positive"])

            # Sample negatives: random ingredients excluding source and positive
            negs = []
            exclude = {sample["source"], sample["positive"]}
            while len(negs) < self.num_negatives:
                neg = random.choice(self.all_indices)
                if neg not in exclude:
                    negs.append(neg)
                    exclude.add(neg)
            negative_ids.append(negs)
            recipe_ingredient_ids.append(sample["recipe_ingredients"])

        return {
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "positive_ids": torch.tensor(positive_ids, dtype=torch.long),
            "negative_ids": torch.tensor(negative_ids, dtype=torch.long),
            "recipe_ingredient_ids": recipe_ingredient_ids,
        }
