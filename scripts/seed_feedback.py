"""
Seed Feedback Script
====================
Generates test feedback by making substitution queries and submitting
accept/reject feedback to the substitution API.

Run on node1:
    python3 seed_feedback.py
"""

import requests
import json

API = "http://192.168.1.11:30808"

test_cases = [
    {"ingredient": "butter", "recipe_name": "Classic Pancakes", "accept_idx": 0},
    {"ingredient": "eggs", "recipe_name": "Classic Pancakes", "accept_idx": 0},
    {"ingredient": "sugar", "recipe_name": "Chocolate Chip Cookies", "accept_idx": 1},
    {"ingredient": "chocolate chips", "recipe_name": "Chocolate Chip Cookies", "accept_idx": 0},
    {"ingredient": "soy sauce", "recipe_name": "Chicken Stir Fry", "accept_idx": 0},
    {"ingredient": "chicken breast", "recipe_name": "Chicken Stir Fry", "accept_idx": 1},
    {"ingredient": "milk", "recipe_name": "Classic Pancakes", "accept_idx": 0},
    {"ingredient": "flour", "recipe_name": "Classic Pancakes", "accept_idx": 0},
    {"ingredient": "vanilla extract", "recipe_name": "Chocolate Chip Cookies", "accept_idx": 0},
    {"ingredient": "garlic", "recipe_name": "Chicken Stir Fry", "accept_idx": 0},
]

print("Seeding feedback data...")
print()

for tc in test_cases:
    # Make substitution query
    r = requests.post(f"{API}/substitute", json={
        "ingredient": tc["ingredient"],
        "recipe_name": tc["recipe_name"],
        "top_k": 3
    })
    result = r.json()
    query_id = result["query_id"]
    suggestions = result["suggestions"]

    if not suggestions:
        print(f"  No suggestions for {tc['ingredient']} -- skipping")
        continue

    # Accept one suggestion
    accept_idx = min(tc["accept_idx"], len(suggestions) - 1)
    accepted = suggestions[accept_idx]["ingredient"]

    r2 = requests.post(f"{API}/feedback", json={
        "request_id": query_id,
        "recipe_id": "test",
        "missing_ingredient": tc["ingredient"],
        "suggested_substitution": accepted,
        "user_accepted": True,
    })

    # Reject others
    for i, s in enumerate(suggestions):
        if i != accept_idx:
            requests.post(f"{API}/feedback", json={
                "request_id": query_id,
                "recipe_id": "test",
                "missing_ingredient": tc["ingredient"],
                "suggested_substitution": s["ingredient"],
                "user_accepted": False,
            })

    print(f"  {tc['ingredient']:20s} -> accepted: {accepted[:40]}")

print(f"\nDone! Seeded feedback for {len(test_cases)} queries")
