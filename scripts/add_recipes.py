"""
Add test recipes to Mealie via API.
Run on node1 after creating a user account in Mealie.

Usage:
    python3 scripts/add_recipes.py
"""

import requests
import time

MEALIE_URL = "http://192.168.1.11:30900"
EMAIL = "krishankumargupta101@gmail.com"
PASSWORD = "mynameiskrishan"

# Login
r = requests.post(f"{MEALIE_URL}/api/auth/token",
    data={"username": EMAIL, "password": PASSWORD},
    headers={"Content-Type": "application/x-www-form-urlencoded"})
token = r.json()["access_token"]
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

RECIPES = [
    # Western / Baking
    {
        "name": "Classic Pancakes",
        "recipeIngredient": [
            {"note": "1 1/2 cups all-purpose flour"},
            {"note": "2 large eggs"},
            {"note": "1 cup whole milk"},
            {"note": "3 tablespoons butter, melted"},
            {"note": "2 tablespoons sugar"},
            {"note": "2 teaspoons baking powder"},
            {"note": "1/2 teaspoon salt"},
            {"note": "1 teaspoon vanilla extract"},
        ],
        "recipeInstructions": [{"text": "Mix dry ingredients. Whisk wet ingredients separately. Combine and cook on griddle."}],
    },
    {
        "name": "Chocolate Chip Cookies",
        "recipeIngredient": [
            {"note": "2 1/4 cups all-purpose flour"},
            {"note": "1 cup butter, softened"},
            {"note": "3/4 cup sugar"},
            {"note": "3/4 cup brown sugar"},
            {"note": "2 large eggs"},
            {"note": "1 teaspoon vanilla extract"},
            {"note": "1 teaspoon baking soda"},
            {"note": "2 cups chocolate chips"},
            {"note": "1 teaspoon salt"},
        ],
        "recipeInstructions": [{"text": "Cream butter and sugars. Add eggs and vanilla. Mix in dry ingredients and chocolate chips. Bake at 375F for 10 minutes."}],
    },
    {
        "name": "Banana Bread",
        "recipeIngredient": [
            {"note": "3 ripe bananas"},
            {"note": "1/3 cup butter, melted"},
            {"note": "3/4 cup sugar"},
            {"note": "1 large egg"},
            {"note": "1 teaspoon vanilla extract"},
            {"note": "1 teaspoon baking soda"},
            {"note": "1/4 teaspoon salt"},
            {"note": "1 1/2 cups all-purpose flour"},
            {"note": "1 teaspoon cinnamon"},
        ],
        "recipeInstructions": [{"text": "Mash bananas. Mix in butter, sugar, egg, vanilla. Add dry ingredients. Bake at 350F for 60 minutes."}],
    },
    {
        "name": "Blueberry Muffins",
        "recipeIngredient": [
            {"note": "2 cups all-purpose flour"},
            {"note": "3/4 cup sugar"},
            {"note": "2 1/2 teaspoons baking powder"},
            {"note": "1/3 cup vegetable oil"},
            {"note": "1 large egg"},
            {"note": "1 cup whole milk"},
            {"note": "1 1/2 cups fresh blueberries"},
            {"note": "1/2 teaspoon salt"},
            {"note": "1 teaspoon vanilla extract"},
        ],
        "recipeInstructions": [{"text": "Mix dry ingredients. Combine wet ingredients. Fold together, add blueberries. Bake at 400F for 20 minutes."}],
    },
    # Italian
    {
        "name": "Pasta Carbonara",
        "recipeIngredient": [
            {"note": "1 lb spaghetti"},
            {"note": "6 oz pancetta"},
            {"note": "4 large egg yolks"},
            {"note": "1 cup parmesan cheese, grated"},
            {"note": "2 cloves garlic"},
            {"note": "2 tablespoons olive oil"},
            {"note": "1/2 teaspoon black pepper"},
            {"note": "1/2 teaspoon salt"},
        ],
        "recipeInstructions": [{"text": "Cook pasta. Fry pancetta with garlic. Whisk egg yolks with parmesan. Toss hot pasta with pancetta, then egg mixture."}],
    },
    {
        "name": "Margherita Pizza",
        "recipeIngredient": [
            {"note": "2 1/4 teaspoons active dry yeast"},
            {"note": "3 cups bread flour"},
            {"note": "1 tablespoon olive oil"},
            {"note": "1 teaspoon sugar"},
            {"note": "1 teaspoon salt"},
            {"note": "1 cup warm water"},
            {"note": "1/2 cup tomato sauce"},
            {"note": "8 oz fresh mozzarella"},
            {"note": "fresh basil leaves"},
        ],
        "recipeInstructions": [{"text": "Make dough, let rise. Stretch, top with sauce, mozzarella, basil. Bake at 475F for 12 minutes."}],
    },
    {
        "name": "Tomato Basil Soup",
        "recipeIngredient": [
            {"note": "2 lbs fresh tomatoes"},
            {"note": "1 medium onion, diced"},
            {"note": "3 cloves garlic, minced"},
            {"note": "2 tablespoons olive oil"},
            {"note": "2 cups chicken broth"},
            {"note": "1/4 cup fresh basil"},
            {"note": "1/2 cup heavy cream"},
            {"note": "1 teaspoon salt"},
            {"note": "1/2 teaspoon black pepper"},
        ],
        "recipeInstructions": [{"text": "Saute onion and garlic. Add tomatoes and broth. Simmer 20 min. Blend, add cream and basil."}],
    },
    # Asian
    {
        "name": "Chicken Stir Fry",
        "recipeIngredient": [
            {"note": "1 lb chicken breast, sliced"},
            {"note": "2 tablespoons soy sauce"},
            {"note": "1 tablespoon sesame oil"},
            {"note": "2 cloves garlic, minced"},
            {"note": "1 teaspoon fresh ginger, grated"},
            {"note": "1 red bell pepper, sliced"},
            {"note": "1 cup broccoli florets"},
            {"note": "2 tablespoons vegetable oil"},
            {"note": "1 tablespoon cornstarch"},
            {"note": "1 teaspoon salt"},
        ],
        "recipeInstructions": [{"text": "Stir fry chicken in oil. Add vegetables and garlic. Mix soy sauce, sesame oil, cornstarch. Toss everything together."}],
    },
    {
        "name": "Pad Thai",
        "recipeIngredient": [
            {"note": "8 oz rice noodles"},
            {"note": "1/2 lb shrimp, peeled"},
            {"note": "2 tablespoons fish sauce"},
            {"note": "1 tablespoon tamarind paste"},
            {"note": "1/4 cup peanuts, crushed"},
            {"note": "1 cup bean sprouts"},
            {"note": "2 large eggs"},
            {"note": "1 lime, juiced"},
            {"note": "2 tablespoons vegetable oil"},
            {"note": "2 green onions, sliced"},
        ],
        "recipeInstructions": [{"text": "Soak noodles. Stir fry shrimp, push aside, scramble eggs. Add noodles, fish sauce, tamarind. Top with peanuts, sprouts, lime."}],
    },
    {
        "name": "Chicken Teriyaki",
        "recipeIngredient": [
            {"note": "1 lb chicken thighs"},
            {"note": "1/4 cup soy sauce"},
            {"note": "2 tablespoons mirin"},
            {"note": "2 tablespoons honey"},
            {"note": "1 tablespoon rice vinegar"},
            {"note": "1 clove garlic, minced"},
            {"note": "1 teaspoon fresh ginger"},
            {"note": "1 tablespoon vegetable oil"},
            {"note": "1 tablespoon sesame seeds"},
        ],
        "recipeInstructions": [{"text": "Mix soy sauce, mirin, honey, vinegar, garlic, ginger for sauce. Cook chicken in oil, glaze with sauce. Garnish with sesame seeds."}],
    },
    {
        "name": "Fried Rice",
        "recipeIngredient": [
            {"note": "3 cups cooked white rice, cold"},
            {"note": "2 tablespoons soy sauce"},
            {"note": "1 tablespoon sesame oil"},
            {"note": "2 large eggs, beaten"},
            {"note": "1 cup frozen peas and carrots"},
            {"note": "3 green onions, chopped"},
            {"note": "2 cloves garlic, minced"},
            {"note": "2 tablespoons vegetable oil"},
        ],
        "recipeInstructions": [{"text": "Heat oil, scramble eggs. Add garlic, vegetables, rice. Toss with soy sauce and sesame oil. Top with green onions."}],
    },
    {
        "name": "Miso Glazed Eggplant",
        "recipeIngredient": [
            {"note": "2 large eggplants"},
            {"note": "3 tablespoons white miso paste"},
            {"note": "2 tablespoons mirin"},
            {"note": "1 tablespoon sake"},
            {"note": "1 tablespoon sesame seeds"},
            {"note": "2 green onions, sliced"},
            {"note": "1 tablespoon rice vinegar"},
            {"note": "1 tablespoon sugar"},
        ],
        "recipeInstructions": [{"text": "Halve eggplants, score flesh. Mix miso, mirin, sake, sugar. Brush on eggplant. Broil until caramelized. Top with sesame seeds and green onions."}],
    },
    # Mexican / Latin
    {
        "name": "Chicken Tacos",
        "recipeIngredient": [
            {"note": "1 lb chicken breast"},
            {"note": "8 corn tortillas"},
            {"note": "1 cup salsa"},
            {"note": "1 avocado, sliced"},
            {"note": "1/2 cup cilantro, chopped"},
            {"note": "1 lime, juiced"},
            {"note": "1 teaspoon cumin"},
            {"note": "1 teaspoon chili powder"},
            {"note": "1/2 cup sour cream"},
            {"note": "1 teaspoon salt"},
        ],
        "recipeInstructions": [{"text": "Season chicken with cumin, chili powder, salt. Grill and slice. Serve in tortillas with salsa, avocado, cilantro, sour cream, lime."}],
    },
    {
        "name": "Guacamole",
        "recipeIngredient": [
            {"note": "3 ripe avocados"},
            {"note": "1 lime, juiced"},
            {"note": "1/2 teaspoon salt"},
            {"note": "1/2 cup onion, diced"},
            {"note": "2 tablespoons cilantro, chopped"},
            {"note": "1 jalapeno, seeded and minced"},
            {"note": "1 medium tomato, diced"},
            {"note": "1 clove garlic, minced"},
        ],
        "recipeInstructions": [{"text": "Mash avocados. Mix in lime juice, salt, onion, cilantro, jalapeno, tomato, garlic. Serve with chips."}],
    },
    # Salads / Light
    {
        "name": "Caesar Salad",
        "recipeIngredient": [
            {"note": "1 head romaine lettuce"},
            {"note": "1/2 cup parmesan cheese, shaved"},
            {"note": "1 cup croutons"},
            {"note": "2 tablespoons lemon juice"},
            {"note": "3 tablespoons olive oil"},
            {"note": "2 cloves garlic, minced"},
            {"note": "1 teaspoon anchovy paste"},
            {"note": "1 teaspoon dijon mustard"},
        ],
        "recipeInstructions": [{"text": "Whisk lemon juice, olive oil, garlic, anchovy paste, mustard for dressing. Toss with lettuce, parmesan, croutons."}],
    },
    {
        "name": "Greek Salad",
        "recipeIngredient": [
            {"note": "2 large cucumbers, chopped"},
            {"note": "4 medium tomatoes, chopped"},
            {"note": "1 red onion, thinly sliced"},
            {"note": "1 cup kalamata olives"},
            {"note": "6 oz feta cheese, crumbled"},
            {"note": "3 tablespoons olive oil"},
            {"note": "1 tablespoon red wine vinegar"},
            {"note": "1 teaspoon dried oregano"},
            {"note": "1/2 teaspoon salt"},
        ],
        "recipeInstructions": [{"text": "Combine cucumbers, tomatoes, onion, olives, feta. Dress with olive oil, vinegar, oregano, salt."}],
    },
    # Comfort
    {
        "name": "Grilled Cheese Sandwich",
        "recipeIngredient": [
            {"note": "4 slices white bread"},
            {"note": "4 slices cheddar cheese"},
            {"note": "2 tablespoons butter"},
        ],
        "recipeInstructions": [{"text": "Butter bread on outside. Layer cheese between slices. Cook in skillet until golden and cheese melts."}],
    },
    {
        "name": "French Onion Soup",
        "recipeIngredient": [
            {"note": "4 large onions, thinly sliced"},
            {"note": "3 tablespoons butter"},
            {"note": "1 tablespoon olive oil"},
            {"note": "4 cups beef broth"},
            {"note": "1/2 cup dry white wine"},
            {"note": "1 teaspoon fresh thyme"},
            {"note": "4 slices French bread"},
            {"note": "1 cup gruyere cheese, shredded"},
            {"note": "1/2 teaspoon salt"},
            {"note": "1/4 teaspoon black pepper"},
        ],
        "recipeInstructions": [{"text": "Caramelize onions in butter and oil for 30 min. Add wine, then broth and thyme. Simmer 20 min. Ladle into bowls, top with bread and cheese. Broil until bubbly."}],
    },
    # Breakfast
    {
        "name": "Eggs Benedict",
        "recipeIngredient": [
            {"note": "4 large eggs"},
            {"note": "2 English muffins, split"},
            {"note": "4 slices Canadian bacon"},
            {"note": "3 egg yolks"},
            {"note": "1/2 cup butter, melted"},
            {"note": "1 tablespoon lemon juice"},
            {"note": "1/4 teaspoon cayenne pepper"},
            {"note": "1 tablespoon white vinegar"},
        ],
        "recipeInstructions": [{"text": "Poach eggs. Toast muffins, top with bacon. Make hollandaise: whisk yolks with lemon juice, slowly add melted butter, season. Top eggs with hollandaise."}],
    },
    {
        "name": "Overnight Oats",
        "recipeIngredient": [
            {"note": "1 cup rolled oats"},
            {"note": "1 cup whole milk"},
            {"note": "1/2 cup Greek yogurt"},
            {"note": "2 tablespoons honey"},
            {"note": "1 tablespoon chia seeds"},
            {"note": "1/2 teaspoon vanilla extract"},
            {"note": "1/2 cup fresh strawberries"},
            {"note": "2 tablespoons almond butter"},
        ],
        "recipeInstructions": [{"text": "Mix oats, milk, yogurt, honey, chia seeds, vanilla. Refrigerate overnight. Top with strawberries and almond butter."}],
    },
]

print(f"Adding {len(RECIPES)} recipes to Mealie...")
print()

added = 0
for recipe in RECIPES:
    # Create recipe
    r = requests.post(f"{MEALIE_URL}/api/recipes",
        json={"name": recipe["name"]},
        headers=headers)

    if r.status_code not in (200, 201):
        print(f"  SKIP {recipe['name']}: {r.status_code} {r.text[:100]}")
        continue

    slug = r.json()

    # Update with full details
    r2 = requests.put(f"{MEALIE_URL}/api/recipes/{slug}",
        json=recipe,
        headers=headers)

    if r2.status_code == 200:
        ing_count = len(recipe["recipeIngredient"])
        print(f"  ✓ {recipe['name']} ({ing_count} ingredients)")
        added += 1
    else:
        print(f"  FAIL {recipe['name']}: {r2.status_code}")

    time.sleep(0.5)

print(f"\nDone! Added {added}/{len(RECIPES)} recipes")
