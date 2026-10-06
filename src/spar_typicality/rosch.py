"""Rosch (1975) category-item pairs and their grammatical forms."""

import csv

from spar_typicality.prompts import MASS, PLURAL, SINGULAR, Category, Noun, Pair

CATEGORIES = {
    "toy": Category("toy", "a toy", "toys"),
    "bird": Category("bird", "a bird", "birds"),
    "sport": Category("sport", "a sport", "sports"),
    "vegetable": Category("vegetable", "a vegetable", "vegetables"),
    "tool": Category("tool", "a tool", "tools"),
    "fruit": Category("fruit", "a fruit", "fruits"),
    "clothing": Category("clothing", "a piece of clothing", "clothing"),
    "vehicle": Category("vehicle", "a vehicle", "vehicles"),
    "furniture": Category("furniture", "a piece of furniture", "furniture"),
    "weapon": Category("weapon", "a weapon", "weapons"),
}

# The raw data gives some plural items in a singular form (for example
# 'pant'). These items get their natural plural form in the prompts.
PLURAL_FORMS = {
    "pant": "pants",
    "slack": "slacks",
    "underpant": "underpants",
    "panty": "panties",
    "pajama": "pajamas",
    "nylon": "nylons",
    "earmuff": "earmuffs",
    "scissor": "scissors",
    "plier": "pliers",
    "brass knuckle": "brass knuckles",
    "drape": "drapes",
    "green": "greens",
    "turnip green": "turnip greens",
    "collard": "collards",
    "baked bean": "baked beans",
}

# Items that are already plural in the raw data.
PLURAL_ITEMS = {
    "marbles",
    "jacks",
    "paper dolls",
    "crayons",
    "skates",
    "dishes",
    "animals",
    "books",
    "cards",
    "stilts",
    "brussels sprouts",
    "feet",
}

# Items that take no article (mass nouns and names of games).
# All items in the 'sport' category are also mass nouns.
MASS_ITEMS = {
    "clay",
    "checkers",
    "monopoly",
    "spinach",
    "broccoli",
    "asparagus",
    "corn",
    "cauliflower",
    "lettuce",
    "celery",
    "okra",
    "parsley",
    "kale",
    "escarole",
    "sauerkraut",
    "seaweed",
    "garlic",
    "rice",
    "watercress",
    "romaine",
    "rhubarb",
    "sandpaper",
    "wood",
    "lumber",
    "chalk",
    "glue",
    "varnish",
    "plaster",
    "cement",
    "teargas",
    "poison",
    "gas",
    "glass",
    "judo",
}

# Spelling changes that are not about number.
SPELLING = {
    "monopoly": "Monopoly",
    "Atom bomb": "atom bomb",
    "brussels sprouts": "Brussels sprouts",
}


def make_noun(item, source_category):
    """Return the prompt form of an item from the given source category."""
    text = SPELLING.get(item, item)
    if source_category == "sport":
        return Noun(text, MASS)
    if item in PLURAL_FORMS:
        return Noun(PLURAL_FORMS[item], PLURAL)
    if item in PLURAL_ITEMS:
        return Noun(text, PLURAL)
    if item in MASS_ITEMS:
        return Noun(text, MASS)
    return Noun(text, SINGULAR)


def read_raw_rows(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def member_pairs(raw_rows):
    """Return a Pair for each row of the raw Rosch data."""
    pairs = []
    for row in raw_rows:
        pair = Pair(
            noun=make_noun(row["item"], row["category"]),
            category=CATEGORIES[row["category"]],
            rating_raw=float(row["rating"]),
            rating_normalised=float(row["rating_zscore"]),
            is_member=True,
        )
        pairs.append(pair)
    return pairs


def non_member_pairs(random_pairs):
    """Return a Pair for each (item, source_category, target_category)."""
    pairs = []
    for item, source_category, target_category in random_pairs:
        pair = Pair(
            noun=make_noun(item, source_category),
            category=CATEGORIES[target_category],
            rating_raw=None,
            rating_normalised=None,
            is_member=False,
        )
        pairs.append(pair)
    return pairs
