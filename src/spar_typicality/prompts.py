"""Build typicality prompt sets from category-item pairs."""

import csv
import random

# Grammatical number of an item noun phrase.
SINGULAR = "singular"
PLURAL = "plural"
MASS = "mass"

FIELDNAMES = [
    "prompt",
    "item",
    "category",
    "typicality_rating_raw",
    "typicality_rating_normalised",
    "is_member",
]


class Noun:
    """An item as it appears in a prompt."""

    def __init__(self, text, number):
        self.text = text
        self.number = number

    def noun_phrase(self):
        """Return the item with its article, for example 'an apple'."""
        if self.number == SINGULAR:
            if self.text[0].lower() in "aeiou":
                return "an " + self.text
            return "a " + self.text
        return self.text

    def verb_be(self):
        if self.number == PLURAL:
            return "are"
        return "is"


class Category:
    """A category as it appears in a prompt."""

    def __init__(self, text, singular_phrase, plural_phrase):
        self.text = text
        self.singular_phrase = singular_phrase
        self.plural_phrase = plural_phrase

    def predicate_for(self, noun):
        """Return the category phrase that agrees with the noun."""
        if noun.number == PLURAL:
            return self.plural_phrase
        return self.singular_phrase


def capitalise_first(text):
    return text[0].upper() + text[1:]


def example_clause(noun, category, negated=False):
    """Return a clause like 'A doll is a toy', with no full stop."""
    verb = noun.verb_be()
    if negated:
        verb = verb + " not"
    subject = capitalise_first(noun.noun_phrase())
    return subject + " " + verb + " " + category.predicate_for(noun)


def label_prompt(noun, category):
    return category.text + ": " + noun.text + "."


def neutral_prompt(noun, category):
    return "Word: " + noun.text + "."


def example_prompt(noun, category):
    return example_clause(noun, category) + "."


def neg_prompt(noun, category):
    return example_clause(noun, category, negated=True) + "."


def make_conjunction_prompt(suffix):
    """Return a prompt function that adds 'and <suffix>' to the example."""

    def conjunction_prompt(noun, category):
        return example_clause(noun, category) + " and " + suffix

    return conjunction_prompt


def story_prompt(noun, category):
    return (
        "Once upon a time there was a girl named Lila. "
        "In her house there was a magical room that contained "
        + category.plural_phrase
        + " and when she entered she saw "
        + noun.noun_phrase()
        + "."
    )


# Prompt set suffix -> (prompt function, does the prompt contain the category).
PROMPT_SETS = {
    "label": (label_prompt, True),
    "neutral": (neutral_prompt, False),
    "example": (example_prompt, True),
    "neg": (neg_prompt, True),
    "paris_true": (make_conjunction_prompt("Paris is in France."), True),
    "paris_false": (make_conjunction_prompt("Paris is in China."), True),
    "beijing_true": (make_conjunction_prompt("Beijing is in China."), True),
    "beijing_false": (make_conjunction_prompt("Beijing is in France."), True),
    "addition_true": (make_conjunction_prompt("11+10=21."), True),
    "addition_false": (make_conjunction_prompt("11+10=25."), True),
    "story": (story_prompt, True),
}


class Pair:
    """One category-item pair, with its typicality data.

    Rating values are None for random (non-member) pairs.
    """

    def __init__(self, noun, category, rating_raw, rating_normalised, is_member):
        self.noun = noun
        self.category = category
        self.rating_raw = rating_raw
        self.rating_normalised = rating_normalised
        self.is_member = is_member


def sample_random_pairs(rows, count, seed):
    """Sample random item-category pairs where the item is not a member.

    `rows` is a list of dicts with the keys 'item' and 'category'.
    Items that are listed in more than one category are not used, because
    their membership in other categories is ambiguous (for example 'knife').
    Returns a list of (item, source_category, target_category) tuples.
    """
    categories_of_item = {}
    for row in rows:
        categories_of_item.setdefault(row["item"], set()).add(row["category"])

    candidate_rows = []
    for row in rows:
        if len(categories_of_item[row["item"]]) == 1:
            candidate_rows.append(row)

    all_categories = sorted({row["category"] for row in rows})
    rng = random.Random(seed)
    chosen = []
    used = set()
    while len(chosen) < count:
        row = rng.choice(candidate_rows)
        target_category = rng.choice(all_categories)
        if target_category == row["category"]:
            continue
        key = (row["item"], target_category)
        if key in used:
            continue
        used.add(key)
        chosen.append((row["item"], row["category"], target_category))
    return chosen


def build_rows(pairs, prompt_function, contains_category):
    """Return the CSV rows of one prompt set."""
    rows = []
    for pair in pairs:
        row = {
            "prompt": prompt_function(pair.noun, pair.category),
            "item": pair.noun.text,
            "category": None,
            "typicality_rating_raw": pair.rating_raw,
            "typicality_rating_normalised": pair.rating_normalised,
            "is_member": None,
        }
        if contains_category:
            row["category"] = pair.category.text
            row["is_member"] = int(pair.is_member)
        rows.append(row)
    return rows


def write_rows(rows, path):
    """Write rows to CSV. None values are written as empty cells."""
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
