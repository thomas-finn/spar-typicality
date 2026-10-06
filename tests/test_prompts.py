from spar_typicality.prompts import (
    MASS,
    PLURAL,
    PROMPT_SETS,
    SINGULAR,
    Noun,
    build_rows,
    sample_random_pairs,
)
from spar_typicality.rosch import CATEGORIES, make_noun, member_pairs


def prompt(set_name, noun, category_name):
    prompt_function = PROMPT_SETS[set_name][0]
    return prompt_function(noun, CATEGORIES[category_name])


def test_templates_singular():
    doll = Noun("doll", SINGULAR)
    assert prompt("label", doll, "toy") == "toy: doll."
    assert prompt("neutral", doll, "toy") == "Word: doll."
    assert prompt("example", doll, "toy") == "A doll is a toy."
    assert prompt("neg", doll, "toy") == "A doll is not a toy."
    assert prompt("paris_false", doll, "toy") == (
        "A doll is a toy and Paris is in China."
    )
    assert prompt("addition_true", doll, "toy") == "A doll is a toy and 11+10=21."
    assert prompt("story", doll, "toy") == (
        "Once upon a time there was a girl named Lila. In her house there was "
        "a magical room that contained toys and when she entered she saw a doll."
    )


def test_article_an():
    assert prompt("example", Noun("apple", SINGULAR), "fruit") == (
        "An apple is a fruit."
    )


def test_plural_and_mass():
    assert prompt("example", Noun("pants", PLURAL), "clothing") == (
        "Pants are clothing."
    )
    assert prompt("neg", Noun("swimming", MASS), "sport") == (
        "Swimming is not a sport."
    )
    assert prompt("example", Noun("hat", SINGULAR), "clothing") == (
        "A hat is a piece of clothing."
    )


def test_make_noun():
    assert make_noun("pant", "clothing").text == "pants"
    assert make_noun("squash", "sport").number == MASS
    assert make_noun("squash", "vegetable").number == SINGULAR
    assert make_noun("cards", "sport").number == MASS
    assert make_noun("cards", "toy").number == PLURAL


def test_random_pairs_are_not_members():
    rows = [
        {"item": "doll", "category": "toy"},
        {"item": "ball", "category": "toy"},
        {"item": "knife", "category": "tool"},
        {"item": "knife", "category": "weapon"},
        {"item": "robin", "category": "bird"},
    ]
    pairs = sample_random_pairs(rows, 5, seed=0)
    assert len(pairs) == 5
    assert len(set(pairs)) == 5
    for item, source_category, target_category in pairs:
        assert item != "knife"
        assert source_category != target_category


def test_build_rows_neutral_has_no_category():
    rows = [{"item": "doll", "category": "toy", "rating": "1.4", "rating_zscore": "-1"}]
    pairs = member_pairs(rows)
    neutral_function, contains_category = PROMPT_SETS["neutral"]
    output = build_rows(pairs, neutral_function, contains_category)
    assert output[0]["category"] is None
    assert output[0]["is_member"] is None

    example_function, contains_category = PROMPT_SETS["example"]
    output = build_rows(pairs, example_function, contains_category)
    assert output[0]["category"] == "toy"
    assert output[0]["is_member"] == 1
