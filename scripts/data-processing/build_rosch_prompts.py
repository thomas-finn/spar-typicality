"""Build the Rosch (1975) prompt sets in data/processed/rosch.

Usage:
    uv run python scripts/data-processing/build_rosch_prompts.py
"""

import argparse
import json
import subprocess
from pathlib import Path

from spar_typicality.prompts import (
    PROMPT_SETS,
    build_rows,
    sample_random_pairs,
    write_rows,
)
from spar_typicality.rosch import member_pairs, non_member_pairs, read_raw_rows

REPO_ROOT = Path(__file__).resolve().parents[2]


def git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw", default=REPO_ROOT / "data/raw/rosch1975_ratings.csv", type=Path
    )
    parser.add_argument("--out", default=REPO_ROOT / "data/processed/rosch", type=Path)
    parser.add_argument("--n-random", default=100, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    raw_rows = read_raw_rows(args.raw)
    random_pairs = sample_random_pairs(raw_rows, args.n_random, args.seed)
    pairs = member_pairs(raw_rows) + non_member_pairs(random_pairs)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, (prompt_function, contains_category) in PROMPT_SETS.items():
        rows = build_rows(pairs, prompt_function, contains_category)
        path = args.out / ("rosch_" + name + ".csv")
        write_rows(rows, path)
        print("Wrote", len(rows), "rows to", path)

    metadata = {
        "raw_data": str(args.raw.relative_to(REPO_ROOT)),
        "n_member_pairs": len(raw_rows),
        "n_random_pairs": args.n_random,
        "seed": args.seed,
        "git_commit": git_commit(),
    }
    with open(args.out / "metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)


if __name__ == "__main__":
    main()
