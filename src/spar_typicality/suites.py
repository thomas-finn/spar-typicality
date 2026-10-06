"""Load a dataset suite: a folder of prompt set CSVs in data/processed."""

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


class Dataset:
    """One prompt set of a suite, for example 'rosch_example'."""

    def __init__(self, name, rows):
        self.name = name
        self.rows = rows

    def prompts(self):
        return [row["prompt"] for row in self.rows]


def parse_optional_float(text):
    if text == "":
        return None
    return float(text)


def parse_optional_int(text):
    if text == "":
        return None
    return int(text)


def read_dataset(path):
    """Read one prompt set CSV. Empty cells become None."""
    rows = []
    with open(path, newline="") as file:
        for raw_row in csv.DictReader(file):
            row = {
                "prompt": raw_row["prompt"],
                "item": raw_row["item"],
                "category": raw_row["category"] or None,
                "typicality_rating_raw": parse_optional_float(
                    raw_row["typicality_rating_raw"]
                ),
                "typicality_rating_normalised": parse_optional_float(
                    raw_row["typicality_rating_normalised"]
                ),
                "is_member": parse_optional_int(raw_row["is_member"]),
            }
            rows.append(row)
    return Dataset(Path(path).stem, rows)


def load_suite(suite, processed_dir=PROCESSED_DIR, dataset_names=None):
    """Return the datasets of a suite, sorted by name.

    If `dataset_names` is given, only load those datasets.
    """
    suite_dir = Path(processed_dir) / suite
    paths = sorted(suite_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError("No CSV files in " + str(suite_dir))
    datasets = []
    for path in paths:
        if dataset_names is not None and path.stem not in dataset_names:
            continue
        datasets.append(read_dataset(path))
    if dataset_names is not None:
        found = {dataset.name for dataset in datasets}
        missing = set(dataset_names) - found
        if missing:
            raise ValueError("Datasets not found in suite: " + str(sorted(missing)))
    return datasets
