"""Record the git commit and environment of a run."""

import platform
import subprocess
import sys
from importlib import metadata

from spar_typicality.suites import REPO_ROOT

PACKAGES = ["numpy", "torch", "transformers", "plotly"]


def git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def git_is_dirty():
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() != ""


def environment():
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def provenance():
    return {
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "environment": environment(),
    }
