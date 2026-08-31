#!/usr/bin/env python3
"""Run the executable claim and frozen robustness contracts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    arguments = parser.parse_args()
    tests = [
        "tests/test_claim_contract.py",
        "tests/test_robustness_characterization.py",
        "tests/test_robustness_thresholds.py",
    ]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *tests],
        cwd=arguments.root.resolve(),
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
