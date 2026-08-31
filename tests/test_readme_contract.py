"""Reviewer landing-page and command contracts.

The numeric contract here is a traceability rule, not a list of literals: every number the
landing page states in its headline section must equal a value carried by one of the compact
records below, at the precision the page quotes it to. Correcting a record therefore corrects
what the page is allowed to say, and a number with no record behind it fails.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
TABLES = ROOT / "results" / "tables"

REQUIRED_HEADINGS = [
    "# The Learned Light Cone of Scientific Emulators",
    "## Headline results",
    "## Install and reproduce",
    "## Outputs",
    "## Repository map",
    "## License",
]

QUOTABLE_CSV = [
    "summary.csv",
    "robustness_summary.csv",
    "deployed/deployed_fleet.csv",
]
QUOTABLE_JSON = [
    "claim_scope.json",
    "deployed/compiled_reach_pangu.json",
    "deployed/deployed_response_census.json",
    "deployed/local_conv_reference.json",
]
NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?![\w])")


def readme() -> str:
    return README.read_text(encoding="utf-8")


def section(text: str, heading: str) -> str:
    body = text.split(heading, 1)[1]
    following = re.search(r"\n## ", body)
    return body[: following.start()] if following else body


def quotable_values() -> list[float]:
    values: list[float] = []
    for name in QUOTABLE_CSV:
        with (TABLES / name).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for cell in row.values():
                    try:
                        values.append(float(cell))
                    except (TypeError, ValueError):
                        continue

    def walk(node) -> None:
        if isinstance(node, dict):
            for item in node.values():
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, bool):
            return
        elif isinstance(node, int | float):
            values.append(float(node))

    for name in QUOTABLE_JSON:
        walk(json.loads((TABLES / name).read_text(encoding="utf-8")))
    assert values
    return values


def traces_to_a_record(token: str, values: list[float]) -> bool:
    quoted = float(token.replace(",", ""))
    decimals = len(token.split(".")[1]) if "." in token else 0
    for value in values:
        for scaled in (value, 100.0 * value):
            if round(scaled, decimals) == quoted:
                return True
    return False


def test_readme_has_required_order_and_concise_length():
    text = readme()
    lines = text.splitlines()
    assert 40 <= len(lines) <= 200
    positions = [text.index(heading) for heading in REQUIRED_HEADINGS]
    assert positions == sorted(positions)


def test_only_canonical_setup_and_replay_commands_are_advertised():
    text = readme()
    assert text.count("bash reproduce.sh") == 1
    assert "pip install" not in text
    assert "conda " not in text
    assert "requirements.txt" not in text
    assert "environment.yml" not in text
    assert (ROOT / "reproduce.sh").is_file()


def test_headline_is_brief_and_every_number_traces_to_a_record():
    text = readme()
    headline = section(text, "## Headline results")
    assert headline.count("\n- ") <= 3
    values = quotable_values()
    quoted = NUMBER.findall(headline)
    assert quoted
    untraceable = [token for token in quoted if not traces_to_a_record(token, values)]
    assert not untraceable, untraceable
    with (TABLES / "deployed" / "deployed_fleet.csv").open(encoding="utf-8", newline="") as handle:
        fleet = list(csv.DictReader(handle))
    released = [row for row in fleet if row["record_role"] == "released_checkpoint"]
    local_references = [row for row in fleet if row["record_role"] == "local_reference"]
    assert f"Across {len(released)} released checkpoints" in headline
    assert len(local_references) == 1
    assert "A separate measured local-convolution reference" in headline


def test_every_headline_bullet_cites_a_record_that_exists():
    bullets = section(readme(), "## Headline results").split("\n- ")[1:]
    assert bullets
    for bullet in bullets:
        cited = re.findall(r"`(results/[^`]+)`", bullet)
        assert cited, bullet
        for target in cited:
            assert (ROOT / target).is_file(), target


def test_internal_vocabulary_stays_out_and_paths_resolve():
    text = readme()
    assert "provenance" not in text.lower()
    assert "audit" not in text.lower()
    assert "round " not in text.lower()
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
        if "://" not in target and not target.startswith("#"):
            assert (ROOT / target).exists(), target
