#!/usr/bin/env python3
"""Generate or verify the canonical result inventory and repository safety scans."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import stat
import subprocess
import sys
import tomllib
from pathlib import Path, PurePosixPath

# Provenance of the frozen records. SOURCE_TREE is the private research repository the
# records were lifted from; SOURCE_COMMIT pins the revision. The two prefixes below are
# paths inside that tree, not paths inside this repository, and they are recorded so the
# provenance resolves if the source tree is ever opened.
SOURCE_TREE = "private research repository, not part of this repository"
SOURCE_COMMIT = "9e5a7dc84f185b8969fbb88887cbd19de32f5a77"
SOURCE_ARTIFACT_PREFIX = "light-cone/iclr2027/artifact/results/"
SOURCE_ROBUSTNESS_PREFIX = "light-cone/iclr2027/evidence/robustness-study/"

SOURCE_PATHS = {
    "results/tables/attention_seeds.csv": SOURCE_ARTIFACT_PREFIX + "attention_seeds.csv",
    "results/tables/consequence/acausal_consequence.json": SOURCE_ARTIFACT_PREFIX
    + "consequence/acausal_consequence.json",
    "results/tables/consequence/causal_reliance.json": SOURCE_ARTIFACT_PREFIX
    + "consequence/causal_reliance.json",
    "results/tables/consequence/false_attribution.json": SOURCE_ARTIFACT_PREFIX
    + "consequence/false_attribution.json",
    "results/tables/consequence/ood_consequence.json": SOURCE_ARTIFACT_PREFIX
    + "consequence/ood_consequence.json",
    "results/tables/deployed/deployed_fleet.csv": SOURCE_ARTIFACT_PREFIX + "deployed_fleet.csv",
    "results/tables/deployed/effective_signal_speed.json": SOURCE_ARTIFACT_PREFIX
    + "deployed/effective_signal_speed.json",
    "results/tables/deployed/real_emulator_audit.json": SOURCE_ARTIFACT_PREFIX
    + "deployed/real_emulator_audit.json",
    "results/tables/existence/corrected_existence_summary.json": SOURCE_ARTIFACT_PREFIX
    + "existence/corrected_existence_summary.json",
    "results/tables/existence/day1_summary.json": SOURCE_ARTIFACT_PREFIX
    + "existence/day1_summary.json",
    "results/tables/existence/dispersion_converged.json": SOURCE_ARTIFACT_PREFIX
    + "existence/dispersion_converged.json",
    "results/tables/mitigation/mitigation_2d_full.json": SOURCE_ARTIFACT_PREFIX
    + "pde/mitigation_2d_full.json",
    "results/tables/mitigation/mitigation_results.json": SOURCE_ARTIFACT_PREFIX
    + "mitigation_results.json",
    "results/tables/pde/burgers_summary.json": SOURCE_ARTIFACT_PREFIX + "pde/burgers_summary.json",
    "results/tables/prediction/predictive_summary.json": SOURCE_ARTIFACT_PREFIX
    + "prediction/predictive_summary.json",
    "results/tables/theory/multilayer_tail_verification.json": SOURCE_ARTIFACT_PREFIX
    + "theory/multilayer_tail_verification.json",
    "results/tables/theory/rollout_certificate_verification.json": SOURCE_ARTIFACT_PREFIX
    + "theory/rollout_certificate_verification.json",
    "results/tables/theory/sharp_cutoff_law_verification.json": SOURCE_ARTIFACT_PREFIX
    + "theory/sharp_cutoff_law_verification.json",
    "results/tables/robustness/mitigation_mid_scale_multiseed.json": SOURCE_ROBUSTNESS_PREFIX
    + "mitigation_seeds/mitigation_mid_seeds.json",
    "results/tables/robustness/mitigation_full_scale_mps_seeds_2_3.json": SOURCE_ROBUSTNESS_PREFIX
    + "mitigation_seeds/mitigation_full_seeds.json",
    "results/tables/robustness/fourcastnet_ic_region_dispersion.json": SOURCE_ROBUSTNESS_PREFIX
    + "ic_errorbars/ic_errorbars_fourcastnet.json",
    "results/tables/robustness/local_operator_ic_region_dispersion.json": SOURCE_ROBUSTNESS_PREFIX
    + "ic_errorbars/ic_errorbars_local.json",
    "results/tables/robustness/era5_ingest_validation.json": SOURCE_ROBUSTNESS_PREFIX
    + "ic_errorbars/fetch_validation.json",
}

# Records lifted from a later revision of the same private tree than SOURCE_COMMIT. That
# revision is not pinned here, so each one carries the later-revision label and the study it
# came from rather than a commit it was never part of.
LATER_SOURCE_TREE = "later revision of the private research repository, not part of this repository"

LATER_SOURCE_STUDIES = {
    "results/tables/consequence/matched_skill_shifted_band.json": "matched-skill shifted-band study",
    "results/tables/controls/locality_control_1d.json": "equal-width locality control, 1D",
    "results/tables/controls/locality_control_2d.json": "equal-width locality control, 2D",
    "results/tables/deployed/static_support_census.json": "census of released emulator graphs",
    "results/tables/era5/resource_accounting.json": "reanalysis reach-control resources",
    "results/tables/era5/horizon_by_capacity.json": "reanalysis reach-control capacity ladder",
    "results/tables/era5/six_hour_z500.json": "reanalysis reach-control capacity ladder",
    "results/tables/theory/higher_dimensional_tail.json": "higher-dimensional cutoff tail",
}

# Frozen records that postdate the pinned source commit, so they carry a record digest or
# an in-repository producer instead of a source-tree path.
DIGEST_PROVENANCE = {
    "results/tables/deployed/local_conv_reference.json": "source_record_sha256",
    "results/tables/deployed/compiled_reach_pangu.json": "checkpoint.sha256",
    "results/tables/deployed/deployed_response_census.json": "records.*.source_record_sha256",
}

GENERATED = {
    "results/tables/summary.csv": "scripts/replay/build_tables.py",
    "results/tables/robustness_summary.csv": "scripts/replay/build_tables.py",
    "results/tables/claim_scope.json": "scripts/replay/build_tables.py",
    "results/figures/fig_attention_window_sweep.pdf": "scripts/replay/build_figures.py",
    "results/figures/fig_consequence_and_mitigation.pdf": "scripts/replay/build_figures.py",
    "results/figures/fig_theory_checks.pdf": "scripts/replay/build_figures.py",
}

REGENERATED = set(GENERATED) | {
    "results/tables/theory/sharp_cutoff_law_verification.json",
    "results/tables/theory/rollout_certificate_verification.json",
}

GENERATOR_COVERAGE = [
    {
        "source_path": SOURCE_ARTIFACT_PREFIX.replace("results/", "experiments/")
        + "make_attention_sweep.py",
        "source_commit": SOURCE_COMMIT,
        "source_git_blob": "02d742332641f0f8aaaa390d797d319c44c97362",
        "source_sha256": "e60fdb3c43f4e4de8927754bd42b43df4d0dac6d61a6a1a70694f94be2415fe9",
        "disposition": "consolidated_generator_retained",
        "retained_path": "scripts/replay/build_figures.py",
        "reason": "The canonical attention figure is rendered from the retained compact CSV.",
    },
    {
        "source_path": SOURCE_ARTIFACT_PREFIX.replace("results/", "experiments/")
        + "make_cone_diagnostic.py",
        "source_commit": SOURCE_COMMIT,
        "source_git_blob": "b622e78ed489ead4a362611af467308dd92b0731",
        "source_sha256": "32936bc4c5f41cedee3396d16a7ab9b8bc766d591cc38af9a27f60c3515b481d",
        "disposition": "frozen_record_only",
        "retained_path": None,
        "reason": (
            "Frozen-record provenance is retained because the source renderer requires "
            "excluded response arrays and cross-domain records; the compact headline values "
            "remain exact records from the declared source commit and blob."
        ),
    },
    {
        "source_path": SOURCE_ARTIFACT_PREFIX.replace("results/", "experiments/")
        + "make_consequence_fix.py",
        "source_commit": SOURCE_COMMIT,
        "source_git_blob": "493c9fa569433606e79809ce540265ec47b52b76",
        "source_sha256": "a9673b2f1dc12fa7f765d69b4909a48e111b5bad2ae0e2c254c23ab0b1478d56",
        "disposition": "consolidated_generator_retained",
        "retained_path": "scripts/replay/build_figures.py",
        "reason": (
            "The compact primary consequence and mitigation panels are retained; "
            "the source supplemental field renderer requires an excluded response array."
        ),
    },
    {
        "source_path": SOURCE_ARTIFACT_PREFIX.replace("results/", "experiments/")
        + "make_table1.py",
        "source_commit": SOURCE_COMMIT,
        "source_git_blob": "4715e4df59fe7ee2482c5eb52419a3a50339ed04",
        "source_sha256": "a1aaa21fe38778ae4a4145416f4c64c33a87259f3d06bda64c21e6f4d97d05a5",
        "disposition": "frozen_record_only",
        "retained_path": None,
        "reason": (
            "Frozen-record provenance is retained for the legacy renderer; the exact "
            "deployed-fleet CSV remains in the declared source commit and blob, while its "
            "PNG and manuscript table outputs are outside the code-only result inventory."
        ),
    },
    {
        "source_path": SOURCE_ARTIFACT_PREFIX.replace("results/", "experiments/")
        + "make_theory.py",
        "source_commit": SOURCE_COMMIT,
        "source_git_blob": "a9394a438a5ebb538dd48fe6bf347d78055a160d",
        "source_sha256": "dfd965dee4234137812928626675b6db3628530fc2c776214cb781abb2d90ada",
        "disposition": "consolidated_generator_retained",
        "retained_path": "scripts/replay/build_figures.py",
        "reason": "The canonical theory figure is regenerated from retained strict JSON records.",
    },
    {
        "source_path": SOURCE_ARTIFACT_PREFIX.replace("results/", "experiments/")
        + "run_predictive_battery.py",
        "source_commit": SOURCE_COMMIT,
        "source_git_blob": "b922659b9468af11d8f66920b11ec4f90de9a189",
        "source_sha256": "e5747e4c4748e612e31ccb73e9a16a2915a9fd7e23dad40fa91c0a35c95117e7",
        "disposition": "secondary_generator_retained",
        "retained_path": "scripts/experiments/run_predictive_battery.py",
        "reason": (
            "The scientific protocol and constants are retained with package imports and "
            "explicit output and checkpoint directories."
        ),
    },
]

SEEDS = {
    "results/tables/attention_seeds.csv": [0, 1, 2],
    "results/tables/existence/day1_summary.json": [0, 1, 2],
    "results/tables/existence/corrected_existence_summary.json": [0],
    "results/tables/mitigation/mitigation_2d_full.json": [0, 1],
    "results/tables/mitigation/mitigation_results.json": [0, 1],
    "results/tables/prediction/predictive_summary.json": [0, 1],
    "results/tables/robustness/mitigation_mid_scale_multiseed.json": [0, 1, 2, 3, 4],
    "results/tables/robustness/mitigation_full_scale_mps_seeds_2_3.json": [2, 3],
    "results/tables/robustness/local_operator_ic_region_dispersion.json": [0],
    "results/tables/theory/sharp_cutoff_law_verification.json": [0],
    "results/tables/theory/rollout_certificate_verification.json": [0],
}

GENERIC_PATTERNS = {
    "absolute_home_path": re.compile(r"/(?:Users|home)/[^/\s]+/"),
    "cluster_mount_path": re.compile(r"/(?:mnt|scratch|gpfs|lustre|workspace)/[^\s]+"),
    "email_address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "orcid": re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "local_hostname": re.compile(r"\b[A-Za-z0-9._-]+\.local\b"),
}

TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: Path):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open(encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON value {value!r} in {path}")
            ),
        )


def safe_path(relative: str, *, canonical_result: bool = False) -> None:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError(f"unsafe relative path: {relative}")
    if canonical_result and any(part in {"logs", "checkpoints", "cache"} for part in path.parts):
        raise ValueError(f"excluded result path class: {relative}")
    if canonical_result and any(word in path.name.lower() for word in ("final", "new", "pending")):
        raise ValueError(f"lifecycle word in canonical result path: {relative}")


def classify(path: str) -> tuple[str, str, str]:
    if path in GENERATED:
        role = "derived_figure" if path.startswith("results/figures/") else "derived_table"
        if path.endswith("claim_scope.json"):
            role = "claim_scope_record"
        return role, "derived", "regenerated_in_default_replay"
    if "/robustness/" in path:
        source = "public_reanalysis" if "mitigation_" not in path else "synthetic"
        return "robustness_record", source, "compared_from_frozen_evidence"
    if "/era5/" in path:
        source = (
            "architecture_accounting"
            if path.endswith("resource_accounting.json")
            else "public_reanalysis"
        )
        return "canonical_run_record", source, "compared_from_frozen_evidence"
    if "/theory/" in path:
        if path.endswith("multilayer_tail_verification.json"):
            return "canonical_run_record", "synthetic", "secondary_external_reproduction"
        replay = (
            "regenerated_in_default_replay"
            if path in REGENERATED
            else "compared_from_frozen_evidence"
        )
        return "canonical_run_record", "closed_form", replay
    if path.endswith("deployed/compiled_reach_pangu.json"):
        return (
            "canonical_run_record",
            "public_checkpoint",
            "regenerated_on_request_from_external_checkpoint",
        )
    if path.endswith("deployed/local_conv_reference.json"):
        return "canonical_run_record", "public_reanalysis", "compared_from_frozen_evidence"
    if path.endswith("deployed/deployed_fleet.csv"):
        return (
            "canonical_run_record",
            "public_checkpoints_and_local_reference",
            "compared_from_frozen_evidence",
        )
    if "/deployed/" in path:
        return "canonical_run_record", "public_checkpoint", "compared_from_frozen_evidence"
    if path.endswith("prediction/predictive_summary.json"):
        return "canonical_run_record", "synthetic", "secondary_external_reproduction"
    return "canonical_run_record", "synthetic", "compared_from_frozen_evidence"


def producer(path: str) -> str:
    if path in GENERATED:
        return GENERATED[path]
    if path.endswith("deployed/compiled_reach_pangu.json"):
        return "tools/compiled_reach/cone_reach.py"
    if path.endswith("sharp_cutoff_law_verification.json"):
        return "scripts/theory/verify_sharp_cutoff_law.py"
    if path.endswith("rollout_certificate_verification.json"):
        return "scripts/theory/verify_rollout_certificate.py"
    if path.endswith("multilayer_tail_verification.json"):
        return "scripts/theory/verify_multilayer_tail.py"
    if path.endswith("prediction/predictive_summary.json"):
        return "scripts/experiments/run_predictive_battery.py"
    return "external_frozen_record"


def config_identifier(path: str) -> str:
    if path in GENERATED:
        return "canonical_replay"
    if "/robustness/" in path:
        return "banked_robustness"
    return "source_record"


def result_paths(root: Path) -> list[Path]:
    files = []
    for directory in (root / "results" / "tables", root / "results" / "figures"):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        files.extend(path for path in directory.rglob("*") if path.is_file())
    return sorted(path for path in files if path.name != "manifest.json")


def build_manifest(root: Path) -> dict:
    paths = result_paths(root)
    relative_paths = [path.relative_to(root).as_posix() for path in paths]
    declared = (
        set(SOURCE_PATHS) | set(LATER_SOURCE_STUDIES) | set(DIGEST_PROVENANCE) | set(GENERATED)
    )
    if set(relative_paths) != declared:
        missing = sorted(declared - set(relative_paths))
        extra = sorted(set(relative_paths) - declared)
        raise ValueError(f"result declaration mismatch; missing={missing}; extra={extra}")
    entries = []
    for path, relative in zip(paths, relative_paths, strict=True):
        safe_path(relative, canonical_result=True)
        if path.is_symlink() or path.stat().st_size == 0:
            raise ValueError(f"invalid result file: {relative}")
        if path.suffix == ".json":
            strict_json(path)
        role, source_classification, replay_classification = classify(relative)
        source_path = SOURCE_PATHS.get(relative)
        later_study = LATER_SOURCE_STUDIES.get(relative)
        transformation = None
        if relative.endswith("sharp_cutoff_law_verification.json"):
            transformation = "undefined zero-edge ratio encoded as null; finite leaves unchanged"
        entries.append(
            {
                "path": relative,
                "role": role,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "producer": producer(relative),
                "seed_inventory": SEEDS.get(relative, []),
                "config_identifier": config_identifier(relative),
                "source_classification": source_classification,
                "replay_classification": replay_classification,
                "expected_schema_version": 1,
                "comparison_mode": "exact_sha256",
                "tolerance": None,
                "provenance_source_tree": (
                    SOURCE_TREE if source_path else LATER_SOURCE_TREE if later_study else None
                ),
                "provenance_source_commit": SOURCE_COMMIT if source_path else None,
                "provenance_original_path": source_path,
                "provenance_source_study": later_study,
                "provenance_digest_field": DIGEST_PROVENANCE.get(relative),
                "transformation": transformation,
            }
        )
    return {
        "schema_version": 1,
        "manifest_scope": "all canonical files under results/tables and results/figures",
        "excluded_self": "results/tables/manifest.json",
        "generator_coverage": GENERATOR_COVERAGE,
        "entries": entries,
    }


def verify_manifest(root: Path) -> int:
    actual = build_manifest(root)
    print(f"verified result inventory: {len(actual['entries'])} files")
    return len(actual["entries"])


def verify_environment(root: Path) -> None:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    if project["project"]["requires-python"] != ">=3.11,<3.12":
        raise ValueError("Python minor-series contract drifted")
    if sys.version_info[:3] != (3, 11, 15):
        raise ValueError(f"replay requires CPython 3.11.15, found {sys.version.split()[0]}")
    dependencies = dict(
        specification.split("==", 1) for specification in project["project"]["dependencies"]
    )
    for distribution, expected in dependencies.items():
        actual = importlib.metadata.version(distribution)
        if actual != expected:
            raise ValueError(f"{distribution} version {actual} != {expected}")
    print(f"verified environment: CPython 3.11.15; {len(dependencies)} pinned packages")


def tracked_paths(root: Path) -> list[PurePosixPath]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode("utf-8")
    return [PurePosixPath(item) for item in output.rstrip("\0").split("\0") if item]


def scan_repository(root: Path) -> int:
    tracked = tracked_paths(root)
    findings = []
    generic_counts = {name: 0 for name in GENERIC_PATTERNS}
    for relative in tracked:
        safe_path(relative.as_posix())
        path = root / relative
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            findings.append((relative.as_posix(), "non_regular_file"))
            continue
        if path.stat().st_size == 0:
            findings.append((relative.as_posix(), "zero_byte_file"))
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            ".gitattributes",
            ".gitignore",
            "LICENSE",
            "uv.lock",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in GENERIC_PATTERNS.items():
            hits = list(pattern.finditer(text))
            if name == "email_address" and relative.as_posix() == "LICENSE":
                hits = []
            generic_counts[name] += len(hits)
            if hits:
                findings.append((relative.as_posix(), name))

    forbidden_names = {
        "identity_tokens.txt",
        ".env",
        "CITATION.cff",
    }
    for relative in tracked:
        if relative.name in forbidden_names:
            findings.append((relative.as_posix(), "forbidden_identity_or_secret_file"))
    if findings:
        rendered = ", ".join(f"{path}:{kind}" for path, kind in sorted(set(findings)))
        raise ValueError(f"repository scan findings: {rendered}")
    print(
        f"repository scan passed: {len(tracked)} tracked files; "
        f"{sum(generic_counts.values())} sensitive-pattern hits"
    )
    return len(tracked)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--scan", action="store_true")
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    verify_environment(root)
    verify_manifest(root)
    if arguments.scan:
        scan_repository(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
