#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"

uv sync --locked

if [[ ! -x "${PYTHON}" ]]; then
  printf 'error: Python interpreter is not executable: %s\n' "${PYTHON}" >&2
  printf 'provision the locked environment before replay\n' >&2
  exit 2
fi
if ! git -C "${ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  printf 'error: replay requires the repository Git worktree for source-contract checks\n' >&2
  exit 2
fi

BEFORE_STATUS="$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)"
if [[ -n "${BEFORE_STATUS}" ]]; then
  printf 'error: replay requires a clean repository tree\n%s\n' "${BEFORE_STATUS}" >&2
  exit 2
fi

mkdir -p "${ROOT}/.cache"
WORK="$(mktemp -d "${ROOT}/.cache/replay.XXXXXX")"
cleanup() {
  if [[ "${WORK}" == "${ROOT}/.cache/replay."* ]]; then
    rm -rf "${WORK}"
    rmdir "${ROOT}/.cache" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
mkdir -p "${WORK}/home" "${WORK}/cache" "${WORK}/mpl" "${WORK}/tmp" "${WORK}/torch"

export SOURCE_DATE_EPOCH=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export HOME="${WORK}/home"
export XDG_CACHE_HOME="${WORK}/cache"
export MPLCONFIGDIR="${WORK}/mpl"
export TORCH_HOME="${WORK}/torch"
export TMPDIR="${WORK}/tmp"
export UV_OFFLINE=1
export UV_NO_SYNC=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export TZ=UTC
export LC_ALL=C

cd "${ROOT}"

"${PYTHON}" scripts/replay/verify_evidence.py --scan

"${PYTHON}" -m pytest -q
"${PYTHON}" scripts/replay/verify_claims.py
"${PYTHON}" scripts/theory/verify_multilayer_tail.py \
  --input results/tables/theory/multilayer_tail_verification.json

GENERATED="${WORK}/generated"
mkdir -p "${GENERATED}/tables/theory" "${GENERATED}/figures"
"${PYTHON}" scripts/theory/verify_sharp_cutoff_law.py \
  --output "${GENERATED}/tables/theory/sharp_cutoff_law_verification.json"
"${PYTHON}" scripts/theory/verify_rollout_certificate.py \
  --output "${GENERATED}/tables/theory/rollout_certificate_verification.json"
"${PYTHON}" scripts/replay/build_tables.py \
  --input-dir results/tables \
  --output-dir "${GENERATED}/tables"
"${PYTHON}" scripts/replay/build_figures.py \
  --input-dir results/tables \
  --output-dir "${GENERATED}/figures"

compare_exact() {
  local generated="$1"
  local canonical="$2"
  if ! cmp --silent "${generated}" "${canonical}"; then
    printf 'error: regenerated output differs: %s\n' "${canonical}" >&2
    exit 1
  fi
  printf '%s  %s\n' "$(shasum -a 256 "${canonical}" | awk '{print $1}')" "${canonical}"
}

compare_json_numeric() {
  local generated="$1"
  local canonical="$2"
  if ! "${PYTHON}" - "${generated}" "${canonical}" <<'PY'
import json
import math
import sys


def close(observed, expected):
    if isinstance(expected, dict):
        return observed.keys() == expected.keys() and all(
            close(observed[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            close(observed_item, expected_item)
            for observed_item, expected_item in zip(observed, expected, strict=True)
        )
    if isinstance(expected, float):
        return math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)
    return observed == expected


with open(sys.argv[1], encoding="utf-8") as handle:
    observed = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    expected = json.load(handle)
raise SystemExit(0 if close(observed, expected) else 1)
PY
  then
    printf 'error: regenerated numeric output differs: %s\n' "${canonical}" >&2
    exit 1
  fi
  printf '%s  %s\n' "$(shasum -a 256 "${canonical}" | awk '{print $1}')" "${canonical}"
}

compare_json_numeric \
  "${GENERATED}/tables/theory/sharp_cutoff_law_verification.json" \
  "results/tables/theory/sharp_cutoff_law_verification.json"
compare_json_numeric \
  "${GENERATED}/tables/theory/rollout_certificate_verification.json" \
  "results/tables/theory/rollout_certificate_verification.json"
for name in summary.csv robustness_summary.csv claim_scope.json; do
  compare_exact "${GENERATED}/tables/${name}" "results/tables/${name}"
done
for name in \
  fig_attention_window_sweep.pdf \
  fig_consequence_and_mitigation.pdf \
  fig_theory_checks.pdf; do
  compare_exact "${GENERATED}/figures/${name}" "results/figures/${name}"
done

"${PYTHON}" scripts/replay/verify_evidence.py
shasum -a 256 -c results/CHECKSUMS.txt
AFTER_STATUS="$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)"
if [[ "${AFTER_STATUS}" != "${BEFORE_STATUS}" ]]; then
  printf 'error: replay mutated the repository tree\n' >&2
  diff -u <(printf '%s\n' "${BEFORE_STATUS}") <(printf '%s\n' "${AFTER_STATUS}") || true
  exit 1
fi

printf 'replay passed: 40 canonical records, 8 verified regenerated outputs, 0 required skips\n'
