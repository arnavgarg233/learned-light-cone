"""Does cone geometry predict rollout error beyond one-step accuracy?

Mode budget changes both the cone and one-step error, so the test is incremental:
regress rollout error on one-step RMSE, then check whether adding cone features
improves held-out R^2. Provides nested regression with k-fold cross-validation,
partial Spearman correlation, and AUROC helpers. NumPy and SciPy only.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import rankdata, spearmanr


# --------------------------------------------------------------------------- #
# low-level OLS helpers (standardize -> add intercept -> least squares)
# --------------------------------------------------------------------------- #
def _stack(features: dict[str, np.ndarray], keys: list[str]) -> np.ndarray:
    """Column-stack the named feature vectors into a design matrix (n, len(keys))."""
    if len(keys) == 0:
        return np.empty((0, 0))
    cols = [np.asarray(features[k], dtype=np.float64).reshape(-1) for k in keys]
    n = cols[0].shape[0]
    for c in cols:
        if c.shape[0] != n:
            raise ValueError("all features must share the same length")
    return np.column_stack(cols)


def _standardize(X: np.ndarray, mu: np.ndarray | None = None, sd: np.ndarray | None = None):
    """Z-score columns. If mu/sd supplied (train stats), reuse them (no leakage)."""
    if X.size == 0:
        return X, np.zeros(0), np.ones(0)
    if mu is None:
        mu = X.mean(axis=0)
    if sd is None:
        sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)  # guard constant columns
    return (X - mu) / sd, mu, sd


def _ols_fit(X: np.ndarray, y: np.ndarray):
    """Fit y ~ [1, X] by least squares. Returns coefficient vector (incl. intercept)."""
    n = y.shape[0]
    A = np.column_stack([np.ones(n), X]) if X.size else np.ones((n, 1))
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """Coefficient of determination 1 - SS_res / SS_tot (variance w.r.t. y's own mean)."""
    y = np.asarray(y, dtype=np.float64)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return 0.0
    ss_res = float(((y - yhat) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def _fit_predict_insample(features: dict, keys: list[str], y: np.ndarray) -> np.ndarray:
    """In-sample predictions of y from the named keys (standardized + intercept)."""
    n = y.shape[0]
    X = _stack(features, keys)
    if X.size == 0:
        return np.full(n, float(np.mean(y)))  # intercept-only -> grand mean
    Xs, _, _ = _standardize(X)
    beta = _ols_fit(Xs, y)
    A = np.column_stack([np.ones(n), Xs])
    return A @ beta


def _cv_predict(
    features: dict, keys: list[str], y: np.ndarray, n_folds: int, seed: int
) -> np.ndarray:
    """Out-of-fold predictions of y. Standardization stats are fit on the TRAIN
    fold only and applied to the held-out fold (no information leakage)."""
    n = y.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, min(n_folds, n))
    oof = np.empty(n, dtype=np.float64)
    X = _stack(features, keys)
    for fold in folds:
        test_idx = fold
        train_idx = np.setdiff1d(np.arange(n), test_idx, assume_unique=False)
        ytr = y[train_idx]
        if X.size == 0:
            oof[test_idx] = float(np.mean(ytr))  # intercept-only baseline = train mean
            continue
        Xtr, mu, sd = _standardize(X[train_idx])
        beta = _ols_fit(Xtr, ytr)
        Xte, _, _ = _standardize(X[test_idx], mu, sd)
        Ate = np.column_stack([np.ones(len(test_idx)), Xte])
        oof[test_idx] = Ate @ beta
    return oof


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def nested_regression(
    features: dict[str, np.ndarray],
    target: np.ndarray,
    base_keys: list[str],
    add_keys: list[str],
    n_folds: int = 5,
    seed: int = 0,
) -> dict:
    """Compare a base OLS model against base + added features.

    Fits target ~ base_keys (nuisance model) and target ~ base_keys + add_keys
    (full model). Features are z-scored and an intercept is included. delta_r2 is
    the incremental variance explained by the added cone features.

    The cross-validated variant is the decisive one: in-sample R^2 can only
    increase when regressors are added, but held-out delta_r2_cv > 0 means the
    cone features genuinely predict E_H beyond one-step accuracy on unseen
    models. delta_r2_cv can go negative (full model overfits) -- that is the
    null result and we report it as-is.

    Returns {r2_base, r2_full, delta_r2, r2_base_cv, r2_full_cv, delta_r2_cv, n}.
    """
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    n = y.shape[0]
    full_keys = list(base_keys) + [k for k in add_keys if k not in base_keys]

    # in-sample
    yhat_base = _fit_predict_insample(features, base_keys, y)
    yhat_full = _fit_predict_insample(features, full_keys, y)
    r2_base = _r2(y, yhat_base)
    r2_full = _r2(y, yhat_full)

    # cross-validated (out-of-fold)
    oof_base = _cv_predict(features, base_keys, y, n_folds, seed)
    oof_full = _cv_predict(features, full_keys, y, n_folds, seed)
    r2_base_cv = _r2(y, oof_base)
    r2_full_cv = _r2(y, oof_full)

    return {
        "r2_base": float(r2_base),
        "r2_full": float(r2_full),
        "delta_r2": float(r2_full - r2_base),
        "r2_base_cv": float(r2_base_cv),
        "r2_full_cv": float(r2_full_cv),
        "delta_r2_cv": float(r2_full_cv - r2_base_cv),
        "n": int(n),
    }


def _residualize(x: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    """Residual of x after OLS regression on the control variables (+intercept)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if not controls:
        return x - x.mean()
    C = np.column_stack([np.asarray(c, dtype=np.float64).reshape(-1) for c in controls])
    Cs, _, _ = _standardize(C)
    beta = _ols_fit(Cs, x)
    A = np.column_stack([np.ones(x.shape[0]), Cs])
    return x - A @ beta


def partial_spearman(x: np.ndarray, y: np.ndarray, controls: list[np.ndarray]):
    """Spearman correlation of x and y after partialling out the controls.

    We regress out the controls *linearly* from both x and y (intercept +
    standardized controls), then take the Spearman rank correlation of the two
    residual vectors. This isolates the monotone association of, e.g., cone
    leakage with rollout error that is not already explained by one-step RMSE and
    parameter count. Returns (rho, p). With <=2 samples or a degenerate residual,
    returns (0.0, 1.0).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape[0] < 3:
        return 0.0, 1.0
    rx = _residualize(x, controls)
    ry = _residualize(y, controls)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return 0.0, 1.0
    rho, p = spearmanr(rx, ry)
    if np.isnan(rho):
        return 0.0, 1.0
    return float(rho), float(p)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC for a binary classifier via the Mann-Whitney rank statistic.

    AUC = P(score(positive) > score(negative)), estimated as

        (sum of ranks of positives - n_pos*(n_pos+1)/2) / (n_pos * n_neg),

    with average ranks for ties (so tied pos/neg pairs count as 0.5). Higher
    score is assumed to indicate the positive (blow-up) class. Degenerate cases
    (all one class) return 0.5. No sklearn.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(int)
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(scores)  # average ranks handle ties
    sum_pos = float(ranks[pos].sum())
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def dominance(
    features: dict[str, np.ndarray], target: np.ndarray, max_features: int = 5
) -> dict[str, float]:
    """Dominance analysis: each feature's average marginal R^2 contribution.

    For every subset S of the other features, we compute the increase in
    in-sample R^2 from adding feature f to S, then average over all subsets
    (equal weight per subset size, then averaged over sizes) -- the Shapley-style
    decomposition of total model R^2. This attributes shared variance fairly
    among collinear predictors (e.g. Lambda and one-step RMSE), unlike a single
    full-model coefficient. Capped at max_features (<=5) since the subset sum is
    exponential. Returns {feature_name: average_marginal_r2}.
    """
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    keys = list(features.keys())
    if len(keys) > max_features:
        raise ValueError(f"dominance capped at {max_features} features, got {len(keys)}")

    # cache R^2 for every subset
    r2_cache: dict[frozenset, float] = {}

    def subset_r2(subset: tuple[str, ...]) -> float:
        fs = frozenset(subset)
        if fs in r2_cache:
            return r2_cache[fs]
        yhat = _fit_predict_insample(features, list(subset), y)
        val = _r2(y, yhat)
        r2_cache[fs] = val
        return val

    out: dict[str, float] = {}
    for f in keys:
        others = [k for k in keys if k != f]
        # average over subset SIZES, then within each size over the subsets:
        # this is the Shapley weighting and sums the per-feature shares to total R^2.
        size_means = []
        for r in range(len(others) + 1):
            marginals = []
            for subset in itertools.combinations(others, r):
                base = subset_r2(subset)
                full = subset_r2(subset + (f,))
                marginals.append(full - base)
            size_means.append(float(np.mean(marginals)))
        out[f] = float(np.mean(size_means))
    return out
