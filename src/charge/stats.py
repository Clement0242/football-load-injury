"""Statistics used by the analysis, kept small and tested."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def rmcorr(df: pd.DataFrame, x: str, y: str, subject: str = "player_id") -> dict:
    """Repeated-measures correlation (Bakdash & Marusich 2017).

    The within-subject association: each player's values are centred on her
    own mean, so differences *between* players (a bigger player covering more
    distance with a lower heart rate) cannot create a correlation. Degrees of
    freedom N - k - 1, as in the R package rmcorr.
    """
    d = df[[subject, x, y]].dropna()
    d = d[d.groupby(subject)[x].transform("size") >= 3]
    xc = d[x] - d.groupby(subject)[x].transform("mean")
    yc = d[y] - d.groupby(subject)[y].transform("mean")
    r = float(np.sum(xc * yc) / np.sqrt(np.sum(xc**2) * np.sum(yc**2)))
    dof = len(d) - d[subject].nunique() - 1
    t = r * np.sqrt(dof / max(1e-12, 1 - r**2))
    p = 2 * stats.t.sf(abs(t), dof)
    z, se = np.arctanh(r), 1 / np.sqrt(max(dof - 1, 1))
    lo, hi = np.tanh([z - 1.96 * se, z + 1.96 * se])
    return {"r": r, "ci_low": float(lo), "ci_high": float(hi), "p": float(p), "dof": int(dof),
            "n_obs": int(len(d)), "n_subjects": int(d[subject].nunique())}


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Agreement between two categorical sequences, corrected for chance."""
    m = a.notna() & b.notna()
    a, b = a[m].astype(str), b[m].astype(str)
    cats = sorted(set(a) | set(b))
    po = float(np.mean(a.to_numpy() == b.to_numpy()))
    pe = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney), ties counted as half."""
    y = np.asarray(y).astype(bool)
    pos, neg = score[y], score[~y]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def leave_one_player_out(df: pd.DataFrame, features: list[str], target: str,
                         group: str = "player_id", C: float = 1.0) -> np.ndarray:
    """Out-of-player predicted probabilities of a standardised L2 logistic model.

    Each player's days are predicted by a model that never saw that player:
    the honest way to ask whether an indicator would warn a staff about a
    player it has not been tuned on.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = np.full(len(df), np.nan)
    X, y, g = df[features].to_numpy(float), df[target].to_numpy(int), df[group].to_numpy()
    for player in np.unique(g):
        test = g == player
        if y[~test].sum() < 5:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
        model.fit(X[~test], y[~test])
        out[test] = model.predict_proba(X[test])[:, 1]
    return out


def cluster_bootstrap_auc(y: np.ndarray, score: np.ndarray, groups: np.ndarray, n_boot: int = 2000,
                          seed: int = 0) -> tuple[float, float]:
    """95 % interval of the AUC, resampling *players* (days of a player are not independent)."""
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    idx = {i: np.flatnonzero(groups == i) for i in ids}
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[i] for i in rng.choice(ids, ids.size, replace=True)])
        a = auc(y[pick], score[pick])
        if np.isfinite(a):
            vals.append(a)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def within_player_auc(y: np.ndarray, score: np.ndarray, groups: np.ndarray) -> float:
    """AUC computed only on pairs of days of the *same* player.

    Pooled AUC under leave-one-player-out is biased: each held-out player gets
    a model fitted on a slightly different base rate, so the ranking *between*
    players is an artefact (it can push the AUC far below 0.5). Comparing a
    player's pre-injury days with her own other days removes that, and is also
    the question a staff asks: is she in an unusual situation for her?
    """
    y = np.asarray(y).astype(bool)
    num = den = 0.0
    for gid in np.unique(groups):
        m = groups == gid
        pos, neg = score[m & y], score[m & ~y]
        if pos.size == 0 or neg.size == 0:
            continue
        ranks = stats.rankdata(np.concatenate([pos, neg]))
        num += ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2
        den += pos.size * neg.size
    return float(num / den) if den else float("nan")


def bootstrap_within_player_auc(y, score, groups, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95 % interval of within_player_auc, resampling players."""
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    idx = {i: np.flatnonzero(groups == i) for i in ids}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ids, ids.size, replace=True)
        sel = np.concatenate([idx[i] for i in pick])
        g = np.concatenate([np.full(idx[i].size, k) for k, i in enumerate(pick)])  # duplicates stay distinct
        a = within_player_auc(y[sel], score[sel], g)
        if np.isfinite(a):
            vals.append(a)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
