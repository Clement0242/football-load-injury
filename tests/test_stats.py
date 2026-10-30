import numpy as np
import pandas as pd
import pytest

from charge.stats import auc, cohen_kappa, rmcorr, within_player_auc


def test_rmcorr_ignores_between_subject_differences():
    rng = np.random.default_rng(0)
    rows = []
    for s in range(20):
        base = rng.normal(0, 10)          # big between-subject offset on both variables
        x = rng.normal(0, 1, 30)
        y = -0.6 * x + rng.normal(0, 0.8, 30)
        rows += [(s, base + a, base + b) for a, b in zip(x, y)]
    d = pd.DataFrame(rows, columns=["player_id", "x", "y"])
    assert np.corrcoef(d.x, d.y)[0, 1] > 0.8          # pooled correlation is misleading
    r = rmcorr(d, "x", "y")
    assert r["r"] == pytest.approx(-0.6, abs=0.08)    # within-player one is recovered
    assert r["dof"] == 600 - 20 - 1 and r["ci_low"] < r["r"] < r["ci_high"]


def test_kappa():
    a = pd.Series(list("aabbcc"))
    assert cohen_kappa(a, a) == pytest.approx(1.0)
    assert cohen_kappa(a, pd.Series(list("bbccaa"))) < 0


def test_auc():
    assert auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_within_player_auc_ignores_between_player_offsets():
    rng = np.random.default_rng(1)
    y, s, g = [], [], []
    for p in range(30):
        offset = -3.0 * p                      # players with more injuries score lower overall
        lab = rng.random(200) < 0.02 * (p + 1)
        y += list(lab); s += list(offset + lab * 1.0 + rng.normal(0, 1, 200)); g += [p] * 200
    y, s, g = np.array(y), np.array(s), np.array(g)
    assert auc(y, s) < 0.4                       # pooled: fooled by the offsets
    assert within_player_auc(y, s, g) > 0.7      # within player: the real signal
