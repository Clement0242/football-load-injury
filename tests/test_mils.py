import numpy as np
import pandas as pd
import pytest

from charge.mils import _em, _forward_backward, _emission, ewma, fit_mils, prepare, redi


def _simulate(n_athletes=12, n_days=200, seed=0, p_missing=0.2):
    rng = np.random.default_rng(seed)
    A = np.array([[0.85, 0.10, 0.05], [0.10, 0.80, 0.10], [0.05, 0.15, 0.80]])
    mu, sd = np.array([200.0, 450.0, 800.0]), np.array([40.0, 60.0, 90.0])
    rows = []
    for a in range(n_athletes):
        s = rng.integers(3)
        for d in range(n_days):
            s = rng.choice(3, p=A[s])
            load = rng.normal(mu[s], sd[s]) if rng.random() > p_missing else 0.0
            rows.append((f"A{a}", pd.Timestamp("2021-01-01") + pd.Timedelta(days=d), load, s))
    return pd.DataFrame(rows, columns=["id", "date", "load", "true_state"]), A


def test_prepare_follows_mils():
    d = pd.DataFrame({"id": ["a"] * 5, "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-02",
                                                               "2021-01-05", "2021-01-06"]),
                      "load": [0, 100, 50, 300, 0]})
    p = prepare(d)
    assert p["date"].min() == pd.Timestamp("2021-01-02")  # leading rest day trimmed
    assert p["date"].max() == pd.Timestamp("2021-01-05")  # trailing rest day trimmed
    assert len(p) == 4 and p["load"].isna().sum() == 2    # missing days added, 0 = NA
    assert p.loc[0, "load"] == 150                         # same-day sessions summed
    assert p["load_scaled"].mean() == pytest.approx(0)


def test_em_matches_hmmlearn_on_complete_data():
    hmm = pytest.importorskip("hmmlearn.hmm")
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(m, 0.5, 60) for m in (-1, 1, 0, 2, -1, 0) * 3])
    pi0, A0 = np.full(3, 1 / 3), np.full((3, 3), 0.1) + np.eye(3) * 0.7
    mu0, sd0 = np.array([-0.5, 0.5, 1.5]), np.array([0.8, 0.8, 0.8])
    ours = _em([x], 3, rng, max_iter=200, tol=0, init=(pi0, A0, mu0, sd0))
    ref = hmm.GaussianHMM(3, covariance_type="diag", n_iter=200, tol=0, init_params="", params="stmc")
    ref.startprob_, ref.transmat_, ref.means_, ref.covars_ = pi0, A0, mu0[:, None], (sd0**2)[:, None]
    ref.fit(x[:, None])
    assert ours[4] == pytest.approx(ref.score(x[:, None]), rel=1e-6)
    assert np.allclose(ours[2], ref.means_.ravel(), atol=1e-5)
    assert np.allclose(ours[1], ref.transmat_, atol=1e-5)


def test_recovers_simulated_states_with_missing_days():
    d, A = _simulate()
    m = fit_mils(d, 3, n_starts=5)
    merged = m.data.merge(d, on=["id", "date"])
    known = merged["load_x"].notna()
    acc = (merged.loc[known, "state"].str[1].astype(int) - 1 == merged.loc[known, "true_state"]).mean()
    assert acc > 0.85
    assert np.allclose(np.diag(m.trans), np.diag(A), atol=0.08)
    assert np.all(np.diff(m.means) > 0)                      # states sorted by load
    assert np.allclose(m.data.filter(like="p_S").sum(axis=1), 1)  # posterior probabilities


def test_missing_observation_has_no_likelihood():
    b = _emission(np.array([np.nan, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0]))
    assert np.allclose(b[0], 1.0)


def test_redi_matches_the_r_formula():
    rng = np.random.default_rng(5)
    x = rng.uniform(100, 600, 60); x[rng.random(60) < 0.3] = np.nan
    lam = 2 / (28 + 1)
    out = redi(x, lam)
    for i in range(60):  # MILS::redi, written as in R
        w = np.exp(-lam * np.arange(1, i + 2))[::-1]
        v = x[: i + 1].copy(); w[np.isnan(v)] = 0; v[np.isnan(v)] = 0
        assert out[i] == pytest.approx((w * v).sum() / w.sum()) if w.sum() > 0 else np.isnan(out[i])


def test_ewma():
    assert np.allclose(ewma(np.array([10.0, 10, 10]), 3), 10)
    assert ewma(np.array([0.0, 10]), 3)[1] == pytest.approx(5.0)


def test_compiled_and_python_forward_backward_agree():
    from charge.mils import _forward_backward_py
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 300); x[rng.random(300) < 0.2] = np.nan
    b = _emission(x, np.array([-1.0, 0, 1]), np.array([0.5, 0.5, 0.5]))
    pi, A = np.full(3, 1 / 3), np.full((3, 3), 0.1) + np.eye(3) * 0.7
    for got, ref in zip(_forward_backward(b, pi, A), _forward_backward_py(b, pi, A)):
        assert np.allclose(got, ref)
