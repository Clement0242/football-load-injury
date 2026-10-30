"""MILS (Markov Index Load State) in Python.

A reimplementation of the model of the R package MILS by Quentin De
Larochelambert and Kilian Barlier (INSEP, GPL-3,
https://github.com/quentincad2/MILS), which fits a hidden Markov model to
daily workloads with depmixS4. The same steps are kept:

1. daily load per athlete, every calendar day present (missing days added);
2. a load of 0 is a rest day, treated as *missing*, not as a low value;
3. days before the first and after the last recorded load are dropped;
4. load z-scored within each athlete, so that states are relative to the
   athlete's own habits;
5. one Gaussian HMM shared by all athletes (each athlete is a sequence),
   fitted by EM; missing days contribute no likelihood, the chain runs on;
6. states sorted by increasing mean, posterior probability of each state
   every day (forward-backward), state of the day = the most probable one.

Pure NumPy. On complete data the EM matches ``hmmlearn`` (see the tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ preprocessing

def prepare(data: pd.DataFrame, id_col: str = "id", date_col: str = "date", load_col: str = "load",
            min_obs: int = 2) -> pd.DataFrame:
    """Steps 1-4: one row per athlete-day with ``load`` and ``load_scaled``."""
    d = data[[id_col, date_col, load_col]].copy()
    d.columns = ["id", "date", "load"]
    d["id"] = d["id"].astype(str)
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["load"] = pd.to_numeric(d["load"], errors="coerce")
    d = d.groupby(["id", "date"], as_index=False)["load"].sum(min_count=1)

    out = []
    for ath, g in d.groupby("id"):
        g = g.set_index("date").reindex(pd.date_range(g["date"].min(), g["date"].max(), freq="D"))
        g["load"] = g["load"].where(g["load"] != 0)  # rest day = missing
        valid = g["load"].notna()
        if not valid.any():
            continue
        g = g.loc[valid.idxmax(): valid[::-1].idxmax()]  # trim leading / trailing missing days
        if len(g) <= min_obs:  # as in MILS: days counted after trimming
            continue
        sd = g["load"].std(ddof=1)
        g["load_scaled"] = (g["load"] - g["load"].mean()) / sd if sd > 0 else 0.0
        g["id"] = ath
        out.append(g.rename_axis("date").reset_index())
    if not out:
        raise ValueError("no athlete has enough observations")
    return pd.concat(out, ignore_index=True)[["id", "date", "load", "load_scaled"]]


# ---------------------------------------------------------------------- the HMM

def _emission(x: np.ndarray, means: np.ndarray, sds: np.ndarray) -> np.ndarray:
    """Gaussian densities (n, K); a missing observation has density 1 in every state."""
    z = (x[:, None] - means[None, :]) / sds[None, :]
    b = np.exp(-0.5 * z**2) / (np.sqrt(2 * np.pi) * sds[None, :])
    b[np.isnan(x)] = 1.0
    return np.maximum(b, 1e-300)


def _forward_backward_py(b: np.ndarray, pi: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Scaled forward-backward. Returns gamma (n, K), xi summed over time (K, K), log-likelihood."""
    n, K = b.shape
    alpha, c = np.empty((n, K)), np.empty(n)
    a = pi * b[0]
    c[0] = a.sum(); alpha[0] = a / c[0]
    for t in range(1, n):
        a = (alpha[t - 1] @ A) * b[t]
        c[t] = a.sum(); alpha[t] = a / c[t]
    beta = np.ones((n, K))
    xi = np.zeros((K, K))
    for t in range(n - 2, -1, -1):
        nb = b[t + 1] * beta[t + 1]
        beta[t] = (A @ nb) / c[t + 1]
        xi += A * np.outer(alpha[t], nb) / c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma, xi, float(np.log(c).sum())


try:  # The recursion is a Python loop over days: compiled when numba is there (~100x faster).
    from numba import njit

    @njit(cache=True)
    def _fb_compiled(b, pi, A):
        n, K = b.shape
        alpha = np.empty((n, K)); c = np.empty(n)
        a = pi * b[0]; c[0] = a.sum(); alpha[0] = a / c[0]
        for t in range(1, n):
            a = (alpha[t - 1] @ A) * b[t]
            c[t] = a.sum(); alpha[t] = a / c[t]
        beta = np.ones((n, K)); xi = np.zeros((K, K))
        for t in range(n - 2, -1, -1):
            nb = b[t + 1] * beta[t + 1]
            beta[t] = (A @ nb) / c[t + 1]
            xi += A * np.outer(alpha[t], nb) / c[t + 1]
        gamma = alpha * beta
        for t in range(n):
            gamma[t] /= gamma[t].sum()
        return gamma, xi, np.log(c).sum()

    def _forward_backward(b, pi, A):
        g, xi, ll = _fb_compiled(np.ascontiguousarray(b), np.ascontiguousarray(pi), np.ascontiguousarray(A))
        return g, xi, float(ll)

    @njit(cache=True)
    def _estep_compiled(x, bounds, pi, A, means, sds):
        """Whole E-step over all athletes (sequences x[bounds[i]:bounds[i+1]]) in one compiled call."""
        K = means.size
        ll = 0.0
        pi_acc = np.zeros(K); xi_acc = np.zeros((K, K))
        w_sum = np.zeros(K); w_x = np.zeros(K); w_x2 = np.zeros(K)
        for s in range(bounds.size - 1):
            seq = x[bounds[s]:bounds[s + 1]]
            n = seq.size
            b = np.empty((n, K))
            for t in range(n):
                for k in range(K):
                    if np.isnan(seq[t]):
                        b[t, k] = 1.0
                    else:
                        z = (seq[t] - means[k]) / sds[k]
                        b[t, k] = max(np.exp(-0.5 * z * z) / (np.sqrt(2 * np.pi) * sds[k]), 1e-300)
            g, xi, l = _fb_compiled(b, pi, A)
            ll += l
            pi_acc += g[0]; xi_acc += xi
            for t in range(n):
                if not np.isnan(seq[t]):
                    for k in range(K):
                        w_sum[k] += g[t, k]; w_x[k] += g[t, k] * seq[t]; w_x2[k] += g[t, k] * seq[t] ** 2
        return ll, pi_acc, xi_acc, w_sum, w_x, w_x2

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _forward_backward = _forward_backward_py
    _HAS_NUMBA = False


@dataclass
class MilsModel:
    """A fitted MILS model. States are sorted by increasing mean (S1 = lowest load)."""

    n_states: int
    init: np.ndarray
    trans: np.ndarray
    means: np.ndarray
    sds: np.ndarray
    loglik: float
    n_iter: int
    data: pd.DataFrame = field(repr=False, default=None)

    @property
    def n_params(self) -> int:
        k = self.n_states
        return (k - 1) + k * (k - 1) + 2 * k

    @property
    def bic(self) -> float:
        n = int(self.data["load_scaled"].notna().sum()) if self.data is not None else 0
        return -2 * self.loglik + self.n_params * np.log(max(n, 1))

    def posterior(self, scaled: np.ndarray) -> np.ndarray:
        gamma, _, _ = _forward_backward(_emission(scaled, self.means, self.sds), self.init, self.trans)
        return gamma

    def summary(self) -> str:
        lines = [f"MILS, {self.n_states} states, log-likelihood {self.loglik:.1f}, BIC {self.bic:.1f}, "
                 f"{self.n_iter} EM iterations",
                 "  state   mean (z)   sd    stays (P[i→i])"]
        for k in range(self.n_states):
            lines.append(f"  S{k+1}     {self.means[k]:+7.2f}  {self.sds[k]:5.2f}   {self.trans[k, k]:.2f}")
        return "\n".join(lines)


def _em(seqs: list[np.ndarray], K: int, rng: np.random.Generator, max_iter: int, tol: float,
        init: tuple | None = None) -> tuple:
    allx = np.concatenate(seqs)
    obs = allx[~np.isnan(allx)]
    if init is None:
        q = np.sort(rng.choice(obs, K, replace=False)) if obs.size >= K else np.linspace(-1, 1, K)
        means = q + rng.normal(0, 0.05, K)
        sds = np.full(K, obs.std() / K + 1e-3)
        A = np.full((K, K), 0.2 / max(K - 1, 1)); np.fill_diagonal(A, 0.8)
        pi = np.full(K, 1.0 / K)
    else:
        pi, A, means, sds = (np.array(p, dtype=float) for p in init)

    prev = -np.inf
    flat = np.ascontiguousarray(np.concatenate(seqs))
    bounds = np.concatenate(([0], np.cumsum([len(x) for x in seqs]))).astype(np.int64)
    for it in range(1, max_iter + 1):
        if _HAS_NUMBA:
            ll, pi_acc, xi_acc, w_sum, w_x, w_x2 = _estep_compiled(flat, bounds, pi, A, means, sds)
        else:  # pragma: no cover
            ll, pi_acc, xi_acc = 0.0, np.zeros(K), np.zeros((K, K))
            w_sum, w_x, w_x2 = np.zeros(K), np.zeros(K), np.zeros(K)
            for x in seqs:
                g, xi, l = _forward_backward(_emission(x, means, sds), pi, A)
                ll += l
                pi_acc += g[0]; xi_acc += xi
                m = ~np.isnan(x)
                w_sum += g[m].sum(0); w_x += (g[m] * x[m, None]).sum(0); w_x2 += (g[m] * x[m, None] ** 2).sum(0)
        pi = pi_acc / pi_acc.sum()
        A = xi_acc / xi_acc.sum(axis=1, keepdims=True)
        means = w_x / np.maximum(w_sum, 1e-12)
        sds = np.sqrt(np.maximum(w_x2 / np.maximum(w_sum, 1e-12) - means**2, 1e-6))
        if abs(ll - prev) < tol * max(1.0, abs(ll)):
            break
        prev = ll
    return pi, A, means, sds, ll, it


def fit_mils(data: pd.DataFrame, n_states: int = 3, *, id_col: str = "id", date_col: str = "date",
             load_col: str = "load", min_obs: int = 2, n_starts: int = 10, max_iter: int = 2000,
             tol: float = 1e-8, seed: int = 1) -> MilsModel:
    """Prepare the data and fit MILS; the best of ``n_starts`` random starts is kept."""
    df = prepare(data, id_col, date_col, load_col, min_obs)
    seqs = [g["load_scaled"].to_numpy(dtype=float) for _, g in df.groupby("id", sort=False)]
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_starts):
        res = _em(seqs, n_states, rng, max_iter, tol)
        if best is None or res[4] > best[4]:
            best = res
    pi, A, means, sds, ll, it = best

    order = np.argsort(means)  # S1 = lowest load, as in MILS
    pi, A, means, sds = pi[order], A[np.ix_(order, order)], means[order], sds[order]

    post = []
    for x in seqs:
        post.append(_forward_backward(_emission(x, means, sds), pi, A)[0])
    post = np.vstack(post)
    names = [f"S{k+1}" for k in range(n_states)]
    df[[f"p_{s}" for s in names]] = post
    df["state"] = np.array(names)[post.argmax(axis=1)]
    return MilsModel(n_states, pi, A, means, sds, ll, it, df)


# ------------------------------------------------------------- REDI and EWMA

def redi(load: np.ndarray, lam: float) -> np.ndarray:
    """Robust Exponential Decreasing Index (Moussa et al. 2019), as in MILS::redi.

    REDI_t = sum_j exp(-lam (t - j)) L_j / sum_j exp(-lam (t - j)), summed over the
    days j <= t where the load is known: missing days do not count as zero.
    """
    x = np.asarray(load, dtype=float)
    known = ~np.isnan(x)
    num = den = 0.0
    out = np.empty(x.size)
    decay = np.exp(-lam)
    for t in range(x.size):
        num, den = num * decay, den * decay
        if known[t]:
            num += x[t]; den += 1.0
        out[t] = num / den if den > 0 else np.nan
    return out


def ewma(load: np.ndarray, n: int) -> np.ndarray:
    """Exponentially weighted moving average with alpha = 2 / (n + 1); missing days count as 0."""
    x = np.nan_to_num(np.asarray(load, dtype=float))
    a = 2.0 / (n + 1)
    out = np.empty(x.size)
    out[0] = x[0]
    for t in range(1, x.size):
        out[t] = a * x[t] + (1 - a) * out[t - 1]
    return out
