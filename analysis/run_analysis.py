"""Load states, heart rate × GPS coupling and injuries in elite women's football.

Reads the governed marts published by the athlete-data-governance repository
and writes every figure and number of the README:

    python analysis/run_analysis.py --marts ../athlete-data-governance/data/marts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from charge.mils import fit_mils, redi
from charge.stats import (bootstrap_within_player_auc, cohen_kappa, leave_one_player_out, rmcorr,
                          within_player_auc)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e8e8e8", "figure.dpi": 150})
BLUE, ORANGE, RED, GREY = "#2f6fdc", "#e8743b", "#c0392b", "#bdbdbd"
STATE_COLORS = ["#9bbcf0", "#2f6fdc", "#c0392b"]

LOADS = {"srpe_au": "sRPE (questionnaire)", "edwards_trimp_au": "Edwards TRIMP (heart rate)",
         "distance_m": "Distance (GNSS)", "distance_hsr_m": "High-speed running",
         "n_accelerations": "Accelerations", "accelerometer_load_au": "Accelerometer load"}


def load_marts(path: str) -> dict[str, pd.DataFrame]:
    base = path.rstrip("/")
    names = ["fct_player_day", "fct_gps_session", "fct_injury_episode", "dim_player", "fct_training_session"]
    out = {n: pd.read_parquet(f"{base}/{n}.parquet") for n in names}
    for n in ("fct_player_day",):
        out[n]["calendar_date"] = pd.to_datetime(out[n]["calendar_date"])
    out["fct_injury_episode"]["onset_date"] = pd.to_datetime(out["fct_injury_episode"]["onset_date"])
    return out


# ------------------------------------------------------------------ A. coupling

def coupling(day: pd.DataFrame, gps: pd.DataFrame, res: dict) -> None:
    both = day[day.has_rpe & day.has_gps].copy()  # has_gps: validated recordings only
    cols = [c for c in LOADS if both[c].notna().sum() > 50]
    mat = pd.DataFrame(np.nan, index=cols, columns=cols)
    pairs = {}
    for i, a in enumerate(cols):
        for b in cols[i:]:
            r = rmcorr(both, a, b)["r"] if a != b else 1.0
            mat.loc[a, b] = mat.loc[b, a] = r
            if a != b:
                pairs[f"{a}~{b}"] = rmcorr(both, a, b)
    res["coupling"] = {"n_player_days_both": int(len(both)), "n_players": int(both.player_id.nunique()), "pairs": pairs}

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(mat.to_numpy(float), vmin=-1, vmax=1, cmap="RdBu_r")
    labels = [LOADS[c] for c in cols]
    ax.set_xticks(range(len(cols)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(cols)), labels)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(mat.iloc[i, j]) > 0.6 else "black")
    ax.grid(False)
    ax.set_title(f"Within-player correlation of daily loads\n(repeated-measures r, {len(both):,} player-days)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "coupling_matrix.png")
    plt.close(fig)

    # Cardiac cost of running against the morning's wellness
    hr = day.dropna(subset=["hr_at_2_ms_bpm"])
    res["hr_cost"] = {"n_sessions": int(len(hr)), "n_players": int(hr.player_id.nunique())}
    for w in ("readiness", "fatigue", "soreness", "sleep_quality"):
        if hr[w].notna().sum() > 50:
            res["hr_cost"][w] = rmcorr(hr, "hr_at_2_ms_bpm", w)
    if len(hr) > 50:
        top = hr.player_id.value_counts().index[:4]
        fig, axes = plt.subplots(1, len(top), figsize=(13, 3.2), sharey=True)
        for k, (ax, p) in enumerate(zip(np.atleast_1d(axes), top)):
            s = hr[hr.player_id == p].sort_values("calendar_date")
            ax.plot(s.calendar_date, s.hr_at_2_ms_bpm, ".", color=GREY, ms=4)
            ax.plot(s.calendar_date, s.hr_at_2_ms_bpm.rolling(7, min_periods=3).median(), color=BLUE, lw=2)
            ax.set_title(f"Player {chr(65 + k)}", fontsize=10)
            ax.tick_params(axis="x", rotation=30, labelsize=8)
        np.atleast_1d(axes)[0].set_ylabel("HR at 2 m/s (bpm)")
        fig.suptitle("Cardiac cost of the same running speed across the season (7-session median)", y=1.03)
        fig.tight_layout()
        fig.savefig(FIG / "hr_cost_timeline.png", bbox_inches="tight")
        plt.close(fig)


# -------------------------------------------------------------------- B. MILS

def mils_states(day: pd.DataFrame, res: dict, quick: bool = False) -> dict[str, pd.DataFrame]:
    out, res["mils"] = {}, {}
    for load in ("srpe_au", "edwards_trimp_au", "accelerometer_load_au"):
        d = day[["player_id", "calendar_date", load]].dropna()
        if d.player_id.nunique() < 5 or len(d) < 500:
            continue
        bic = {}
        for k in ((3,) if quick else (2, 3, 4)):
            m = fit_mils(d, k, id_col="player_id", date_col="calendar_date", load_col=load,
                         n_starts=2 if quick else 6)
            bic[k] = m.bic
            if k == 3:
                model = m
        st = model.data.rename(columns={"id": "player_id", "date": "calendar_date"})
        typical = st.dropna(subset=["load"]).groupby("state")["load"].median()
        res["mils"][load] = {
            "bic": bic, "means_z": model.means.round(3).tolist(), "sds_z": model.sds.round(3).tolist(),
            "trans": model.trans.round(3).tolist(), "median_load_by_state": typical.round(1).to_dict(),
            "share_of_days": st.dropna(subset=["load"])["state"].value_counts(normalize=True).round(3).to_dict(),
            "n_players": int(st.player_id.nunique()), "n_days": int(len(st)),
        }
        out[load] = st
    if "srpe_au" in out and "edwards_trimp_au" in out:
        m = out["srpe_au"].merge(out["edwards_trimp_au"], on=["player_id", "calendar_date"], suffixes=("_s", "_t"))
        m = m.dropna(subset=["load_s", "load_t"])
        res["mils"]["agreement_srpe_trimp"] = {"kappa": cohen_kappa(m.state_s, m.state_t), "n_days": int(len(m)),
                                                "crosstab": pd.crosstab(m.state_s, m.state_t).to_dict()}
    return out


def mils_figure(states: pd.DataFrame, injuries: pd.DataFrame, res: dict) -> None:
    # The player with the most injury onsets and a long record, shown as one timeline.
    cand = injuries.player_id.value_counts()
    cand = cand[(cand >= 2) & (cand <= 4)]  # a typical player, not the one who reports everything
    p = next((x for x in cand.index if (states.player_id == x).sum() > 400), states.player_id.iloc[0])
    s = states[states.player_id == p].sort_values("calendar_date")
    inj = injuries[injuries.player_id == p]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 4.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for k, c in enumerate(STATE_COLORS, 1):
        m = s.state == f"S{k}"
        a1.bar(s.calendar_date[m], s.load[m], width=1, color=c, label=f"S{k}")
    for d in inj.onset_date:
        a1.axvline(d, color="k", lw=1, ls="--")
    a1.set_ylabel("sRPE (AU)")
    a1.legend(title="MILS state", ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0))
    a1.set_title("Daily sRPE coloured by MILS state · dashed lines = injury onsets", loc="right", fontsize=10)
    a2.stackplot(s.calendar_date, s.p_S1, s.p_S2, s.p_S3, colors=STATE_COLORS)
    a2.set_ylabel("P(state)")
    a2.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG / "mils_timeline.png")
    plt.close(fig)


# ---------------------------------------------------------------- C. injuries

def injury_model(day: pd.DataFrame, states: dict[str, pd.DataFrame], res: dict, horizon: int = 7) -> None:
    d = day.sort_values(["player_id", "calendar_date"]).copy()
    s = states["srpe_au"][["player_id", "calendar_date", "p_S3"]]
    d = d.merge(s, on=["player_id", "calendar_date"], how="left")
    g = d.groupby("player_id", group_keys=False)

    d["redi_acute"] = g["srpe_au"].transform(lambda x: pd.Series(redi(x.to_numpy(), 2 / (7 + 1)), index=x.index))
    d["redi_chronic"] = g["srpe_au"].transform(lambda x: pd.Series(redi(x.to_numpy(), 2 / (28 + 1)), index=x.index))
    d["redi_ratio"] = d.redi_acute / d.redi_chronic
    d["high_state_7d"] = g["p_S3"].transform(lambda x: x.rolling(7, min_periods=4).mean())
    for w in ("readiness", "fatigue", "soreness"):
        z = (d[w] - g[w].transform("mean")) / g[w].transform("std")
        d[f"{w}_7d"] = z.groupby(d.player_id).transform(lambda x: x.rolling(7, min_periods=3).mean())
    # Outcome: an injury starts within the next `horizon` days.
    fut = g["is_injury_onset"].transform(
        lambda x: x[::-1].rolling(horizon, min_periods=1).max()[::-1].shift(-1).fillna(0))
    d["onset_next"] = fut.astype(int)

    # Wellness items are scored so that a *worse* state is a higher number.
    for w in ("readiness_7d", "fatigue_7d", "soreness_7d"):
        d[w] = -d[w]
    at_risk = d[~d.is_injured & d.redi_chronic.notna() & d.high_state_7d.notna()].copy()
    sets = {"REDI acute:chronic (sRPE)": ["redi_ratio"], "MILS high-load state, 7 d": ["high_state_7d"],
            "Poor readiness, 7 d": ["readiness_7d"], "Fatigue, 7 d": ["fatigue_7d"], "Soreness, 7 d": ["soreness_7d"],
            "Wellness (3 items, model)": ["readiness_7d", "fatigue_7d", "soreness_7d"]}
    sets["All (model)"] = ["redi_ratio", "high_state_7d", "readiness_7d", "fatigue_7d", "soreness_7d"]
    res["injury"] = {"horizon_days": horizon, "n_at_risk_days": int(len(at_risk)),
                     "n_positive_days": int(at_risk.onset_next.sum()),
                     "n_onsets": int(d.is_injury_onset.sum()), "models": {}}
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for i, (name, feats) in enumerate(sets.items()):
        sub = at_risk.dropna(subset=feats)
        if len(feats) == 1:  # one indicator: its own value is the score, nothing is fitted
            score = sub[feats[0]].to_numpy(float)
        else:                # several: a logistic model that never saw the player
            score = leave_one_player_out(sub, feats, "onset_next")
        ok = np.isfinite(score)
        y, g = sub.onset_next.to_numpy()[ok], sub.player_id.to_numpy()[ok]
        a = within_player_auc(y, score[ok], g)
        lo, hi = bootstrap_within_player_auc(y, score[ok], g, 500)
        res["injury"]["models"][name] = {"auc_within_player": a, "ci": [lo, hi], "n_days": int(ok.sum()),
                                         "n_positive": int(y.sum()), "n_players": int(len(set(g[y == 1])))}
        ax.errorbar(a, i, xerr=[[a - lo], [hi - a]], fmt="o", color=BLUE, capsize=3)
    ax.axvline(0.5, color=GREY, ls="--")
    ax.set_yticks(range(len(sets)), list(sets))
    ax.set_xlim(0.3, 0.8)
    ax.set_xlabel("Within-player AUC (95 % CI, players resampled)")
    ax.set_title(f"Does anything warn of an injury in the next {horizon} days?")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "injury_auc.png")
    plt.close(fig)

    # Event study: high-load state probability around injury onsets.
    on = d[d.is_injury_onset]
    rows = []
    for _, r in on.iterrows():
        w = d[(d.player_id == r.player_id) & (d.calendar_date.between(r.calendar_date - pd.Timedelta(days=28),
                                                                      r.calendar_date))]
        rows.append(pd.DataFrame({"lag": (w.calendar_date - r.calendar_date).dt.days, "p": w.p_S3}))
    if rows:
        ev = pd.concat(rows).groupby("lag")["p"].agg(["mean", "count"])
        base = float(d.p_S3.mean())
        res["injury"]["event_study_baseline"] = base
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        ax.plot(ev.index, ev["mean"], color=RED, lw=2, label="before an injury onset")
        ax.axhline(base, color=GREY, ls="--", label="all days")
        ax.set(xlabel="days relative to injury onset", ylabel="P(high-load state S3)",
               title="High-load state in the 4 weeks before injuries")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIG / "injury_event_study.png")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one state count, two starts (debugging)")
    ap.add_argument("--marts", default=str(ROOT.parent / "athlete-data-governance" / "data" / "marts"),
                    help="folder or URL holding the governed marts (parquet)")
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    m = load_marts(a.marts)
    day, gps, inj = m["fct_player_day"], m["fct_gps_session"], m["fct_injury_episode"]
    res = {"n_players": int(day.player_id.nunique()), "n_player_days": int(len(day)),
           "n_gps_sessions": int(len(gps)), "n_injury_episodes": int(len(inj)),
           "n_unlinked_gps_players": int((m["dim_player"].has_objective_data & ~m["dim_player"].has_subjective_data).sum())}
    coupling(day, gps, res)
    states = mils_states(day, res, a.quick)
    mils_figure(states["srpe_au"], inj, res)
    injury_model(day, states, res)
    (ROOT / "docs" / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float)[:6000])


if __name__ == "__main__":
    main()
