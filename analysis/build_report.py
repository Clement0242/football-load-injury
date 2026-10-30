"""Write the project page (docs/index.html) from docs/results.json and the figures.

Every number on the page comes from results.json, written by run_analysis.py:
re-running the analysis and this script keeps text and results in step.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
r = json.loads((DOCS / "results.json").read_text())


def f(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}".replace("-", "−")


def rm(key: str) -> dict:
    return r["coupling"]["pairs"][key]


def ci(d: dict, a: str = "ci_low", b: str = "ci_high") -> str:
    return f"[{f(d[a])}; {f(d[b])}]"


def main() -> None:
    shutil.copy(ROOT / "docs" / "assets" / "projet.css", DOCS / "projet.css")
    c = r["coupling"]
    srpe_trimp, trimp_dist = rm("srpe_au~edwards_trimp_au"), rm("edwards_trimp_au~distance_m")
    srpe_dist, trimp_acc = rm("srpe_au~distance_m"), rm("edwards_trimp_au~accelerometer_load_au")
    hc = r["hr_cost"]
    ms, mt = r["mils"]["srpe_au"], r["mils"].get("edwards_trimp_au")
    kappa = r["mils"].get("agreement_srpe_trimp", {}).get("kappa")
    inj = r["injury"]
    rows = "".join(
        f'<tr><td>{name}</td><td class="num">{f(m["auc_within_player"])}</td>'
        f'<td class="num">{f(m["ci"][0])} – {f(m["ci"][1])}</td><td class="num">{m["n_positive"]:,}</td>'
        f'<td class="num">{m["n_players"]}</td></tr>'
        for name, m in inj["models"].items())
    best = max(inj["models"].items(), key=lambda kv: kv[1]["auc_within_player"])
    trans = ms["trans"]
    stay = " · ".join(f"S{i + 1} {trans[i][i]:.0%}" for i in range(3))
    med = ms["median_load_by_state"]
    page = TEMPLATE.format(
        n_players=r["n_players"], n_days=f'{r["n_player_days"]:,}', n_gps=f'{r["n_gps_sessions"]:,}',
        n_inj=r["n_injury_episodes"], n_unlinked=r["n_unlinked_gps_players"], n_both=f'{c["n_player_days_both"]:,}', n_both_players=c["n_players"],
        r_st=f(srpe_trimp["r"]), ci_st=ci(srpe_trimp), n_st=f'{srpe_trimp["n_obs"]:,}',
        r_td=f(trimp_dist["r"]), r_sd=f(srpe_dist["r"]), r_ta=f(trimp_acc["r"]),
        n_hr=hc["n_sessions"], n_hr_p=hc["n_players"],
        r_hr_ready=f(hc["readiness"]["r"]) if "readiness" in hc else "n/a",
        ci_hr_ready=ci(hc["readiness"]) if "readiness" in hc else "",
        r_hr_fat=f(hc["fatigue"]["r"]) if "fatigue" in hc else "n/a",
        s1=med.get("S1", 0), s2=med.get("S2", 0), s3=med.get("S3", 0), stay=stay,
        share3=f'{ms["share_of_days"].get("S3", 0):.0%}', bic=", ".join(f"{k} states {v:,.0f}" for k, v in ms["bic"].items()),
        kappa=f(kappa) if kappa is not None else "n/a",
        n_risk=f'{inj["n_at_risk_days"]:,}', n_onsets=inj["n_onsets"], horizon=inj["horizon_days"],
        auc_rows=rows, best_name=best[0], best_auc=f(best[1]["auc_within_player"]),
        best_ci=f'{f(best[1]["ci"][0])} – {f(best[1]["ci"][1])}',
        n_inj_players=max(m["n_players"] for m in inj["models"].values()),
    )
    (DOCS / "index.html").write_text(page)
    print("docs/index.html written")


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Load states and injuries</title>
<meta name="description" content="Hidden Markov load states (MILS), heart rate × GPS coupling and injuries in elite women's football, on the SoccerMon open dataset.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Instrument+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="projet.css">
</head>
<body>
<header class="barre"><div class="enveloppe">
  <a class="marque" href="https://clement0242.github.io"><b>CV</b> Clément Verdier</a>
  <nav aria-label="Sections">
    <a href="#coupling">Heart rate × GPS</a><a href="#mils">Load states</a><a href="#injuries">Injuries</a>
    <a href="#method">Method</a>
  </nav>
</div></header>

<main>
<div class="tete"><div class="enveloppe">
  <p class="surtitre">Sport science · hidden Markov models · open data</p>
  <h1>What load says, <em>and what it does not</em></h1>
  <p class="chapeau">Two seasons of an elite women's football squad: questionnaires, GNSS and heart rate,
  aligned day by day. How internal and external load agree, what hidden load states look like with MILS,
  and whether any of it warns of an injury.</p>
  <div class="boutons">
    <a class="bouton" href="https://github.com/Clement0242/football-load-injury">Code on GitHub</a>
    <a class="bouton creux" href="https://clement0242.github.io/athlete-data-governance/">The governed data behind it</a>
  </div>
  <div class="chiffres">
    <div><b>{n_players}</b><span>players, two teams, 2020–2021</span></div>
    <div><b>{n_days}</b><span>player-days aligned across sources</span></div>
    <div><b>{n_gps}</b><span>GNSS + heart-rate recordings, 99 GB streamed</span></div>
    <div><b>{n_inj}</b><span>injury episodes rebuilt from daily reports</span></div>
  </div>
</div></div>

<section id="coupling"><div class="enveloppe">
  <p class="etiq">1 · Heart rate × GPS</p>
  <h2>Internal and external load agree, within a player</h2>
  <div class="prose">
    <p>Correlations are computed <strong>within each player</strong> (repeated-measures correlation): a player who
    always covers more distance at a lower heart rate cannot create a correlation by herself.
    On {n_both} player-days ({n_both_players} players) with both a questionnaire and a recording:</p>
    <ul>
      <li>Heart-rate load (Edwards TRIMP) follows distance (r = {r_td}) and accelerometer load (r = {r_ta}) closely.</li>
      <li>The questionnaire (sRPE) agrees with heart rate (r = {r_st}, 95 % CI {ci_st}, {n_st} days) and with
      distance (r = {r_sd}), but less: perceived effort carries something the sensors do not.</li>
    </ul>
  </div>
  <figure><div class="feuille"><img src="figures/coupling_matrix.png" alt="Matrix of within-player correlations between six daily load measures." width="960" height="780" loading="lazy"></div>
  <figcaption>Within-player correlation between daily loads. Questionnaire, heart rate, GNSS and accelerometer.</figcaption></figure>
  <h3>The cardiac cost of the same running speed</h3>
  <div class="prose"><p>From 30-second epochs, heart rate is regressed on speed in each session and read at 2 m/s.
  Only {n_hr} sessions ({n_hr_p} players) qualify: a valid strap <em>and</em> enough running epochs for a stable
  regression. Its within-player link with the morning's wellness is weak and not in the expected direction
  (readiness r = {r_hr_ready} {ci_hr_ready}; fatigue r = {r_hr_fat}). As measured here, it is not yet a usable
  indicator: it needs a strap worn every session, or a standard running drill to read it on.</p></div>
  <figure><div class="feuille"><img src="figures/hr_cost_timeline.png" alt="Heart rate at 2 m/s across the season for four players." width="1950" height="520" loading="lazy"></div>
  <figcaption>Heart rate at 2 m/s for four players (7-session rolling median).</figcaption></figure>
</div></section>

<section id="mils"><div class="enveloppe">
  <p class="etiq">2 · Load states (MILS)</p>
  <h2>Three hidden states in the daily load</h2>
  <div class="prose">
    <p>MILS (Markov Index Load State, Q. De Larochelambert &amp; K. Barlier, INSEP) fits a hidden Markov model to
    each player's daily load, z-scored against her own habits. A rest day is <em>missing</em>, not a low value.
    The package is in R; it is reimplemented here in Python and checked against <code>hmmlearn</code>.</p>
    <ul>
      <li>Typical sRPE in each state: S1 {s1:.0f} AU, S2 {s2:.0f} AU, S3 {s3:.0f} AU; S3 holds {share3} of the days.</li>
      <li>Probability of staying in the same state the next day: {stay}. S2 is a lasting regime of
      below-habit load; S1 days are isolated light days within training weeks.</li>
      <li>The BIC ({bic}) would prefer more states for sRPE; three are kept, the MILS default, because they
      stay interpretable for a staff.</li>
      <li>States from the questionnaire and from heart rate agree little (Cohen's κ = {kappa}): "high load" depends on
      which load you measure.</li>
    </ul>
  </div>
  <figure><div class="feuille"><img src="figures/mils_timeline.png" alt="One player's daily sRPE over two seasons, coloured by state, with the probability of each state below." width="1800" height="690" loading="lazy"></div>
  <figcaption>One player, two seasons. Bars: daily sRPE coloured by the most probable state. Below: probability of each state.</figcaption></figure>
</div></section>

<section id="injuries"><div class="enveloppe">
  <p class="etiq">3 · Injuries</p>
  <h2>Does anything warn of an injury in the next {horizon} days?</h2>
  <div class="prose"><p>For each day a player was not injured ({n_risk} days, {n_onsets} injury onsets), the outcome is an
  injury starting in the next {horizon} days. Each indicator is judged by its <strong>within-player AUC</strong>:
  are a player's days before an injury ranked above her own other days? 0.5 is chance. Combined indicators use a
  logistic model that never saw the player it predicts.</p></div>
  <div class="tableau"><table>
    <thead><tr><th>Indicator</th><th class="num">AUC</th><th class="num">95 % CI</th><th class="num">Days before injury</th><th class="num">Players</th></tr></thead>
    <tbody>{auc_rows}</tbody>
  </table></div>
  <div class="constat"><p>No indicator does better than chance with any confidence: every interval includes 0.5.
  The best, {best_name}, reaches {best_auc} ({best_ci}).</p>
  <p>The {n_onsets} injury onsets come from only {n_inj_players} players, and injuries are self-reported, so the test has
  little power. The finding still matches the literature (Impellizzeri et al. 2020): these indicators help plan
  training; they do not predict a given player's injury.</p></div>
  <div class="duo">
    <figure><div class="feuille"><img src="figures/injury_auc.png" alt="AUC of each indicator with its confidence interval." width="990" height="510" loading="lazy"></div>
    <figcaption>Within-player AUC, players resampled for the interval.</figcaption></figure>
    <figure><div class="feuille"><img src="figures/injury_event_study.png" alt="Probability of the high-load state in the 28 days before injury onsets, against all days." width="960" height="480" loading="lazy"></div>
    <figcaption>Probability of the high-load state in the 4 weeks before injuries, against all days.</figcaption></figure>
  </div>
</div></section>

<section id="method"><div class="enveloppe">
  <p class="etiq">Method</p>
  <h2>Choices that matter</h2>
  <div class="cartes">
    <div class="carte"><h3>Governed data</h3><p>Every input comes from the marts of a dbt warehouse whose rules found frozen
      heart-rate sensors, missing sessions and twice-pseudonymised players before any analysis ran.</p></div>
    <div class="carte"><h3>Within-player statistics</h3><p>Correlations and AUCs compare each player with herself.
      A pooled leave-one-player-out AUC is biased here (it fell far below 0.5): the reason is in the code.</p></div>
    <div class="carte"><h3>Missing is not zero</h3><p>No report, no session and no strap are three different things,
      and none of them becomes a 0 in a model.</p></div>
    <div class="carte"><h3>Limits</h3><p>Two teams, self-reported injuries, heart rate on a minority of sessions,
      and {n_unlinked} players whose GNSS data cannot be linked to their questionnaires.</p></div>
  </div>
<pre><code>pip install -e .
python analysis/run_analysis.py --marts https://raw.githubusercontent.com/Clement0242/athlete-data-governance/main/data/marts
python analysis/build_report.py</code></pre>
  <div class="prose"><p>Data: SoccerMon, Midoglu C. et al. (2024), <em>Scientific Data</em> 11, 553 (CC-BY 4.0).
  MILS: De Larochelambert Q., Barlier K., github.com/quentincad2/MILS (GPL-3); this repository is GPL-3 as well.</p></div>
</div></section>
</main>

<footer><div class="enveloppe">
  <a href="https://clement0242.github.io">Clément Verdier</a> · data engineer &amp; sport scientist ·
  <a href="https://github.com/Clement0242/football-load-injury">source</a>
</div></footer>
</body>
</html>
"""

if __name__ == "__main__":
    main()
