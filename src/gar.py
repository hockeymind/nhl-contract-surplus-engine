"""Workstream A baseline: goals above replacement (GAR) with intervals.

Components (all in goals, relative to a replacement-level pool at the same position group):
  ev_off  even-strength offence: on-ice xGF/60 relative to own team
  ev_def  even-strength defence: on-ice xGA/60 relative to own team (sign flipped)
  pp_off  power-play offence:    on-ice xGF/60 at 5on4 relative to team
  pk_def  penalty-kill defence:  on-ice xGA/60 at 4on5 relative to team (sign flipped)
  pen     net penalties drawn minus taken, valued at the league goal swing per minor
  fin     finishing: goals minus xG (own shots)

Each component rate is shrunk toward the position-group mean with an empirical-Bayes
estimate (true-talent variance tau2 from year-over-year covariance, noise k/hours from
the remaining variance). Intervals combine component posterior SDs assuming independence.
"""
import numpy as np
import pandas as pd

SITS = ["5on5", "5on4", "4on5", "all"]
# component -> (situation, hours threshold used when estimating variance components)
COMPONENTS = {
    "ev_off": ("5on5", 5.0),
    "ev_def": ("5on5", 5.0),
    "pp_off": ("5on4", 0.5),
    "pk_def": ("4on5", 0.5),
    "pen":    ("all", 10.0),
    "fin":    ("all", 10.0),
}
REPL_CUT = {"F": 12, "D": 6}      # forwards ranked 13+ and defensemen ranked 7+ by ice time on their team


def build_wide(d):
    """One row per player-season with rates and hours per component."""
    keys = ["playerId", "season"]
    base = d[d.situation == "all"][keys + ["name", "team", "position", "pos_group", "games_played"]].copy()
    out = base.set_index(keys)
    for s in SITS:
        x = d[d.situation == s].set_index(keys)
        out[f"h_{s}"] = x["icetime"] / 3600.0
        out[f"F_{s}"] = x["OnIce_F_xGoals"]
        out[f"A_{s}"] = x["OnIce_A_xGoals"]
    a = d[d.situation == "all"].set_index(keys)
    out["pd"] = a["penaltiesDrawn"]; out["pt"] = a["penalties"]
    out["goals"] = a["I_F_goals"]; out["ixg"] = a["I_F_xGoals"]
    out["h_all"] = a["icetime"] / 3600.0
    out = out.reset_index()
    for s in SITS:
        out[[f"h_{s}", f"F_{s}", f"A_{s}"]] = out[[f"h_{s}", f"F_{s}", f"A_{s}"]].fillna(0.0)
    # ice-time rank within team x position group (all situations), for the replacement pool
    out["toi_rank"] = out.groupby(["season", "team", "pos_group"])["h_all"].rank(ascending=False, method="first")
    return out


def add_rates(w):
    """Relative-to-team rates (per hour) and component hours."""
    for s in ["5on5", "5on4", "4on5"]:
        g = w.groupby(["season", "team"])
        th = g[f"h_{s}"].transform("sum")
        tF = g[f"F_{s}"].transform("sum") / th.replace(0, np.nan)
        tA = g[f"A_{s}"].transform("sum") / th.replace(0, np.nan)
        h = w[f"h_{s}"].replace(0, np.nan)
        w[f"relF_{s}"] = (w[f"F_{s}"] / h - tF)
        w[f"relA_{s}"] = (w[f"A_{s}"] / h - tA)
    h_all = w["h_all"].replace(0, np.nan)
    w["y_ev_off"] = w["relF_5on5"];  w["h_ev_off"] = w["h_5on5"]
    w["y_ev_def"] = -w["relA_5on5"]; w["h_ev_def"] = w["h_5on5"]
    w["y_pp_off"] = w["relF_5on4"];  w["h_pp_off"] = w["h_5on4"]
    w["y_pk_def"] = -w["relA_4on5"]; w["h_pk_def"] = w["h_4on5"]
    w["y_pen"] = (w["pd"] - w["pt"]) / h_all;  w["h_pen"] = w["h_all"]
    w["y_fin"] = (w["goals"] - w["ixg"]) / h_all; w["h_fin"] = w["h_all"]
    return w


def goals_per_minor(d):
    """League goal swing from one minor penalty (2 min), from team strength rates."""
    r = {}
    for s in ["5on5", "5on4", "4on5"]:
        x = d[d.situation == s]
        h = x.icetime.sum() / 3600.0
        r[s] = (x.OnIce_F_xGoals.sum() / h, x.OnIce_A_xGoals.sum() / h)
    gain = (r["5on4"][0] - r["5on5"][0]) + (r["5on5"][1] - r["5on4"][1])   # drawing team's swing per hour of PP
    # penalised team plays 4on5: symmetrical view as a cross-check
    loss = (r["5on5"][0] - r["4on5"][0]) + (r["4on5"][1] - r["5on5"][1])
    return (gain + loss) / 2.0 * (2.0 / 60.0), gain * (2.0 / 60.0), loss * (2.0 / 60.0)


def estimate_variance_components(w, comp, hmin):
    """tau2 (true-talent variance) and k (noise coefficient, var = k/hours) per position group."""
    res = {}
    for pg in ["F", "D"]:
        x = w[(w.pos_group == pg) & (w[f"h_{comp}"] >= hmin) & w[f"y_{comp}"].notna()].copy()
        mu = np.average(x[f"y_{comp}"], weights=x[f"h_{comp}"])
        a = x[["playerId", "season", f"y_{comp}", f"h_{comp}"]].copy()
        b = a.copy(); b["season"] -= 1
        m = a.merge(b, on=["playerId", "season"], suffixes=("", "_n"))
        wt = np.minimum(m[f"h_{comp}"], m[f"h_{comp}_n"])
        tau2 = np.sum(wt * (m[f"y_{comp}"] - mu) * (m[f"y_{comp}_n"] - mu)) / np.sum(wt)
        tau2 = max(tau2, 1e-9)
        # E[(y-mu)^2] = tau2 + k/h  ->  k = mean(h*((y-mu)^2 - tau2))
        k = np.mean(x[f"h_{comp}"] * ((x[f"y_{comp}"] - mu) ** 2 - tau2))
        res[pg] = dict(mu=mu, tau2=tau2, k=max(k, 1e-9), n_pairs=len(m), n_obs=len(x))
    return res


CAL = {"fin": 0.89}   # finishing: slope of next-season rate on shrunk rate (calibration)
def shrink(w, comp, vc, cal=None):
    mu = w.pos_group.map(lambda g: vc[g]["mu"]); tau2 = w.pos_group.map(lambda g: vc[g]["tau2"])
    k = w.pos_group.map(lambda g: vc[g]["k"])
    h = w[f"h_{comp}"]
    y = w[f"y_{comp}"].fillna(mu)
    noise = k / h.replace(0, np.nan)
    wgt = (tau2 / (tau2 + noise)).fillna(0.0)
    c = CAL.get(comp, 1.0) if cal is None else cal
    post = mu + c * wgt * (y - mu)
    post_var = (tau2 * noise / (tau2 + noise)).fillna(tau2)
    return post, np.sqrt(post_var), wgt


def replacement_levels(w, comp):
    """Hours-weighted raw mean of the replacement pool, per season and position group."""
    pool = w[(w.toi_rank > w.pos_group.map(REPL_CUT)) & (w.games_played >= 10) & w[f"y_{comp}"].notna()]
    def wm(g):
        return np.average(g[f"y_{comp}"], weights=g[f"h_{comp}"]) if g[f"h_{comp}"].sum() > 0 else np.nan
    return pool.groupby(["season", "pos_group"]).apply(wm, include_groups=False).rename(f"repl_{comp}")


def run(d, repl_cut=None):
    if repl_cut:
        REPL_CUT.update(repl_cut)
    w = add_rates(build_wide(d))
    gpm, gpm_gain, gpm_loss = goals_per_minor(d)
    vcs = {}
    for comp, (sit, hmin) in COMPONENTS.items():
        vc = estimate_variance_components(w, comp, hmin)
        vcs[comp] = vc
        post, sd, wgt = shrink(w, comp, vc)
        rl = replacement_levels(w, comp)
        w = w.merge(rl.reset_index(), on=["season", "pos_group"], how="left")
        conv = gpm if comp == "pen" else 1.0      # penalty rate is in minors per hour -> goals
        w[f"postraw_{comp}"] = shrink(w, comp, vc, cal=1.0)[0]
        w[f"post_{comp}"] = post; w[f"sd_{comp}"] = sd; w[f"wgt_{comp}"] = wgt
        w[f"gar_{comp}"] = (post - w[f"repl_{comp}"]) * w[f"h_{comp}"] * conv
        w[f"garsd_{comp}"] = sd * w[f"h_{comp}"] * conv
        w[f"garraw_{comp}"] = (w[f"y_{comp}"].fillna(0) - w[f"repl_{comp}"]) * w[f"h_{comp}"] * conv
    comps = list(COMPONENTS)
    w["GAR"] = w[[f"gar_{c}" for c in comps]].sum(axis=1)
    w["GAR_sd"] = np.sqrt(sum(w[f"garsd_{c}"] ** 2 for c in comps))
    w["GAR_lo"] = w["GAR"] - 1.2816 * w["GAR_sd"]; w["GAR_hi"] = w["GAR"] + 1.2816 * w["GAR_sd"]   # 80% interval
    w["GAR_raw"] = w[[f"garraw_{c}" for c in comps]].sum(axis=1)
    w["GAR_per82"] = np.where(w.games_played > 0, w.GAR / w.games_played * 82, np.nan)
    w["GAR_raw_per82"] = np.where(w.games_played > 0, w.GAR_raw / w.games_played * 82, np.nan)
    w["WAR"] = w.GAR / 6.0
    return w, vcs, dict(goals_per_minor=gpm, drawing_side=gpm_gain, penalised_side=gpm_loss)


def walk_forward(w, min_gp=30):
    """Predict next season's raw GAR/82 from this season's data. Players need min_gp games in both seasons."""
    a = w[w.games_played >= min_gp][["playerId", "season", "GAR_per82", "GAR_raw_per82", "games_played"]]
    nxt = a[["playerId", "season", "GAR_raw_per82", "games_played"]].copy(); nxt["season"] -= 1
    prv = a[["playerId", "season", "GAR_raw_per82"]].copy(); prv["season"] += 1
    m = a.merge(nxt, on=["playerId", "season"], suffixes=("", "_n"))
    wt = m.games_played_n.values; y = m.GAR_raw_per82_n.values
    mean_all = float(np.average(m.GAR_raw_per82, weights=m.games_played))
    preds = {"last season raw": m.GAR_raw_per82.values,
             "shrunk (this model)": m.GAR_per82.values,
             "all-player average": np.full(len(m), mean_all)}
    m2 = m.merge(prv.rename(columns={"GAR_raw_per82": "prev"}), on=["playerId", "season"])
    out = {k: float(np.sqrt(np.average((y - p) ** 2, weights=wt))) for k, p in preds.items()}
    wt2 = m2.games_played_n.values; y2 = m2.GAR_raw_per82_n.values
    out["avg of last 2 seasons raw (subset)"] = float(np.sqrt(np.average((y2 - (m2.GAR_raw_per82 + m2.prev) / 2) ** 2, weights=wt2)))
    out["shrunk, same subset"] = float(np.sqrt(np.average((y2 - m2.GAR_per82) ** 2, weights=wt2)))
    out["n_pairs"] = int(len(m)); out["n_pairs_subset"] = int(len(m2))
    return out
