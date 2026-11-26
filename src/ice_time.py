"""Workstream B, phase 1: does per-minute even-strength impact fall as ice time rises? (game-level data)

Outcome: 5on5 score-and-venue-adjusted on-ice xG differential per 60 in one game.
Regressor: that player's 5on5 minutes in the game. Player fixed effects (within-player comparison), weighted by minutes.
Identification: instrument 5on5 minutes with the team's total 5on5 minutes in that game, which moves with penalties
(more special-teams time leaves less 5on5 time for everyone), not with a player's form. Standard errors clustered by player.
Placebo: next game's 5on5 minutes cannot cause this game's result, so it should show no effect.
"""
import glob
import numpy as np, pandas as pd

COLS = ["playerId", "gameId", "season", "playerTeam", "opposingTeam", "home_or_away", "gameDate", "position", "situation",
        "icetime", "OnIce_F_scoreVenueAdjustedxGoals", "OnIce_A_scoreVenueAdjustedxGoals"]

def load(paths, min_toi=3.0):
    d = pd.concat([pd.read_csv(p, usecols=COLS) for p in paths], ignore_index=True)
    d = d[(d.situation == "5on5") & (d.icetime > 0)].copy()
    d["toi"] = d.icetime / 60.0
    d["pos_group"] = np.where(d.position == "D", "D", "F")
    d["team5"] = d.groupby(["gameId", "playerTeam"])["toi"].transform("sum") / 5.0
    d["xgd60"] = (d.OnIce_F_scoreVenueAdjustedxGoals - d.OnIce_A_scoreVenueAdjustedxGoals) / (d.icetime / 3600.0)
    d["home"] = (d.home_or_away == "HOME").astype(float)
    d = d.sort_values(["playerId", "gameDate"])
    d["toi_next"] = d.groupby(["playerId", "season"])["toi"].shift(-1)
    return d[d.toi >= min_toi].copy()

def _demean(M, w, groups):
    out = np.empty_like(M, dtype=float)
    sw = pd.Series(w).groupby(groups).transform("sum").values
    for j in range(M.shape[1]):
        m = pd.Series(M[:, j] * w).groupby(groups).transform("sum").values / sw
        out[:, j] = M[:, j] - m
    return out

def tsls(y, X, Z, w, groups, cluster):
    """Weighted 2SLS with within-group demeaning. X: endogenous+exogenous regressors, Z: instruments+exogenous."""
    yt = _demean(y[:, None], w, groups)[:, 0]; Xt = _demean(X, w, groups); Zt = _demean(Z, w, groups)
    sw = np.sqrt(w)[:, None]
    Xs, Zs, ys = Xt * sw, Zt * sw, yt * np.sqrt(w)
    ZZi = np.linalg.pinv(Zs.T @ Zs)
    XZ = Xs.T @ Zs
    beta = np.linalg.solve(XZ @ ZZi @ XZ.T, XZ @ ZZi @ (Zs.T @ ys))
    e = ys - Xs @ beta
    Xh = Zs @ (ZZi @ (Zs.T @ Xs))                   # fitted regressors
    bread = np.linalg.inv(Xh.T @ Xh)
    scores = pd.DataFrame(Xh * e[:, None]).groupby(cluster).sum().values
    meat = scores.T @ scores
    ng = scores.shape[0]
    V = bread @ meat @ bread * ng / (ng - 1)
    return beta, np.sqrt(np.diag(V)), ng

def design(sub, quad=False, iv=True):
    """Returns y, X (toi [, toi^2] + controls), Z (instruments + controls)."""
    toi = sub.toi.values; team = sub.team5.values
    ctrl = [sub.home.values]
    opp = pd.get_dummies(sub.opposingTeam, drop_first=True).values.astype(float)
    C = np.column_stack(ctrl + [opp])
    cen_t, cen_z = toi.mean(), team.mean()
    t1 = toi - cen_t; z1 = team - cen_z
    if quad:
        X = np.column_stack([t1, t1 ** 2, C]); Z = np.column_stack([z1, z1 ** 2, C]) if iv else X
    else:
        X = np.column_stack([t1, C]); Z = np.column_stack([z1, C]) if iv else X
    return sub.xgd60.values, X, Z, cen_t

def run(d, pos, quad=False, iv=True, placebo=False, min_games=30):
    sub = d[d.pos_group == pos].copy()
    if placebo:
        sub = sub[sub.toi_next.notna()].copy(); sub["toi"] = sub.toi_next
    ng = sub.groupby(["playerId", "season"]).toi.transform("size")
    sub = sub[ng >= min_games]
    sub["grp"] = sub.playerId.astype(str) + "_" + sub.season.astype(str)
    y, X, Z, cen = design(sub, quad, iv)
    w = sub.toi.values / 60.0 if not placebo else sub.toi.values / 60.0
    beta, se, ncl = tsls(y, X, Z, w, sub.grp.values, sub.playerId.values)
    return dict(pos=pos, n=len(sub), players=int(ncl), beta=beta[:2 if quad else 1], se=se[:2 if quad else 1], center=cen,
                mean_toi=float(sub.toi.mean()))
