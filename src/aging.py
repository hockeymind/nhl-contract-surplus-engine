"""Aging and availability curves from the GAR baseline plus birth dates (delta method).

Age = age on Feb 1 of the season's second calendar year (mid-season), in whole years.
Aging delta at age a = weighted mean change in unshrunk GAR per 82 between seasons played at age a and a+1,
for players with enough games in both seasons. Cumulative curves are anchored at age 25.
Survivorship caveat: only players who play in both seasons enter, so late-age curves are optimistic.
"""
import numpy as np, pandas as pd

MIN_FRAC = 0.4   # minimum share of a full season's games (handles 48-, 56- and 70-game seasons)

def add_age(w, birth):
    b = birth.copy(); b["birth"] = pd.to_datetime(b.birthDate, errors="coerce")
    w = w.merge(b[["playerId", "birth"]], on="playerId", how="left")
    mid = pd.to_datetime((w.season + 1).astype(str) + "-02-01")
    w["age"] = np.floor((mid - w.birth).dt.days / 365.25)
    w["season_games"] = w.groupby("season")["games_played"].transform("max")
    w["gp_frac"] = w.games_played / w.season_games
    return w

def pairs(w):
    a = w[(w.gp_frac >= MIN_FRAC) & w.age.notna()][["playerId", "season", "pos_group", "age", "gp_frac", "games_played", "GAR_raw_per82"]]
    n = a[["playerId", "season", "GAR_raw_per82", "gp_frac"]].copy(); n["season"] -= 1
    m = a.merge(n, on=["playerId", "season"], suffixes=("", "_n"))
    m["delta"] = m.GAR_raw_per82_n - m.GAR_raw_per82
    m["dgp"] = m.gp_frac_n - m.gp_frac
    m["wt"] = np.minimum(m.gp_frac, m.gp_frac_n)
    return m

def delta_table(m, col="delta", min_n=30):
    rows = []
    for (pg, age), g in m.groupby(["pos_group", "age"]):
        if len(g) < min_n: continue
        rows.append(dict(pos_group=pg, age=int(age), n=len(g), delta=np.average(g[col], weights=g.wt)))
    return pd.DataFrame(rows)

def cumulative(dt, anchor=25):
    out = []
    for pg, g in dt.groupby("pos_group"):
        g = g.set_index("age").delta
        ages = list(range(int(g.index.min()), int(g.index.max()) + 2))
        lvl = {anchor: 0.0}
        for a in range(anchor, ages[-1]):
            lvl[a + 1] = lvl[a] + g.get(a, np.nan)
        for a in range(anchor, ages[0], -1):
            lvl[a - 1] = lvl[a] - g.get(a - 1, np.nan)
        for a, v in sorted(lvl.items()): out.append(dict(pos_group=pg, age=a, level=v))
    return pd.DataFrame(out)

def bootstrap_ci(m, col="delta", reps=300, seed=1, min_n=30):
    rng = np.random.default_rng(seed)
    ids = m.playerId.unique(); groups = {k: g for k, g in m.groupby("playerId")}
    draws = []
    for _ in range(reps):
        pick = rng.choice(ids, len(ids))
        mm = pd.concat([groups[i] for i in pick])
        draws.append(delta_table(mm, col, min_n).set_index(["pos_group", "age"]).delta)
    d = pd.concat(draws, axis=1)
    return pd.DataFrame({"lo": d.quantile(.1, axis=1), "hi": d.quantile(.9, axis=1)})

def walk_forward_aging(w, m):
    """Project next-season raw GAR/82 as shrunk current + age delta (curve fitted without the target season)."""
    base = w[(w.gp_frac >= MIN_FRAC) & w.age.notna()][["playerId", "season", "pos_group", "age", "GAR_per82", "GAR_raw_per82", "gp_frac"]]
    n = base[["playerId", "season", "GAR_raw_per82", "gp_frac"]].copy(); n["season"] -= 1
    t = base.merge(n, on=["playerId", "season"], suffixes=("", "_n"))
    preds = []
    for s in sorted(t.season.unique()):
        fit = m[(m.season != s) & (m.season != s - 0)]   # exclude pairs starting in the target season
        dt = delta_table(fit, min_n=30)
        look = {(r.pos_group, r.age): r.delta for r in dt.itertuples()}
        x = t[t.season == s].copy()
        x["d"] = [look.get((pg, a), 0.0) for pg, a in zip(x.pos_group, x.age)]
        preds.append(x)
    x = pd.concat(preds); wt = x.gp_frac_n.values; y = x.GAR_raw_per82_n.values
    rm = lambda p: float(np.sqrt(np.average((y - p) ** 2, weights=wt)))
    return {"last season raw": rm(x.GAR_raw_per82.values), "shrunk": rm(x.GAR_per82.values),
            "shrunk + age curve": rm((x.GAR_per82 + x.d).values), "n": int(len(x))}
