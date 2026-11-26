"""Strict out-of-time test: everything used to forecast a contract is known on its signing date.
 - age adjustment: delta table from player-season pairs completed by the contract's info season only
 - forecast weights: fit on contracts signed earlier, with outcomes truncated at the last season before the test signings
 - price curve and market-adjustment weight: fit on earlier signings only
Test: contracts signed 2023-24 (first season 2023 or 2024). Train: contracts signed 2018-22."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json, sys
sys.path.insert(0, "src"); import aging
A = pd.read_csv("output/signings_all_scored.csv"); A = A[A.playerId.notna() & A.league_year.between(2018, 2024)].copy()
w = pd.read_csv("output/player_seasons_gar.csv"); w2 = aging.add_age(w.drop(columns=["age", "season_games", "gp_frac", "birth"], errors="ignore"), pd.read_csv("data/birthdates.csv"))
P = aging.pairs(w2); tabs = {}
for s in sorted(A.info_season.unique()):
    dt = aging.delta_table(P[P.season + 1 <= s]); tabs[s] = {(r.pos_group, r.age): r.delta for r in dt.itertuples()}
A["proj_strict"] = [r.GAR_per82 + tabs[r.info_season].get((r.pos_group, r.age), 0.0) for r in A.itertuples()]
print("max |proj82 strict - proj82 used| on test contracts: %.2f (mean abs %.3f)" % ((A.proj_strict - A.proj82).abs().max(), (A.proj_strict - A.proj82).abs().mean()))
_p = w[w.games_played >= 40]; sd = _p.groupby("season").GAR_raw_per82.std(); ref = sd.loc[2017:2019].mean(); look = w.set_index(["playerId", "season"])
hpg0 = [look.loc[(p, s), "h_all"] / look.loc[(p, s), "games_played"] if (p, s) in look.index else np.nan for p, s in zip(A.playerId, A.info_season)]; A["hpg0"] = hpg0
def outcome(r, last):   # realized GAR/82 at constant ice time, 2017-19 units, using only seasons <= last
    yrs = range(int(r.first_season), min(int(r.first_season) + int(r.term_years) - 1, last) + 1)
    x = w[(w.playerId == r.playerId) & w.season.isin(yrs)]
    if x.games_played.sum() < 40: return np.nan
    return (x.GAR_raw.values * np.array([ref / sd[s] for s in x.season])).sum() / x.h_all.sum() * 82 * r.hpg0
A["y_train"] = [outcome(r, 2022) if r.league_year <= 2022 else np.nan for r in A.itertuples()]   # known on the test signing date
A["y_test"] = [outcome(r, 2025) if r.league_year >= 2023 else np.nan for r in A.itertuples()]
A["p"] = A.proj_strict * A.era; A["a1"] = A.g1 * A.era; A["a2"] = A.g2 * A.era
X = lambda d: np.column_stack([np.ones(len(d)), d.p, d.a1, d.a2, d.hist1])
tr = A[(A.league_year <= 2022) & A.y_train.notna()]; te = A[A.league_year.isin((2023, 2024)) & A.y_test.notna()]
fc = np.linalg.lstsq(X(tr), tr.y_train.values, rcond=None)[0]; rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
yt = te.y_test.values * te.era.values; # y_test is in info-season units? outcome() already in 2017-19 units; do not rescale
yt = te.y_test.values
fte = X(te) @ fc
# price curve + adjustment fit on earlier signings only (prices known); resid for test uses that curve
Xp = lambda d, f: np.column_stack([np.ones(len(d)), f, d.ufa, d.age_at_signing - 27, np.log(d.term_years)])
ftr = X(tr) @ fc; pc = {}; rtr = np.zeros(len(tr)); rte = np.zeros(len(te))
for g in "FD":
    i = (tr.grp == g).values; j = (te.grp == g).values; pc[g] = np.linalg.lstsq(Xp(tr[i], ftr[i]), tr.share.values[i], rcond=None)[0]
    rtr[i] = tr.share.values[i] - Xp(tr[i], ftr[i]) @ pc[g]; rte[j] = te.share.values[j] - Xp(te[j], fte[j]) @ pc[g]
Xa = lambda f, r: np.column_stack([np.ones(len(f)), f, r]); ba = np.linalg.lstsq(Xa(ftr, rtr), tr.y_train.values, rcond=None)[0]
last = te.g0.values * te.era.values
res = dict(n_train=len(tr), n_test=len(te), forecast=rm(yt - fte), forecast_market_adjusted=rm(yt - Xa(fte, rte) @ ba), last_season_only=rm(yt - last), train_mean=rm(yt - tr.y_train.mean()), bias_forecast=float((yt - fte).mean()), weights=fc.round(3).tolist(), adj=ba.round(3).tolist())
print(json.dumps(res, indent=1)); json.dump(res, open("output/strict_backtest.json", "w"))
