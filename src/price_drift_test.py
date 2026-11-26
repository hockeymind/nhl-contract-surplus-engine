"""Out-of-time test of price-curve updates. Target year T (2021-2026): predict each contract's cap share from information before T.
V0 static 2018-20 curve | V1 curve refit on all earlier years | V2 = V1 + learned top-end premium | V3 = V1 + last year's mean residual | V4 = V2 + last year's mean residual (non-top)"""
import sys, warnings, json; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
import market_price as mp, aging
R = json.load(open("output/grade_results.json"))
a = pd.read_csv(sys.argv[1]); b = pd.read_csv(sys.argv[2]); sg = pd.concat([a, b]).drop_duplicates("signing_id"); sg.to_csv("/tmp/all_signings.csv", index=False)
s, w = mp.load("/tmp/all_signings.csv"); s = s[s.sign_date >= "2018-06-30"]; m = mp.match(s, w); m = m[m.playerId.notna() & (m.cap_hit_usd >= 1e6)].copy()
look = w.set_index(["playerId", "season"])
get = lambda p, ss, c: look[c].get((p, ss), np.nan)
w2 = pd.read_csv("output/player_seasons_gar.csv"); w2 = aging.add_age(w2.drop(columns=["age", "season_games", "gp_frac", "birth"], errors="ignore"), pd.read_csv("data/birthdates.csv")); dl = {(r.pos_group, r.age): r.delta for r in aging.delta_table(aging.pairs(w2)).itertuples()}
m["proj82"] = m.GAR_per82 + [dl.get((pg, a_), 0.0) for pg, a_ in zip(m.pos_group, m.age)]
m["g0"] = m.GAR_per82; m["g1"] = [get(p, ss - 1, "GAR_per82") for p, ss in zip(m.playerId, m.info_season)]; m["g2"] = [get(p, ss - 2, "GAR_per82") for p, ss in zip(m.playerId, m.info_season)]
m["hist1"] = m.g1.notna().astype(int); m["g1"] = m.g1.fillna(m.g0); m["g2"] = m.g2.fillna(m.g1)
_p = w[w.games_played >= 40]; sdv = _p.groupby("season").GAR_raw_per82.std(); ref = sdv.loc[2017:2019].mean(); m["era"] = ref / m.info_season.map(sdv)
for c in ["proj82", "g1", "g2"]: m[c] = m[c] * m.era
m["f"] = np.column_stack([np.ones(len(m)), m.proj82, m.g1, m.g2, m.hist1]) @ np.array(R["forecast_coefs"])
CAPS = {2018: 79.5e6, 2019: 81.5e6, 2020: 81.5e6, 2021: 81.5e6, 2022: 82.5e6, 2023: 83.5e6, 2024: 88e6, 2025: 95.5e6, 2026: 104e6}
m["share"] = m.cap_hit_usd / m.first_season.map(lambda y: CAPS.get(int(y), 104e6 * (1.04 if int(y) == 2027 else 1))) * 100
m["ufa"] = (m.sign_status != "RFA").astype(int); m["grp"] = np.where(m.position == "D", "D", "F")
X = lambda d: np.column_stack([np.ones(len(d)), d.f, d.ufa, d.age_at_signing - 27, np.log(d.term_years)])
def fit_pred(train, test):
    out = pd.Series(np.nan, index=test.index)
    for g in "FD":
        tr_, te_ = train[train.grp == g], test[test.grp == g]
        if len(te_): out[te_.index] = X(te_) @ np.linalg.lstsq(X(tr_), tr_.share.values, rcond=None)[0]
    return out
TOP = 8.0
m["v0"] = np.nan
for g, key in [("F", "price_coefs_F"), ("D", "price_coefs_D")]:
    i = m.grp == g; m.loc[i, "v0"] = X(m[i]) @ np.array(R[key])
rows = []; m["ly"] = m.league_year
for T in range(2021, 2027):
    tr = m[m.ly < T]; te = m[m.ly == T].copy()
    te["v1"] = fit_pred(tr, te); trp = tr.copy(); trp["p1"] = fit_pred(trp[trp.ly < T], trp)   # in-sample fitted for residual bookkeeping
    res = trp.share - trp.p1; top = trp.p1 >= TOP
    prem = res[top].mean(); last = trp[trp.ly == T - 1]; lres = (last.share - last.p1)
    te["v2"] = te.v1 + np.where(te.v1 >= TOP, prem, 0.0)
    te["v3"] = te.v1 + lres.mean()
    te["v4"] = te.v2 + np.where(te.v1 >= TOP, 0.0, lres[last.p1 < TOP].mean())
    for v in ["v0", "v1", "v2", "v3", "v4"]:
        e = te.share - te[v]; tp = te.v1 >= TOP
        rows.append(dict(year=T, model=v, n=len(te), rmse=np.sqrt((e ** 2).mean()), bias=e.mean(), n_top=int(tp.sum()), rmse_top=np.sqrt((e[tp] ** 2).mean()), bias_top=e[tp].mean()))
r = pd.DataFrame(rows); print(r.round(2).to_string(index=False))
print("\nMean over target years 2021-2026:"); print(r.groupby("model")[["rmse", "rmse_top", "bias_top"]].mean().round(2).to_string())
print("\nMean over 2024-2026 only:"); print(r[r.year >= 2024].groupby("model")[["rmse", "rmse_top", "bias_top"]].mean().round(2).to_string())
m.to_csv("output/signings_all_scored.csv", index=False)
