"""Score 2023+ signings against the price curve fitted on 2018-2020 contracts (no refit): where does the market sit relative to the old curve?
usage: python src/score_new.py <combined_signings.csv>"""
import sys, warnings, json; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
import market_price as mp, aging
R = json.load(open("output/grade_results.json"))
s, w = mp.load(sys.argv[1]); s = s[s.sign_date >= "2023-06-30"]; m = mp.match(s, w); m["matched"] = m.playerId.notna()
look = w.set_index(["playerId", "season"])
def get(pid, ss, col):
    try: return look.loc[(pid, ss), col]
    except KeyError: return np.nan
w2 = pd.read_csv("output/player_seasons_gar.csv"); birth = pd.read_csv("data/birthdates.csv")
w2 = aging.add_age(w2.drop(columns=["age", "season_games", "gp_frac", "birth"], errors="ignore"), birth); dt = aging.delta_table(aging.pairs(w2)); dl = {(r.pos_group, r.age): r.delta for r in dt.itertuples()}
m = m[m.matched & (m.cap_hit_usd >= 1e6)].copy()
m["proj82"] = m.GAR_per82 + [dl.get((pg, a), 0.0) for pg, a in zip(m.pos_group, m.age)]
m["g0"] = m.GAR_per82; m["g1"] = [get(p, ss - 1, "GAR_per82") for p, ss in zip(m.playerId, m.info_season)]; m["g2"] = [get(p, ss - 2, "GAR_per82") for p, ss in zip(m.playerId, m.info_season)]
m["hist1"] = m.g1.notna().astype(int); m["g1"] = m.g1.fillna(m.g0); m["g2"] = m.g2.fillna(m.g1)
_p = pd.read_csv("output/player_seasons_gar.csv"); _p = _p[_p.games_played >= 40]; sdv = _p.groupby("season").GAR_raw_per82.std(); ref = sdv.loc[2017:2019].mean()
m["era"] = ref / m.info_season.map(sdv)     # rescale inputs to the 2018-20 fitting era (GAR spread grew after 2020)
if "--raw" not in sys.argv:
    for c in ["proj82", "g1", "g2"]: m[c] = m[c] * m.era
bf = np.array(R["forecast_coefs"]); m["f"] = np.column_stack([np.ones(len(m)), m.proj82, m.g1, m.g2, m.hist1]) @ bf
m["ufa"] = (m.sign_status != "RFA").astype(int); m["grp"] = np.where(m.position == "D", "D", "F")
CAPS = {2023: 83.5e6, 2024: 88e6, 2025: 95.5e6, 2026: 104e6}
m["cap_basis"] = m.first_season.map(lambda y: CAPS.get(int(y), 104e6)); m["cap_share_pct"] = m.cap_hit_usd / m.cap_basis * 100
m["price_fit"] = np.nan
for g, key in [("F", "price_coefs_F"), ("D", "price_coefs_D")]:
    i = m.grp == g; b = np.array(R[key]); d = m[i]
    m.loc[i, "price_fit"] = np.column_stack([np.ones(len(d)), d.f, d.ufa, d.age_at_signing - 27, np.log(d.term_years)]) @ b
m["resid"] = m.cap_share_pct - m.price_fit
m["gp_info"] = [get(p, ss, "games_played") for p, ss in zip(m.playerId, m.info_season)]
m.to_csv("output/new_signings_scored.csv", index=False)
m["young_rfa"] = (m.sign_status == "RFA") & (m.age_at_signing <= 23)
print(f"scored {len(m)} signings (matched, $1M+, 2023-06-30 on)")
print(m.groupby("league_year").agg(n=("resid", "size"), resid=("resid", "mean"), se=("resid", lambda x: x.std() / np.sqrt(len(x))), cap_share=("cap_share_pct", "mean"), f=("f", "mean")).round(2).to_string())
print("\nBy status and year (mean residual, cap-share points):"); print(m.groupby(["league_year", "sign_status"]).resid.agg(["count", "mean"]).round(2).unstack().to_string())
print("\nTop 15 cap shares:"); print(m.nlargest(15, "cap_share_pct")[["player_name", "league_year", "sign_status", "age_at_signing", "term_years", "cap_hit_usd", "cap_share_pct", "f", "price_fit", "resid", "gp_info"]].round(1).to_string(index=False))
big = m[m.cap_share_pct >= 10]; print("\nContracts at 10%+ of cap, by year: n, mean resid"); print(big.groupby("league_year").resid.agg(["count", "mean"]).round(2).to_string())

print("\nBy segment and year (mean residual, cap-share points; n):")
m["seg"] = np.where(m.young_rfa, "RFA age<=23", np.where(m.sign_status == "RFA", "RFA 24+", "UFA"))
print(m.groupby(["seg", "league_year"]).resid.agg(["count", "mean"]).round(2).unstack().to_string())
print("\nEra factor by info season:", {int(k): round(float(v), 2) for k, v in (ref / sdv.loc[2017:2025]).items()})
