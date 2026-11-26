"""Grade and project the contracts signed 2023-06-30 onward (no realized outcomes yet): same formulas as grade.py / project_contracts.py with parameters fixed from 2018-20."""
import sys, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd, aging
R = json.load(open("output/grade_results.json")); MY = json.load(open("output/multiyear.json")); LAM = json.load(open("output/aging_scale.json"))["aging_scale"]
A, B, S, C = MY["coef"]; SC = MY["scale"]
CAP = {2018: 79.5e6, 2019: 81.5e6, 2020: 81.5e6, 2021: 81.5e6, 2022: 82.5e6, 2023: 83.5e6, 2024: 88e6, 2025: 95.5e6, 2026: 104e6}
G = {"low": .02, "base": .04, "high": .06}; FLOOR = 0.8
capp = lambda g, seasons: np.array([CAP[s] if s in CAP else CAP[2026] * (1 + g) ** (s - 2026) for s in seasons])
m = pd.read_csv("output/signings_all_scored.csv", parse_dates=["sign_date"]); m = m[m.league_year >= 2023].copy()
m["cap_share_pct"] = m.share; m["resid"] = m.share - m.v0
m["k"] = np.where(m.grp == "D", R["price_coefs_D"][1], R["price_coefs_F"][1])
a = np.array(R["adj_coefs"]); m["f_star"] = a[0] + a[1] * m.f + a[2] * m.resid
m["exp_surplus_pts"] = m.k * (m.f_star - m.f) - m.resid
m["sd82"] = m.GAR_sd / m.games_played * 82
w = pd.read_csv("output/player_seasons_gar.csv"); w = aging.add_age(w.drop(columns=["age", "season_games", "gp_frac", "birth"], errors="ignore"), pd.read_csv("data/birthdates.csv"))
dt = aging.delta_table(aging.pairs(w)); last = {pg: g.sort_values("age").iloc[-1] for pg, g in dt.groupby("pos_group")}; look = {(r.pos_group, r.age): r.delta for r in dt.itertuples()}
delta = lambda pg, age: look.get((pg, age), last[pg].delta if age > last[pg].age else 0.0)
FB = json.load(open("output/fwd_bias.json"))
rows = []
for r in m.itertuples():
    N = int(min(r.term_years, 8)); seasons = [int(r.first_season) + t for t in range(N)]
    f = np.array([r.f_star + LAM * sum(delta(r.grp, int(r.age) + j) for j in range(1, s - int(r.info_season))) for s in seasons])
    val1 = r.cap_share_pct + r.exp_surplus_pts + (r.k * FB["c_gar82"] if (r.grp == "F" and r.cap_share_pct >= FB["threshold_cap_share_pct"]) else 0.0); val = np.maximum(FLOOR, val1 + r.k * (f - f[0]))
    out = dict(signing_id=r.signing_id, term_modeled=N, val1=val1, **{f"val_y{t+1}": val[t] for t in range(N)})
    for nm, g in G.items(): out[f"surplus_{nm}"] = float(((val / 100) * capp(g, seasons) - r.cap_hit_usd).sum() / 1e6)
    out["surplus_flat"] = float(((val / 100) * CAP[min(int(r.first_season), 2026)] - r.cap_hit_usd).sum() / 1e6)
    var = SC ** 2 * (B * (N * r.sd82) ** 2 + S * N * (N + 1) * (2 * N + 1) / 6 + N * (A + C))
    out["surplus_band_sd"] = float(np.sqrt(var) * r.k / 100 * capp(G["base"], seasons).mean() / 1e6); rows.append(out)
P = m.merge(pd.DataFrame(rows), on="signing_id"); P["top"] = (P.val1 >= 8).astype(int)
P.to_csv("output/contracts_projected_new.csv", index=False)
print(len(P), "contracts;", int(P.top.sum()), "with year-1 value >= 8% of cap")
c = ["player_name", "team", "league_year", "term_years", "cap_hit_usd", "cap_share_pct", "val1", "surplus_base"]
print("\nBest 10 (total term surplus, $M, 4% cap growth):"); print(P.nlargest(10, "surplus_base")[c].round(1).to_string(index=False))
print("\nWorst 10:"); print(P.nsmallest(10, "surplus_base")[c].round(1).to_string(index=False))
print("\nBy signing year (mean surplus, $M):", P.groupby("league_year").surplus_base.mean().round(2).to_dict())
