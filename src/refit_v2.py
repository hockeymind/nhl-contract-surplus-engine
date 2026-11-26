"""v2 parameters: refit forecast weights, price curve and market-adjustment weight on every signing that has the data for it.
 - forecast weights + adjustment weight need realized outcomes -> contracts signed 2018-2024 (1,126)
 - price curve needs only prices -> all signings 2018-2026 (matched, $1M+)"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json
R1 = json.load(open("output/grade_results_v1.json"))
re = pd.read_csv("output/contracts_realized_all.csv").drop(columns=["f", "v0", "grp", "ly"], errors="ignore"); A = pd.read_csv("output/signings_all_scored_v1.csv")
Xf = lambda d: np.column_stack([np.ones(len(d)), d.proj82_r, d.g1_r, d.g2_r, d.hist1])
train = re[re.league_year <= 2024]; fc = np.linalg.lstsq(Xf(train), train.y_info_r.values, rcond=None)[0]
A["proj82_r"] = A.proj82 * A.era; A["g1_r"] = A.g1 * A.era; A["g2_r"] = A.g2 * A.era; A["f"] = Xf(A) @ fc
Xp = lambda d: np.column_stack([np.ones(len(d)), d.f, d.ufa, d.age_at_signing - 27, np.log(d.term_years)])
def fitp(d): return {g: np.linalg.lstsq(Xp(d[d.grp == g]), d[d.grp == g].share.values, rcond=None)[0] for g in "FD"}
pool = fitp(A); static = fitp(A[A.league_year <= 2020])
A["v0"] = np.nan
for g in "FD": A.loc[A.grp == g, "v0"] = Xp(A[A.grp == g]) @ pool[g]
re = re.merge(A[["signing_id", "f", "v0", "grp"]], on="signing_id", how="left"); re["resid"] = re.share - re.v0
tr = re[re.league_year <= 2024]; Xa = lambda d: np.column_stack([np.ones(len(d)), d.f, d.resid]); ba = np.linalg.lstsq(Xa(tr), tr.y_info_r.values, rcond=None)[0]
res = dict(version=2, k_F=float(pool["F"][1]), k_D=float(pool["D"][1]), price_coefs_F=pool["F"].tolist(), price_coefs_D=pool["D"].tolist(), price_static_F=static["F"].tolist(), price_static_D=static["D"].tolist(),
           forecast_coefs=fc.tolist(), adj_coefs=ba.tolist(), n_forecast_fit=int(len(train)), n_price_fit=int(len(A)))
json.dump(res, open("output/grade_results.json", "w")); A.drop(columns=["proj82_r", "g1_r", "g2_r"]).to_csv("output/signings_all_scored.csv", index=False)
print("forecast", fc.round(3), "n", len(train)); print("price F", pool["F"].round(3), "D", pool["D"].round(3)); print("static(<=2020) F", static["F"].round(3), "D", static["D"].round(3)); print("adjust a0,a1,a2", ba.round(3))
# graded table for contracts with >= 2 seasons possible (signed through 2023) in 2017-19 units
g = re[re.league_year <= 2023].copy(); g["y"] = g.y_info_r; g["k"] = np.where(g.grp == "D", pool["D"][1], pool["F"][1]); g["price_resid"] = g.resid; g["price_fit"] = g.v0
g["f_star"] = Xa(g.assign(resid=g.resid)) @ ba; g["exp_surplus_pts"] = g.k * (g.f_star - g.f) - g.price_resid; g["real_surplus_pts"] = g.k * (g.y - g.f) - g.price_resid
g["cap_share_pct"] = g.share; g["ufa"] = g.ufa.astype(int)
g.to_csv("output/contracts_graded.csv", index=False); print("graded contracts (2018-23):", len(g))
