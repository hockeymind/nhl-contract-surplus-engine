"""Step 3: contract grades. Parameters fitted on 2018-2020 contracts, checked on 2021-2022, then applied to all 2018-2022 contracts."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json
m = pd.read_csv("output/contracts_recal.csv")
tr_mask = m.league_year <= 2020
def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None); return b
Xf = lambda d: np.column_stack([np.ones(len(d)), d.proj82, d.g1, d.g2, d.hist1])
bf = ols(Xf(m[tr_mask]), m[tr_mask].y.values); m["f"] = Xf(m) @ bf
Xp = lambda d, f: np.column_stack([np.ones(len(d)), f, d.ufa, d.age_at_signing - 27, np.log(d.term_years)])
m["grp"] = np.where(m.position == "D", "D", "F"); bp = {}; m["price_fit"] = np.nan; m["k"] = np.nan
for g in ["F", "D"]:
    idx = (m.grp == g); bp[g] = ols(Xp(m[tr_mask & idx], m[tr_mask & idx].f), m[tr_mask & idx].cap_share_pct.values)
    m.loc[idx, "price_fit"] = Xp(m[idx], m[idx].f) @ bp[g]; m.loc[idx, "k"] = bp[g][1]
m["price_resid"] = m.cap_share_pct - m.price_fit
Xa = lambda d: np.column_stack([np.ones(len(d)), d.f, d.price_resid])
ba = ols(Xa(m[tr_mask]), m[tr_mask].y.values); m["f_star"] = Xa(m) @ ba
m["exp_surplus_pts"] = m.k * (m.f_star - m.f) - m.price_resid
m["real_surplus_pts"] = m.k * (m.y - m.f) - m.price_resid
res = dict(k_F=float(bp["F"][1]), k_D=float(bp["D"][1]), price_coefs_F=bp["F"].tolist(), price_coefs_D=bp["D"].tolist(), forecast_coefs=bf.tolist(), adj_coefs=ba.tolist(), n_train=int(tr_mask.sum()), n_test=int((~tr_mask).sum()))
for kk, v in res.items(): print(kk, v if not isinstance(v, (float, list)) else (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]))
json.dump(res, open("output/grade_results.json", "w")); m.to_csv("output/contracts_graded.csv", index=False)
