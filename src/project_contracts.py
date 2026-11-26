"""Multi-year contract projection: aging path + salary cap path.
 forecast GAR/82 in contract year t = market-adjusted year-1 forecast + age deltas (carried forward past the last age with data)
 value (share of cap, %) in year t   = value in year 1 + k * (forecast_t - forecast_1), floored at the league minimum share
 surplus in year t ($)               = value_t/100 * cap_t - cap hit      (cap hit is flat in dollars; the cap is not)"""
import sys, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
import aging
CAP = {2018: 79.5e6, 2019: 81.5e6, 2020: 81.5e6, 2021: 81.5e6, 2022: 82.5e6, 2023: 83.5e6, 2024: 88.0e6, 2025: 95.5e6, 2026: 104.0e6}
SCENARIOS = {"low": 0.02, "base": 0.04, "high": 0.06}      # cap growth per year after 2026-27 (assumptions, not forecasts)
VAL_FLOOR = 0.8                                            # league-minimum salary as % of cap
MAXT = 8
import os
LAM = json.load(open("output/aging_scale.json"))["aging_scale"] if os.path.exists("output/aging_scale.json") and "--raw" not in sys.argv else 1.0
def cap_path(g, seasons):
    return np.array([CAP[s] if s in CAP else CAP[2026] * (1 + g) ** (s - 2026) for s in seasons])

w = pd.read_csv("output/player_seasons_gar.csv"); birth = pd.read_csv("data/birthdates.csv")
w2 = aging.add_age(w.drop(columns=["age", "season_games", "gp_frac", "birth"], errors="ignore"), birth); dt = aging.delta_table(aging.pairs(w2))
last = {pg: g.sort_values("age").iloc[-1] for pg, g in dt.groupby("pos_group")}
look = {(r.pos_group, r.age): r.delta for r in dt.itertuples()}
def delta(pg, age): return look.get((pg, age), last[pg].delta if age > last[pg].age else 0.0)
my = json.load(open("output/multiyear.json")); A, B, S, C = my["coef"]; SC = my["scale"]

m = pd.read_csv("output/contracts_graded.csv"); m["pg"] = np.where(m.position == "D", "D", "F")
m["sd82"] = m.GAR_sd / m.games_played * 82
FB = json.load(open("output/fwd_bias.json"))  # big-forward correction (GAR/82), applied to value level only, not the aging path
rows = []
for r in m.itertuples():
    N = int(min(r.term_years, MAXT)); seasons = [int(r.first_season) + t for t in range(N)]
    f = []; fr = []
    for s in seasons:
        n = s - int(r.info_season); cum = sum(delta(r.pg, int(r.age) + j) for j in range(1, n)); f.append(r.f_star + LAM * cum); fr.append(r.f_star + cum)
    f = np.array(f); fr = np.array(fr); val1 = r.cap_share_pct + r.exp_surplus_pts + (r.k * FB["c_gar82"] / 1.0 if (r.pg == "F" and r.cap_share_pct >= FB["threshold_cap_share_pct"]) else 0.0)
    val = np.maximum(VAL_FLOOR, val1 + r.k * (f - f[0]))
    out = dict(signing_id=r.signing_id, **{f"f_y{t+1}": f[t] for t in range(N)}, **{f"f_raw_y{t+1}": fr[t] for t in range(N)}, **{f"val_y{t+1}": val[t] for t in range(N)})
    for name, g in {"flat": None, **SCENARIOS}.items():
        cp = np.full(N, CAP[int(r.first_season)] if int(r.first_season) in CAP else CAP[2026]) if g is None else cap_path(g, seasons)
        out[f"surplus_{name}"] = float(((val / 100) * cp - r.cap_hit_usd).sum() / 1e6)
    cp = cap_path(SCENARIOS["base"], seasons)
    var = SC ** 2 * (B * (N * r.sd82) ** 2 + S * N * (N + 1) * (2 * N + 1) / 6 + N * (A + C))
    out["surplus_band_sd"] = float(np.sqrt(var) * r.k / 100 * cp.mean() / 1e6)
    out["term_modeled"] = N; rows.append(out)
P = m.merge(pd.DataFrame(rows), on="signing_id")
P["surplus_base_lo"] = P.surplus_base - 1.2816 * P.surplus_band_sd; P["surplus_base_hi"] = P.surplus_base + 1.2816 * P.surplus_band_sd
P.to_csv("output/contracts_projected.csv", index=False)

print("Average projected surplus over the term ($M) by contract length, flat cap vs base cap path:")
print(P.groupby("term_years").agg(n=("signing_id", "size"), flat=("surplus_flat", "mean"), low=("surplus_low", "mean"), base=("surplus_base", "mean"), high=("surplus_high", "mean")).round(1).to_string())
# --- backtest of the aging path: forecast vs realized by contract year (2018-2023 signings, seasons through 2025) ---
_p = w[w.games_played >= 40]; sd = _p.groupby("season").GAR_raw_per82.std(); look_w = w.set_index(["playerId", "season"])
res = []
REF = sd.loc[2017:2019].mean()
for r in P[P.league_year <= 2023].itertuples():
    for t in range(1, int(r.term_modeled) + 1):
        s = int(r.first_season) + t - 1
        if s > 2025 or (r.playerId, s) not in look_w.index: continue
        x = look_w.loc[(r.playerId, s)]
        if x.games_played < 20: continue
        y = x.GAR_raw / x.h_all * 82 * r.hpg0 * REF / sd[s]
        res.append(dict(t=t, y=y, f_path=getattr(r, f"f_raw_y{t}"), f_flat=r.f_raw_y1))
R = pd.DataFrame(res)
print("\nBacktest by contract year (seasons played, 2018-2021 signings, 20+ games that season):")
g = R.groupby("t").apply(lambda d: pd.Series({"n": len(d), "mean_err_path": (d.y - d.f_path).mean(), "mean_err_flat": (d.y - d.f_flat).mean(), "rmse_path": np.sqrt(((d.y - d.f_path) ** 2).mean()), "rmse_flat": np.sqrt(((d.y - d.f_flat) ** 2).mean())}))
print(g.round(2).to_string())

# --- fit a scale lambda on cumulative aging: (y - f1) = lambda * (f_path - f1) ---
R["dy"] = R.y - R.f_flat; R["dp"] = R.f_path - R.f_flat; R["fp"] = R.f_path
q = R[R.t >= 2]
lam = (q.dy * q.dp).sum() / (q.dp ** 2).sum()
# bootstrap by row for SE
rng = np.random.default_rng(0); bs = []
for _ in range(500):
    s = q.sample(len(q), replace=True, random_state=int(rng.integers(1e9))); bs.append((s.dy * s.dp).sum() / (s.dp ** 2).sum())
print(f"\naging scale lambda = {lam:.2f} (bootstrap SE {np.std(bs):.2f}); rmse path {np.sqrt(((q.y-q.f_path)**2).mean()):.3f} vs scaled {np.sqrt(((q.dy-lam*q.dp)**2).mean()):.3f} vs flat {np.sqrt((q.dy**2).mean()):.3f}")
json.dump({"aging_scale": float(lam), "se": float(np.std(bs)), "n": int(len(q))}, open("output/aging_scale.json", "w"))
R[R.t>=2][["t","y","f_flat","f_path"]].to_csv("output/worked/aging_rows.csv", index=False)
