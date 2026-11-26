"""Rolling-origin backtest of the forecast using every signing with realized seasons (2018-2025 league years).
Outcome y = realized GAR/82 over the contract's completed seasons (through 2025-26), at constant ice time, in 2017-19 GAR units (era-scaled)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json
R = json.load(open("output/grade_results.json"))
m = pd.read_csv("output/signings_all_scored.csv"); m = m[m.playerId.notna() & m.league_year.between(2018, 2025)].copy()
w = pd.read_csv("output/player_seasons_gar.csv")[["playerId", "season", "games_played", "h_all", "GAR_raw", "GAR_per82"]]; look = w.set_index(["playerId", "season"])
_p = pd.read_csv("output/player_seasons_gar.csv"); _p = _p[_p.games_played >= 40]; sd = _p.groupby("season").GAR_raw_per82.std(); ref = sd.loc[2017:2019].mean()
def get(pid, s, c):
    try: return look.loc[(pid, s), c]
    except KeyError: return np.nan
m["hpg0"] = [get(p, s, "h_all") / get(p, s, "games_played") for p, s in zip(m.playerId, m.info_season)]
y, gp = [], []
for r in m.itertuples():
    yrs = range(int(r.first_season), min(int(r.first_season) + int(r.term_years) - 1, 2025) + 1)
    x = w[(w.playerId == r.playerId) & w.season.isin(yrs)]; f = np.array([sd[r.info_season] / sd[s] for s in x.season])
    y.append(((x.GAR_raw.values * f).sum() / x.h_all.sum() * 82 * r.hpg0) if x.games_played.sum() >= 40 else np.nan); gp.append(x.games_played.sum())
m["y_info"] = y; m["gp_after"] = gp; m = m[m.y_info.notna()].copy()
m["era"] = ref / m.info_season.map(sd)
for c in ["proj82", "g1", "g2", "g0", "y_info"]: m[c + "_r"] = m[c] * m.era          # everything in 2017-19 GAR units
m["seasons_obs"] = (np.minimum(m.first_season + m.term_years - 1, 2025) - m.first_season + 1).astype(int)
m.to_csv("output/contracts_realized_all.csv", index=False)
print("contracts with realized outcomes by league year:", m.league_year.value_counts().sort_index().to_dict())
X = lambda d: np.column_stack([np.ones(len(d)), d.proj82_r, d.g1_r, d.g2_r, d.hist1])
def fit(d): return np.linalg.lstsq(X(d), d.y_info_r.values, rcond=None)[0]
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
old = np.array(R["forecast_coefs"])
print("\nOLD weights (fit 2018-20):", old.round(3))
rows = []
for name, tr_years, te_years in [("2021-22 (original test)", (2018, 2019, 2020), (2021, 2022)), ("2023", (2018, 2019, 2020, 2021, 2022), (2023,)), ("2024", (2018, 2019, 2020, 2021, 2022, 2023), (2024,)),
                                 ("2023-24", (2018, 2019, 2020, 2021, 2022), (2023, 2024)), ("2025 (1 season only)", (2018, 2019, 2020, 2021, 2022, 2023, 2024), (2025,))]:
    tr = m[m.league_year.isin(tr_years)]; te = m[m.league_year.isin(te_years)]
    b = fit(tr); te_y = te.y_info_r.values
    rows.append(dict(test=name, n_test=len(te), n_train=len(tr), old_weights=rm(te_y - X(te) @ old), refit_expanding=rm(te_y - X(te) @ b), last_season_only=rm(te_y - te.g0_r.values), train_mean=rm(te_y - tr.y_info_r.mean()),
                     bias_old=float((te_y - X(te) @ old).mean()), bias_refit=float((te_y - X(te) @ b).mean())))
print(pd.DataFrame(rows).round(2).to_string(index=False))
ball = fit(m[m.league_year <= 2024]); print("\nRefit on all 2018-24 (n=%d):" % (m.league_year <= 2024).sum(), ball.round(3))
json.dump(dict(forecast_coefs_all=ball.tolist(), n=int((m.league_year <= 2024).sum())), open("output/forecast_refit.json", "w"))
