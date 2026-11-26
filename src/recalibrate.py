"""Step 2: does adding earlier seasons fix the compressed forecast and absorb the price signal? Fit on 2018-2020 contracts, test on 2021-2022."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
m = pd.read_csv("output/signings_matched.csv")
m = m[m.playerId.notna() & (m.cap_hit_usd >= 1e6) & m.league_year.between(2018, 2022)].copy()
m["ufa"] = (m.sign_status != "RFA").astype(int); m["cap_share_pct"] = m.cap_share * 100
w = pd.read_csv("output/player_seasons_gar.csv")[["playerId", "season", "games_played", "h_all", "GAR_raw", "GAR_per82"]]
look = w.set_index(["playerId", "season"])
def get(pid, s, col):
    try: return look.loc[(pid, s), col]
    except KeyError: return np.nan
m["g0"] = m.GAR_per82
m["g1"] = [get(p, s - 1, "GAR_per82") for p, s in zip(m.playerId, m.info_season)]
m["g2"] = [get(p, s - 2, "GAR_per82") for p, s in zip(m.playerId, m.info_season)]
m["hist1"] = m.g1.notna().astype(int); m["hist2"] = m.g2.notna().astype(int)
m["g1"] = m.g1.fillna(m.g0); m["g2"] = m.g2.fillna(m.g1)
m["hpg0"] = [get(p, s, "h_all") / get(p, s, "games_played") for p, s in zip(m.playerId, m.info_season)]
_p = pd.read_csv("output/player_seasons_gar.csv"); _p = _p[_p.games_played >= 40]; sd = _p.groupby("season").GAR_raw_per82.std()
rf, rh = [], []
for r in m.itertuples():
    yrs = range(int(r.first_season), min(int(r.first_season) + int(r.term_years) - 1, 2025) + 1)
    x = w[(w.playerId == r.playerId) & w.season.isin(yrs)]
    f = np.array([sd[r.info_season] / sd[s] for s in x.season])
    rf.append(((x.GAR_raw.values * f).sum() / x.h_all.sum() * 82 * r.hpg0) if x.games_played.sum() >= 40 else np.nan); rh.append(x.games_played.sum())
m["y"] = rf; m = m[m.y.notna()].copy()
print(len(m), "contracts with realized outcomes")
m.to_csv("output/contracts_recal.csv", index=False)
