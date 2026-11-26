"""First pass: match signings to model player-seasons (information available on the signing date) and fit the market price of forecast GAR."""
import sys, re, unicodedata, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
import aging

norm = lambda s: re.sub(r"[^a-z ]", "", unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()).strip()
def info_season(dt):
    """Last regular season that had ended before the signing date (season labelled by start year)."""
    return dt.year - 1 if dt.month >= 5 else dt.year - 2

def load(sig_csv):
    s = pd.read_csv(sig_csv, parse_dates=["sign_date"]); s = s[~s.is_goalie & (s.level == "STD")].copy()
    s["league_year"] = (s.sign_date - pd.Timedelta(days=181)).dt.year
    s["info_season"] = s.sign_date.map(info_season); s["key"] = s.player_name.map(norm)
    w = pd.read_csv("output/player_seasons_gar.csv"); w["key"] = w.name.map(norm)
    return s, w

def match(s, w):
    cols = ["key", "season", "playerId", "pos_group", "games_played", "GAR_per82", "GAR_sd", "age", "team"]
    # fallback for nicknames (Mitch/Mitchell, Josh/Joshua): same last name and one first name is a prefix of the other
    w = w.copy(); w["last"] = w.key.str.split().str[-1]; w["first"] = w.key.str.split().str[0]
    by = {k: g for k, g in w.groupby(["last", "season"])}
    s = s.copy(); s["last"] = s.key.str.split().str[-1]; s["first"] = s.key.str.split().str[0]
    exact = set(zip(w.key, w.season)); fix = {}
    for r in s.itertuples():
        if (r.key, r.info_season) in exact: continue
        g = by.get((r.last, r.info_season))
        if g is None: continue
        c = g[[(f.startswith(r.first) or r.first.startswith(f)) and len(min(f, r.first, key=len)) >= 3 for f in g["first"]]]
        if len(c) >= 1: fix[r.signing_id] = c.key.iloc[0]
    s["key"] = [fix.get(i, k) for i, k in zip(s.signing_id, s.key)]
    print("nickname fixes applied:", len(fix))
    m = s.merge(w[cols].rename(columns={"season": "info_season", "team": "team_mp"}), on=["key", "info_season"], how="left")
    # same-name players: keep the one whose position group matches the signing position, else first
    m["pos_ok"] = np.where(m.position.eq("D"), m.pos_group.eq("D"), m.pos_group.eq("F"))
    m = m.sort_values(["signing_id", "pos_ok"], ascending=[True, False]).drop_duplicates("signing_id")
    return m

if __name__ == "__main__":
    s, w = load(sys.argv[1]); m = match(s, w)
    m["matched"] = m.playerId.notna()
    print("match rate by league year (all / at $1M+):")
    print(m.groupby("league_year").apply(lambda g: pd.Series({"signings": len(g), "matched": g.matched.mean().round(3),
          "matched_1m+": g[g.cap_hit_usd >= 1e6].matched.mean().round(3), "n_1m+": int((g.cap_hit_usd >= 1e6).sum())})).to_string())
    um = m[(~m.matched) & (m.cap_hit_usd >= 1e6)]
    print("\nunmatched $1M+ signings, biggest first:"); print(um.nlargest(12, "cap_hit_usd")[["player_name", "sign_date", "cap_hit_usd", "age_at_signing", "info_season", "playerId", "games_played"]].to_string(index=False))
    # projection: shrunk GAR/82 at the info season + age delta
    w2 = pd.read_csv("output/player_seasons_gar.csv"); birth = pd.read_csv("data/birthdates.csv")
    w2 = aging.add_age(w2.drop(columns=["age", "season_games", "gp_frac", "birth"], errors="ignore"), birth); am = aging.pairs(w2); dt = aging.delta_table(am)
    look = {(r.pos_group, r.age): r.delta for r in dt.itertuples()}
    m["proj82"] = m.GAR_per82 + [look.get((pg, a), 0.0) for pg, a in zip(m.pos_group, m.age)]
    d = m[m.matched & (m.cap_hit_usd >= 1e6) & m.league_year.between(2018, 2022)].copy()
    d["ufa"] = (d.sign_status != "RFA").astype(int); d["cap_share_pct"] = d.cap_share * 100
    y = d.cap_share_pct.values
    for label, cols_, names_ in [("without term", [np.ones(len(d)), d.proj82, d.ufa, d.age_at_signing - 27], ["intercept", "per +1 GAR/82", "UFA vs RFA", "per year of age over 27"]),
                                 ("with term", [np.ones(len(d)), d.proj82, d.ufa, d.age_at_signing - 27, np.log(d.term_years)], ["intercept", "per +1 GAR/82", "UFA vs RFA", "per year of age over 27", "ln(term)"])]:
        X = np.column_stack(cols_); b, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ b
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * res.var(ddof=X.shape[1])); r2 = 1 - res.var() / y.var()
        print(f"\nPrice model {label} (n={len(d)}, 2018-2022 league years): cap share % ; R2={r2:.2f}")
        for n_, bb, ss in zip(names_, b, se): print(f"  {n_:26s} {bb:7.3f}  (se {ss:.3f})")
    print("\nslope per league year (cap share points per +1 GAR/82):")
    for ly, g in d.groupby("league_year"):
        if len(g) < 30: continue
        Xg = np.column_stack([np.ones(len(g)), g.proj82, g.ufa]); bg, *_ = np.linalg.lstsq(Xg, g.cap_share_pct.values, rcond=None); print(f"  {ly}: {bg[1]:.3f}  (n={len(g)})")
    m.to_csv("output/signings_matched.csv", index=False)

    # --- Backtest: does the model, or the price, predict how the player actually performed over the contract? ---
    w3 = pd.read_csv("output/player_seasons_gar.csv")[["playerId", "season", "games_played", "GAR_raw_per82"]]
    rows = []
    for r in d[d.league_year <= 2021].itertuples():
        yrs = range(int(r.first_season), min(int(r.first_season) + int(r.term_years) - 1, 2025) + 1)
        x = w3[(w3.playerId == r.playerId) & w3.season.isin(yrs)]
        gp = x.games_played.sum()
        rows.append(dict(signing_id=r.signing_id, seasons_seen=len(x), gp=gp, realized82=np.average(x.GAR_raw_per82, weights=x.games_played) if gp >= 40 else np.nan))
    bt = d.merge(pd.DataFrame(rows), on="signing_id"); bt = bt[bt.realized82.notna()].copy()
    print(f"\nBacktest (2018-2021 signings with 40+ games since signing; n={len(bt)}); outcome = unshrunk GAR/82 over contract seasons played so far")
    def ols(cols_, names_):
        X = np.column_stack([np.ones(len(bt))] + cols_); y = bt.realized82.values; b, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ b
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * res.var(ddof=X.shape[1])); return b, se, 1 - res.var() / y.var()
    print("  corr(forecast, realized) =", round(np.corrcoef(bt.proj82, bt.realized82)[0, 1], 3), "| corr(cap share, realized) =", round(np.corrcoef(bt.cap_share_pct, bt.realized82)[0, 1], 3), "| corr(last-season raw...)")
    print("  RMSE forecast:", round(np.sqrt(np.mean((bt.realized82 - bt.proj82) ** 2)), 2), "| RMSE all-signed average:", round(bt.realized82.std(), 2))
    Xp = np.column_stack([np.ones(len(bt)), bt.proj82, bt.ufa, bt.age_at_signing - 27]); bp, *_ = np.linalg.lstsq(Xp, bt.cap_share_pct.values, rcond=None)
    bt["price_resid"] = bt.cap_share_pct - Xp @ bp
    b, se, r2 = ols([bt.proj82.values, bt.price_resid.values], ["forecast", "price above forecast-implied"])
    print(f"  realized = a + {b[1]:.2f}*forecast (se {se[1]:.2f}) + {b[2]:.2f}*price_residual (se {se[2]:.2f}); R2={r2:.2f}")
    print("  (price_residual in cap-share points: positive = paid more than the model's forecast implies)")
    bt.to_csv("output/backtest_contracts.csv", index=False)
