"""Fair-market-value estimates for players on contracts that expire after 2026-27 (class of 2027) and 2027-28 (class of 2028).
FMV = what the market is likely to pay (cap share), from the price curve; three curves bracket the repricing question."""
import sys, json, re, unicodedata, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd, aging
R = json.load(open("output/grade_results.json")); norm = lambda s: re.sub(r"[^a-z ]", "", unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()).strip()
s = pd.read_csv("/tmp/all_signings.csv", parse_dates=["sign_date"]); s["key"] = s.player_name.map(norm)
s = s[~s.is_goalie].sort_values("sign_date"); cur = s.groupby("key").tail(1).copy(); cur["end"] = cur.first_season + cur.term_years - 1
w = pd.read_csv("output/player_seasons_gar.csv"); w = aging.add_age(w.drop(columns=["age", "season_games", "gp_frac", "birth"], errors="ignore"), pd.read_csv("data/birthdates.csv")); w["key"] = w.name.map(norm)
dl = {(r.pos_group, r.age): r.delta for r in aging.delta_table(aging.pairs(w)).itertuples()}; lastd = aging.delta_table(aging.pairs(w)); lastd = {pg: g.sort_values("age").iloc[-1] for pg, g in lastd.groupby("pos_group")}
delta = lambda pg, a: dl.get((pg, a), lastd[pg].delta if a > lastd[pg].age else 0.0)
_p = w[w.games_played >= 40]; sdv = _p.groupby("season").GAR_raw_per82.std(); era = sdv.loc[2017:2019].mean() / sdv[2025]
birth = pd.read_csv("data/birthdates.csv").set_index("playerId").birthDate; fc = np.array(R["forecast_coefs"])
def forecast(row, Y):
    """Forecast GAR/82 for a July-Y signing from 2025-26 data: project forward with the age curve to the information season Y-1."""
    pid = row.playerId; pg = row.pos_group; gs = w[(w.playerId == pid)].set_index("season").GAR_per82
    if 2025 not in gs.index: return None
    age25 = int(w[(w.playerId == pid) & (w.season == 2025)].age.iloc[0]); h = {2025: gs[2025]}
    for sx in range(2026, Y): h[sx] = h[sx - 1] + delta(pg, age25 + (sx - 1 - 2025))   # h[s] = forecast for season s
    I = Y - 1; ageI = age25 + (I - 2025)
    proj = (h[I] + delta(pg, ageI)) * era
    g1 = h.get(I - 1, gs.get(I - 1, np.nan)); g2 = h.get(I - 2, gs.get(I - 2, np.nan)); hist1 = 0 if np.isnan(g1) else 1
    g1 = proj / era if np.isnan(g1) else g1; g2 = g1 if np.isnan(g2) else g2
    return float(fc @ np.array([1, proj, g1 * era, g2 * era, hist1])), proj, gs[2025]
# ---- price curves
m = pd.read_csv("output/signings_all_scored.csv"); m["resid"] = m.share - m.v0
X = lambda d: np.column_stack([np.ones(len(d)), d.f, d.ufa, d.age_at_signing - 27, np.log(d.term_years)])
def fit(d): return {g: np.linalg.lstsq(X(d[d.grp == g]), d[d.grp == g].share.values, rcond=None)[0] for g in "FD"}
curves = {"low": {"F": np.array(R["price_static_F"]), "D": np.array(R["price_static_D"])}, "base": fit(m[m.league_year >= 2023])}
n26 = m[m.league_year == 2026].grp.value_counts().to_dict(); hi_src = m[m.league_year == 2026] if min(n26.get("F", 0), n26.get("D", 0)) >= 40 else m[m.league_year >= 2025]
curves["high"] = fit(hi_src); print("high curve fitted on", "2026 only" if hi_src is m[m.league_year == 2026] else "2025-26", len(hi_src), "contracts;", n26)
XT = lambda d: np.column_stack([np.ones(len(d)), d.f, d.age_at_signing - 27, d.ufa])
tm = {g: np.linalg.lstsq(XT(m[m.grp == g]), np.log(m[m.grp == g].term_years.values), rcond=None)[0] for g in "FD"}
print("expected-term model ln(term) ~ 1, f, age-27, UFA:", {g: tm[g].round(3).tolist() for g in tm})
rec = m[m.league_year >= 2023].copy(); rec["p"] = np.nan
for g in "FD": rec.loc[rec.grp == g, "p"] = X(rec[rec.grp == g]) @ curves["base"][g]
TOPFIT = 8.0; prem = float((rec.share - rec.p)[rec.p >= TOPFIT].mean()); r_ = rec.share - rec.p
sd_top = float(r_[rec.p >= TOPFIT].std()); sd_oth = float(r_[rec.p < TOPFIT].std())
sd_D = float(r_[(rec.grp == "D")].std()); print("residual SD D / F", round(sd_D, 2), round(float(r_[rec.grp == "F"].std()), 2))
LOWD = float(r_[(rec.grp == "D") & (rec.f > 0) & (rec.f < 5)].mean()); print("D low-forecast (0<f<5) mean residual, subtracted:", round(LOWD, 2))
def term_cap(f, grp):  # 75th percentile of terms actually signed (2023-26) by similar-forecast players of the same position group
    d = rec[(rec.grp == grp) & ((rec.f - f).abs() <= 2.5)]
    return int(np.ceil(d.term_years.quantile(.75))) if len(d) >= 15 else 7
print("top-end premium", round(prem, 2), "| residual SD top / other", round(sd_top, 2), round(sd_oth, 2))
def term_long(age):  # market's long deal by age: 90th percentile of UFA terms signed 2018-26, capped at 7 (the maximum with another team)
    return 7 if age <= 30 else 5 if age == 31 else 4 if age <= 34 else 3 if age <= 36 else 2 if age == 37 else 1
def cohort_cap(f, term, grp):  # defense with low forecast: never price above what similar signed D actually got (+0.5)
    if grp != "D" or f >= 5: return 99.0
    lo, hi = (-99, 1) if f <= 1 else (1, 5); d = rec[(rec.grp == "D") & (rec.f > lo) & (rec.f <= hi) & (rec.term_years.clip(upper=5) == min(term, 5))]
    return float(d.share.mean() + 0.5) if len(d) >= 10 else 99.0
def price(f, ufa, age, term, grp, which):
    x = np.array([1, f, ufa, age - 27, np.log(term)]); p = float(x @ curves[which][grp])
    p = p + (LOWD if (grp == "D" and 0 < f < 5) else 0.0)
    return min(20.0, max(0.8, min(p + (prem if (which == "base" and p >= TOPFIT) else 0.0), cohort_cap(f, term, grp) + (0.7 if which == "high" else 0.0))))   # 20% of the cap is the CBA maximum
out = []
for Y in (2027, 2028):
    c = cur[cur.end == Y - 1]
    for r in c.itertuples():
        cand = w[(w.key == r.key) & (w.season >= 2024)]
        if len(cand) == 0: continue
        pgw = cand.pos_group.mode().iloc[0] if not (cand.pos_group == ("D" if r.position == "D" else "F")).any() else ("D" if r.position == "D" else "F")
        cand = cand[cand.pos_group == pgw]
        if len(cand) == 0 or 2025 not in set(cand.season): out.append(dict(player=r.player_name, cls=Y, status="no 2025-26 data")); continue
        row = cand[cand.season == 2025].iloc[0]; fo = forecast(row, Y)
        if fo is None: continue
        f, proj, gar25 = fo; bd = pd.to_datetime(birth.get(row.playerId)); age = int((pd.Timestamp(f"{Y}-07-01") - bd).days // 365.25) if pd.notna(bd) else int(row.age) + (Y - 2026)
        ufa = int(r.sign_expiry != "RFA"); d = dict(cls=Y, player=r.player_name, team=r.team, pos=r.position, grp=pgw, age_at_signing=age, expiry_status=r.sign_expiry, cur_cap_hit=r.cap_hit_usd, cur_term_end=int(r.end) + 1, gp_2025=int(row.games_played), gar82_2025=gar25, forecast_f=f, playerId=int(row.playerId))
        tl = term_long(age); te = int(np.clip(round(float(np.exp(np.array([1, f, age - 27, ufa]) @ tm[pgw]))), 1, min(tl, term_cap(f, pgw)))); d["term_exp"] = te; d["term_max_age"] = tl
        for lab, term in (("short", 2), ("long", te)):
            for which in ("low", "base", "high"): d[f"share_{lab}_{which}"] = price(f, ufa, age, term, pgw, which)
        d["sd_pts"] = max(sd_D if pgw == "D" else 0.0, sd_top if d["share_long_base"] >= TOPFIT else sd_oth); d["flag"] = ("defense: provisional (GAR undercounts defensive value); " if pgw == "D" else "") + ("age 34+: few long deals, wide error; " if age >= 34 else "") + ("low 2025-26 games" if row.games_played < 40 else ""); out.append(d)
F = pd.DataFrame(out); F = F[F.get("forecast_f").notna()].copy() if "forecast_f" in F else F
F.to_csv("output/fmv_expiring.csv", index=False)
print(F.groupby("cls").size().to_dict(), "| by status:", F.groupby(["cls", "expiry_status"]).size().to_dict())
f27 = F[F.cls == 2027].sort_values("share_long_base", ascending=False)
f27["usd7_base_M"] = f27.share_long_base / 100 * 104 * 1.04; f27["usd7_low_M"] = f27.share_long_low / 100 * 104 * 1.04; f27["usd7_high_M"] = f27.share_long_high / 100 * 104 * 1.04
print(f27[["player", "team", "pos", "age_at_signing", "expiry_status", "term_exp", "cur_cap_hit", "forecast_f", "usd7_low_M", "usd7_base_M", "usd7_high_M", "flag"]].head(25).round(1).to_string(index=False))
print(F[F.player.str.contains("Frost")][["player", "cls", "forecast_f", "term_exp", "share_long_low", "share_long_base", "share_long_high", "share_short_base"]].round(2).to_string(index=False))

json.dump(dict(curves={k: {g: v[g].tolist() for g in v} for k, v in curves.items()}, prem=prem, sd_top=sd_top, sd_oth=sd_oth, sd_D=sd_D, lowD=LOWD, topfit=TOPFIT, term_model={g: tm[g].tolist() for g in tm}), open("output/fmv_params.json", "w"))
