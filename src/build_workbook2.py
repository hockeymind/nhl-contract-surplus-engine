import pandas as pd, numpy as np, json
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter as L
P = pd.read_csv("output/contracts_projected_new.csv").sort_values("surplus_base").reset_index(drop=True)
FB = json.load(open("output/fwd_bias.json")); FP = json.load(open("output/fmv_params.json")); GR = json.load(open("output/grade_results.json"))
F = pd.read_csv("output/fmv_expiring.csv"); LAM = json.load(open("output/aging_scale.json"))["aging_scale"]
wb = Workbook(); H = PatternFill("solid", fgColor="1F3A5F"); HF = Font(bold=True, color="FFFFFF"); INP = Font(color="0000FF", bold=True); YEL = PatternFill("solid", fgColor="FFF2CC")
def hdr(ws, r, cols):
    for i, c in enumerate(cols, 1):
        x = ws.cell(r, i, c); x.fill = H; x.font = HF; x.alignment = Alignment(wrap_text=True, vertical="center")
# README
ws = wb.active; ws.title = "README"
txt = ["NHL Contract Surplus Engine: 2023-26 contract grades and fair-market-value (FMV) projections",
"",
"WHAT THIS IS: model estimates, not facts. Surplus = (what the player's projected production is worth under the model, as % of cap x cap) minus the cap hit, summed over the contract.",
f"Aging: the age curve is scaled by {LAM:.2f} (SE 0.10, 1,440 player-years). Cap growth is a user input (Inputs sheet). Parameters are the v2 refit: forecast weights and market-adjustment weight fitted on 2018-24 contracts with realized seasons (1,126), price curve fitted on all 2018-26 signings (1,515).",
"",
"SHEETS",
"Inputs: change the yellow cells (cap growth, repricing bump, top-tier threshold, start season). Everything else recalculates.",
"Contracts: 684 contracts signed 2023-06-30 onward ($1M+, non-ELC). Zero-repricing column = model as validated. Scenario column = with your repricing bump.",
"Leaders: best and worst contracts under the current scenario.",
"FMV_2027 / FMV_2028: projected market price for players whose current deals expire after 2026-27 / 2027-28 (low / base / high price curves, 80% band).",
"",
"HOW THE REPRICING INPUT WORKS: from the start season on, every contract SIGNED BEFORE that season has its value path raised by the bump (cap-share points): top tier = year-1 value at or above the threshold, other = everyone else. Contracts signed at the new prices are not bumped (already priced in).",
"Why it exists: the static price curve (2018-20) cannot see the 2026 top-end jump. Observed top-tier gap in 2026 signings is about +2.5 points (n=17, correlated deals, so treat as an upper-ish reading). Mid-tier observed about -0.4.",
"",
"FMV METHOD: price curve (cap share ~ forecast GAR, UFA, age, ln term), fit separately for F and D. Low = 2018-20 curve (refit with the v2 forecast). Base = 2023-26 refit plus a "+f"{FP['prem']:.2f}-point premium on fitted shares of 8%+. High = 2025-26 refit. Term shown = model-expected term for that player (capped by age); short = 2 years.",
"80% band = +/-1.2816 x residual SD (2.18 points for 8%+ fitted shares, 1.53 otherwise, 1.72 for defensemen if larger). Dollars = share x cap for the signing season, with cap growth from Inputs. Max 20% of cap.",
"",
"KNOWN LIMITS (read before posting anything)",
"1. DEFENSE IS PROVISIONAL: GAR undercounts defensive value. Defense rows are labeled, and leaderboards exclude defensemen by default (Inputs!B10). Low-forecast defense prices are capped at what similar signed defensemen actually received (+0.5 pt) and shifted down 0.32 pt (in-sample), because longer terms go to better players.",
"2. Forecast uses 2025-26 and prior; 2026-27 games are not in the data. Age 34+ rows are flagged and unreliable.",
"3. Players whose current deal is an ELC or under $1M (many RFAs) are missing from the FMV lists. Rows with odd contracts carry a 'check' flag.",
"4. 2028 class is preliminary (two more seasons of change). 2026 signings have no realized outcomes, so the 30% price-to-performance recovery cannot be validated.",
"5. Contracts starting in 2027-28 (e.g. Makar) use a projected 2027-28 cap of $108.2M (4% growth) for cap share, not the 2026-27 cap.",
"5b. The Data check flag column marks contracts that look inconsistent (e.g. a 36-year-old on a 6-year $1.5M deal). Verify against PuckPedia before quoting them.",
"5c. BIG-FORWARD CORRECTION: forwards signed at 7%+ of the cap have beaten their forecasts. A correction of +" + str(FB["c_gar82"]) + " GAR/82 (SE " + str(FB["se"]) + ", " + str(FB["n"]) + " contracts signed 2018-23) raises their value level. It is unstable by period: -0.76 for 2018-20 signings, +2.3 for 2021+. A correction fitted on 2018-20 alone would NOT have helped the 2021-22 test, so this is the pooled average, applied in-sample, and likely still too low for recent deals.",
"5d. Age 34+ on 5+ year deals (e.g. Gudas, 36, 6 years at $1.5M) are consistent with the source data but look like retirement-style structures, so they are flagged and excluded from leaders.",
"6. Bump is a scenario, not a forecast. The out-of-time test showed a learned year effect did NOT improve held-out accuracy; a top-end premium did, partially."]
for i, t in enumerate(txt, 1):
    ws.cell(i, 1, t).alignment = Alignment(wrap_text=True, vertical="top")
ws["A1"].font = Font(bold=True, size=14); ws.column_dimensions["A"].width = 150
# Inputs
wi = wb.create_sheet("Inputs"); wi["A1"] = "Inputs (yellow = change me)"; wi["A1"].font = Font(bold=True, size=14)
inp = [("Cap growth per year after 2026-27", 0.04, "0.0%", "Low 2% / base 4% / high 6% in the model"),
       ("Top-tier threshold (year-1 value, % of cap)", 8, "0.0", "Contracts at or above this are 'top tier'"),
       ("Repricing bump, top tier (cap-share points)", 0, "0.00", "Try 2.5 (observed in 2026 signings). 0 = model as validated"),
       ("Repricing bump, other contracts (cap-share points)", 0, "0.00", "Try 0.5-1.0 for a Frost-style mid-tier lift. Observed mid-tier: about -0.4"),
       ("Bump start season (2026 = 2026-27)", 2026, "0", "Applies to contracts signed before this league year, in seasons from here on"),
       ("Value floor (% of cap)", 0.8, "0.0", ""), ("Band z for 80%", 1.2816, "0.0000", ""),
       ("Include defensemen in leaderboards (1 = yes)", 0, "0", "Default 0: defense grades are provisional (GAR undercounts defensive value)")]
for i, (a, v, f, n) in enumerate(inp, 3):
    wi.cell(i, 1, a); c = wi.cell(i, 2, v); c.number_format = f; c.font = INP; c.fill = YEL; wi.cell(i, 3, n)
wi["A11"] = "Cap by season ($M)"; wi["A11"].font = Font(bold=True)
hdr(wi, 12, ["Season start", "Cap $M", "Note"])
act = {2023: 83.5, 2024: 88, 2025: 95.5, 2026: 104}
for j, s in enumerate(range(2023, 2038)):
    r = 13 + j; wi.cell(r, 1, s)
    if s in act: wi.cell(r, 2, act[s]); wi.cell(r, 3, "actual")
    else: wi.cell(r, 2, f"=B{r-1}*(1+$B$3)"); wi.cell(r, 3, "projected")
    wi.cell(r, 2).number_format = "0.0"
CAPR = "Inputs!$B$13:$B$27"
wi.column_dimensions["A"].width = 52; wi.column_dimensions["B"].width = 14; wi.column_dimensions["C"].width = 70
# Contracts
wc = wb.create_sheet("Contracts"); n = len(P)
cols = ["Player", "Team", "Pos", "Signed (league yr)", "First season", "Term (yrs)", "Yrs modeled", "Cap hit $M", "Cap share %", "Yr-1 value %", "Top tier?"] \
     + [f"Value Y{t} %" for t in range(1, 9)] + [f"Cap Y{t} $M" for t in range(1, 9)] + [f"Surplus Y{t} $M (scenario)" for t in range(1, 9)] \
     + ["Total surplus, zero repricing $M", "Total surplus, scenario $M", "Remaining from 2026-27, scenario $M", "Scenario minus zero $M", "80% band +/- $M (SD x 1.28)", "Python check (4% cap, zero)", "Data check flag", "Reliability", "Rank value (leaders)"]
hdr(wc, 1, cols); wc.row_dimensions[1].height = 45
V0, C0, S0 = 12, 20, 28; T = 36
for i, r in enumerate(P.itertuples(), 2):
    N = int(r.term_modeled); E = int(r.first_season)
    vals = [r.player_name, r.team, r.position, int(r.league_year), E, int(r.term_years), N, r.cap_hit_usd / 1e6, r.cap_share_pct, r.val1, f"=IF(J{i}>=Inputs!$B$4,1,0)"]
    for k, v in enumerate(vals, 1): wc.cell(i, k, v)
    for t in range(1, 9):
        if t <= N: wc.cell(i, V0 + t - 1, getattr(r, f"val_y{t}"))
        wc.cell(i, C0 + t - 1, f'=IF({t}>$G{i},"",INDEX({CAPR},$E{i}+{t-1}-2022))')
        v = f"{L(V0+t-1)}{i}"; c = f"{L(C0+t-1)}{i}"; s = f"($E{i}+{t-1})"
        wc.cell(i, S0 + t - 1, f'=IF({t}>$G{i},"",(MAX(Inputs!$B$8,{v}+IF(AND($D{i}<Inputs!$B$7,{s}>=Inputs!$B$7),IF($K{i}=1,Inputs!$B$5,Inputs!$B$6),0))/100*{c})-$H{i})')
    vr = f"{L(V0)}{i}:{L(V0+7)}{i}"; cr = f"{L(C0)}{i}:{L(C0+7)}{i}"; sr = f"{L(S0)}{i}:{L(S0+7)}{i}"
    wc.cell(i, T, f"=SUMPRODUCT({vr},{cr})/100-H{i}*G{i}")
    wc.cell(i, T + 1, f"=SUM({sr})")
    k = max(0, min(N, 2026 - E)); wc.cell(i, T + 2, f"=AK{i}" if k == 0 else (f"=AK{i}-SUM({L(S0)}{i}:{L(S0+k-1)}{i})" if k < N else "=0"))
    wc.cell(i, T + 3, f"=AK{i}-AJ{i}"); wc.cell(i, T + 4, r.surplus_band_sd * 1.2816); wc.cell(i, T + 5, r.surplus_base)
    fl = []
    if r.age_at_signing >= 34 and r.term_years >= 5: fl.append("age 34+ on a 5+ yr term: likely retirement/LTIR-style structure; model assumes he plays every year")
    if r.cap_hit_usd < 2e6 and r.term_years >= 5: fl.append("sub-$2M cap hit on a 5+ yr term: same structure risk")
    if r.term_years > 8: fl.append("term over 8 modeled as 8")
    rel = ("Provisional: GAR undercounts defensive value" if r.position == "D" else "Standard") + ("; big-forward correction applied (+" + str(FB["c_gar82"]) + " GAR/82)" if (r.position != "D" and r.cap_share_pct >= 7) else "")
    wc.cell(i, T + 6, "; ".join(fl)); wc.cell(i, T + 7, rel)
    wc.cell(i, T + 8, f'=IF(AND(AP{i}="",OR(Inputs!$B$10=1,C{i}<>"D")),AK{i},"")')
for row in wc.iter_rows(min_row=2, max_row=n + 1, min_col=8, max_col=T + 5):
    for c in row: c.number_format = "0.0"
wc.freeze_panes = "B2"; wc.auto_filter.ref = f"A1:{L(T+8)}{n+1}"; wc.column_dimensions[L(T+6)].width = 46; wc.column_dimensions[L(T+7)].width = 46; wc.column_dimensions["A"].width = 24; wc.column_dimensions["B"].width = 22
for c in range(C0, S0): wc.column_dimensions[L(c)].outlineLevel = 1; wc.column_dimensions[L(c)].hidden = True
# Leaders
wl = wb.create_sheet("Leaders"); R_ = f"Contracts!$AR$2:$AR${n+1}"
wl["A1"] = "Best and worst contracts under the current scenario (total surplus $M, 2023-26 signings). Excludes flagged contracts; defensemen excluded unless Inputs!B10 = 1."; wl["A1"].font = Font(bold=True, size=13)
for blk, (title, fn, r0) in enumerate((("BEST", "LARGE", 3), ("WORST", "SMALL", 21))):
    wl.cell(r0, 1, title).font = Font(bold=True); hdr(wl, r0 + 1, ["Rank", "Player", "Team", "Pos", "Signed", "Term", "Cap hit $M", "Total surplus $M", "Remaining $M", "Data check"])
    for k in range(1, 16):
        r = r0 + 1 + k; wl.cell(r, 1, k); wl.cell(r, 8, f"={fn}({R_},A{r})")
        m_ = f"MATCH(H{r},{R_},0)"
        for col, src in ((2, "A"), (3, "B"), (4, "C"), (5, "D"), (6, "F"), (7, "H"), (9, "AL"), (10, "AP")):
            wl.cell(r, col, f"=INDEX(Contracts!${src}$2:${src}${n+1},{m_})" + ('&""' if col == 10 else ""))
        for c in (7, 8, 9): wl.cell(r, c).number_format = "0.0"
for c, w in zip("ABCDEFGHIJ", (6, 24, 24, 6, 8, 6, 12, 16, 14, 46)): wl.column_dimensions[c].width = w
# Summary by signing year on Inputs
wi["E2"] = "League summary (live)"; wi["E2"].font = Font(bold=True)
for j, h_ in enumerate(["Signed league yr", "Contracts", "Mean surplus, zero $M", "Mean surplus, scenario $M", "Total scenario $M"]): 
    c = wi.cell(3, 5 + j, h_); c.fill = H; c.font = HF
for k, y in enumerate((2023, 2024, 2025, 2026), 4):
    wi.cell(k, 5, y); wi.cell(k, 6, f"=COUNTIF(Contracts!$D$2:$D${n+1},E{k})")
    wi.cell(k, 7, f"=AVERAGEIF(Contracts!$D$2:$D${n+1},E{k},Contracts!$AJ$2:$AJ${n+1})")
    wi.cell(k, 8, f"=AVERAGEIF(Contracts!$D$2:$D${n+1},E{k},Contracts!$AK$2:$AK${n+1})")
    wi.cell(k, 9, f"=SUMIF(Contracts!$D$2:$D${n+1},E{k},Contracts!$AK$2:$AK${n+1})")
    for c in (7, 8, 9): wi.cell(k, c).number_format = "0.0"
for c in "EFGHI": wi.column_dimensions[c].width = 22
# FMV sheets
for cls, season in ((2027, 2027), (2028, 2028)):
    d = F[F.cls == cls].sort_values("share_long_base", ascending=False).reset_index(drop=True)
    wf = wb.create_sheet(f"FMV_{cls}")
    wf["A1"] = f"Projected market price for contracts expiring after {cls-1}-{str(cls)[2:]} (new deal starts {cls}-{str(cls+1)[2:]})" + ("  [PRELIMINARY]" if cls == 2028 else "")
    wf["A1"].font = Font(bold=True, size=13); wf["A2"] = "Cap for new deal $M:"; wf["B2"] = f"=INDEX({CAPR},{season}-2022)"; wf["B2"].number_format = "0.0"
    cols = ["Player", "Team", "Pos", "Status", "Age at new deal", "Current cap hit $M", "Forecast f (GAR pts)", "Expected term (yrs)", "Base share %", "Low share %", "High share %", "BASE $M (expected term)", "Low $M", "High $M", "80% band low $M", "80% band high $M", "Short 2-yr base $M", "Flag", "SD pts"]
    hdr(wf, 4, cols); wf.row_dimensions[4].height = 45
    for i, r in enumerate(d.itertuples(), 5):
        vals = [r.player, r.team, r.pos, r.expiry_status, r.age_at_signing, r.cur_cap_hit / 1e6, r.forecast_f, r.term_exp, r.share_long_base, r.share_long_low, r.share_long_high]
        for k, v in enumerate(vals, 1): wf.cell(i, k, v)
        cap = "$B$2"
        wf.cell(i, 12, f"=MIN(20,I{i})/100*{cap}"); wf.cell(i, 13, f"=MIN(20,J{i})/100*{cap}"); wf.cell(i, 14, f"=MIN(20,K{i})/100*{cap}")
        wf.cell(i, 15, f"=MAX(Inputs!$B$8,MIN(20,I{i})-Inputs!$B$9*S{i})/100*{cap}"); wf.cell(i, 16, f"=MIN(20,MIN(20,I{i})+Inputs!$B$9*S{i})/100*{cap}")
        wf.cell(i, 17, f"=MIN(20,{r.share_short_base})/100*{cap}")
        fl = r.flag if isinstance(r.flag, str) else ""
        if r.expiry_status == "RFA": fl = (fl + "; " if fl else "") + "RFA: arbitration/offer-sheet dynamics, not UFA price"
        if r.cur_cap_hit < 1.0e6: fl = (fl + "; " if fl else "") + "check: current deal under $1M"
        wf.cell(i, 18, fl); wf.cell(i, 19, r.sd_pts)
        for c in (6, 7) + tuple(range(9, 18)): wf.cell(i, c).number_format = "0.0"
    wf.freeze_panes = "B5"; wf.auto_filter.ref = f"A4:S{len(d)+4}"; wf.column_dimensions["A"].width = 24; wf.column_dimensions["B"].width = 22; wf.column_dimensions["R"].width = 44
wb.save("output/NHL_Contract_Surplus_FMV.xlsx")
print("saved", n)
