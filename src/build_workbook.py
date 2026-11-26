"""Build the Excel workbook from the model output. Components are model outputs; totals, intervals, WAR, leaders and lookup are live formulas."""
import sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter as L
import gar, aging
from load_data import load_skaters

d = load_skaters("data/raw")
w, vcs, gm = gar.run(d)
wf = gar.walk_forward(w)
birth = pd.read_csv("data/birthdates.csv")
w = aging.add_age(w, birth)
am = aging.pairs(w)
adt = aging.delta_table(am).merge(aging.bootstrap_ci(am, reps=200).reset_index(), on=["pos_group", "age"])
awf = aging.walk_forward_aging(w, am)
avail = aging.delta_table(am, "dgp").rename(columns={"delta": "dgp"})
adt = adt.merge(avail[["pos_group", "age", "dgp"]], on=["pos_group", "age"], how="left")
# replacement-level sensitivity (2024 season)
base = w[["playerId", "season", "GAR"]].rename(columns={"GAR": "base"})
sens = []
for cut in [{"F": 11, "D": 5}, {"F": 13, "D": 7}]:
    w2, _, _ = gar.run(d, cut); m = base.merge(w2[["playerId", "season", "GAR"]], on=["playerId", "season"])
    x = m[m.season == 2024]; sens.append((f"Forwards ranked {cut['F']+1}+ / defensemen ranked {cut['D']+1}+", x.base.corr(x.GAR), (x.GAR - x.base).mean()))
gar.REPL_CUT.update({"F": 12, "D": 6})
comps = list(gar.COMPONENTS)
w = w.sort_values(["season", "GAR"], ascending=[False, False]).reset_index(drop=True)
w["label"] = w.season.astype(str) + "-" + ((w.season + 1) % 100).astype(str).str.zfill(2)
w.to_csv("output/player_seasons_gar.csv", index=False)

F = "Arial"
f_norm = Font(name=F, size=10); f_b = Font(name=F, size=10, bold=True); f_h = Font(name=F, size=10, bold=True, color="FFFFFF")
f_in = Font(name=F, size=10, color="0000FF"); f_title = Font(name=F, size=14, bold=True); f_note = Font(name=F, size=9, italic=True, color="595959")
fill_h = PatternFill("solid", fgColor="1F3864"); fill_in = PatternFill("solid", fgColor="FFFF00"); fill_sub = PatternFill("solid", fgColor="D9E1F2")
wrap = Alignment(wrap_text=True, vertical="top")

wb = Workbook()
# ---------- README ----------
ws = wb.active; ws.title = "README"
ws.column_dimensions["A"].width = 26; ws.column_dimensions["B"].width = 110
rows = [
 ("NHL Contract Surplus Engine: Workstream A baseline", None),
 ("What this is", "Goals above replacement (GAR) for every NHL skater-season from 2007-08 to 2025-26, built from MoneyPuck season files, with an 80% interval for each player. This is the player-value layer that the contract grader will use. It contains no contract data yet."),
 ("How to use", "Leaders: pick a season and minimum games to see the top 25. Player_Lookup: type a name and season to see the component breakdown and a next-season projection. Aging_Curves: yearly change in GAR by age. Player_Data: every player-season. Assumptions: the interval level and goals-per-win inputs (blue on yellow) drive the totals."),
 ("How GAR is built", "Six components, each in goals relative to a replacement-level pool at the same position group: even-strength offense, even-strength defense, power-play offense, penalty-kill defense, net penalties (drawn minus taken, valued at the league goal swing per minor), and finishing (goals minus expected goals). On-ice rates are measured relative to the player's own team. Each rate is shrunk toward the position mean with an empirical-Bayes estimate, so small samples pull toward average and their intervals widen."),
 ("Replacement level", "Forwards ranked 13th or lower and defensemen ranked 7th or lower in ice time on their team, season by season. This is an assumption. Moving the cut by one rank changes the 2024-25 ranking very little (see Validation) but shifts everyone's GAR up or down by about one goal."),
 ("Not included", "Faceoffs, hits, blocked shots, giveaways and takeaways. Tests on 2018-2025 data showed hits, blocks and giveaways add almost nothing to next-season prediction once on-ice xG is known; takeaways add a little and need an arena adjustment before use. Goalies are excluded. Age enters only through the Aging_Curves sheet and the next-season projection on Player_Lookup; GAR itself is not age-adjusted."),
 ("Main limits", "1) Relative-to-team on-ice rates still credit a player for strong linemates; players on stacked lines (for example, line mates of a star) can look better than they are. The game-level and shift-level data will fix this. 2) Public data has no pass, entry or tracking data. 3) Intervals describe uncertainty in the player's true talent from this season's sample, assuming components are independent; they are not yet calibrated against outcomes. 4) Components are model outputs from the Python code in src/gar.py; only totals, intervals, WAR, leaders and lookup are live formulas."),
 ("Data credit", "Player and team data: MoneyPuck.com (free for non-commercial use with credit). Season labels in the data are the start year: 2024 means 2024-25."),
]
for i, (a, b) in enumerate(rows, 1):
    ws.cell(i, 1, a).font = f_title if i == 1 else f_b
    if b: c = ws.cell(i, 2, b); c.font = f_norm; c.alignment = wrap; ws.cell(i, 1).alignment = wrap
    if i > 1: ws.row_dimensions[i].height = max(30, 15 * (len(b) // 95 + 1))

# ---------- Assumptions ----------
wa = wb.create_sheet("Assumptions")
wa.column_dimensions["A"].width = 44; wa.column_dimensions["B"].width = 14; wa.column_dimensions["C"].width = 90
wa["A1"] = "Assumptions and model constants"; wa["A1"].font = f_title
wa["A2"] = "Inputs (edit these)"; wa["A2"].font = f_b
inp = [("z for 80% interval (1.2816)", 1.2816, "Standard normal value for a two-sided 80% interval. Use 1.645 for 90%."),
       ("Goals per win", 6, "Rule of thumb of roughly 6 goals per win; used only to convert GAR to WAR. [Likely]")]
for i, (a, v, n) in enumerate(inp, 3):
    wa.cell(i, 1, a).font = f_norm; c = wa.cell(i, 2, v); c.font = f_in; c.fill = fill_in; wa.cell(i, 3, n).font = f_note
wa["A6"] = "Model constants from the Python run (read-only; changing these does not recalculate the components)"; wa["A6"].font = f_b
wa["A7"] = "Goals per minor penalty"; wa["B7"] = round(gm["goals_per_minor"], 4); wa["C7"] = "League goal swing for one 2-minute penalty from 5on4 and 4on5 expected-goal rates across all 19 seasons."
wa["A7"].font = f_norm; wa["B7"].font = f_norm; wa["C7"].font = f_note
hdr = ["Component", "Position", "Mean rate", "True-talent SD (tau)", "Noise coef. k", "Obs.", "Pairs"]
for j, h in enumerate(hdr, 1):
    c = wa.cell(9, j, h); c.font = f_h; c.fill = fill_h
for j, w_ in enumerate([44, 14, 14, 20, 16, 10, 10], 1): wa.column_dimensions[L(j)].width = max(wa.column_dimensions[L(j)].width or 0, w_)
r = 10
for comp in comps:
    for pg in ["F", "D"]:
        v = vcs[comp][pg]
        for j, val in enumerate([comp, pg, round(v["mu"], 4), round(v["tau2"] ** .5, 4), round(v["k"], 4), v["n_obs"], v["n_pairs"]], 1):
            c = wa.cell(r, j, val); c.font = f_norm
        r += 1
wa.cell(r + 1, 1, "Rates are per hour of the component's own ice time. Shrinkage weight = tau^2 / (tau^2 + k / hours).").font = f_note

# ---------- Player_Data ----------
wd = wb.create_sheet("Player_Data")
heads = ["Season", "Season label", "PlayerId", "Player", "Team", "Pos", "GP", "Hours (all)"] + \
        [f"GAR {c}" for c in comps] + [f"SD {c}" for c in comps] + \
        ["GAR", "GAR SD", "GAR low (80%)", "GAR high (80%)", "GAR per 82", "WAR", "GAR raw (unshrunk)", "Key", "Leader helper", "Age (mid-season)"]
for j, h in enumerate(heads, 1):
    c = wd.cell(1, j, h); c.font = f_h; c.fill = fill_h; c.alignment = Alignment(wrap_text=True, vertical="center")
n = len(w)
last = n + 1
vals = w[["season", "label", "playerId", "name", "team", "position", "games_played", "h_all"] +
         [f"gar_{c}" for c in comps] + [f"garsd_{c}" for c in comps] + ["GAR_raw", "age"]].values.tolist()
for i, row in enumerate(vals, 2):
    for j in range(1, 15): wd.cell(i, j, row[j - 1])
    for j in range(15, 21): wd.cell(i, j, row[j - 1])
    # row[20] is GAR_raw -> column 27
    wd.cell(i, 27, row[20])
    wd.cell(i, 30, row[21])
    wd.cell(i, 21, f"=SUM(I{i}:N{i})")
    wd.cell(i, 22, f"=SQRT(SUMSQ(O{i}:T{i}))")
    wd.cell(i, 23, f"=U{i}-Assumptions!$B$3*V{i}")
    wd.cell(i, 24, f"=U{i}+Assumptions!$B$3*V{i}")
    wd.cell(i, 25, f"=IF(G{i}>0,U{i}/G{i}*82,0)")
    wd.cell(i, 26, f"=U{i}/Assumptions!$B$4")
    wd.cell(i, 28, f'=D{i}&"|"&A{i}')
    wd.cell(i, 29, f'=IF(AND(A{i}=Leaders!$B$2,G{i}>=Leaders!$B$3),U{i}+ROW()/1000000000,"")')
for col in range(1, 31):
    wd.column_dimensions[L(col)].width = 12
wd.column_dimensions["D"].width = 22; wd.column_dimensions["AB"].width = 26
wd.freeze_panes = "E2"; wd.row_dimensions[1].height = 42
for row in wd.iter_rows(min_row=2, max_row=last):
    for c in row:
        c.font = f_norm
        if c.column in range(9, 28) and c.column != 27 or c.column == 27: c.number_format = "0.0"
        if c.column == 8: c.number_format = "0.0"
wd.auto_filter.ref = f"A1:AD{last}"

# ---------- Leaders ----------
wl = wb.create_sheet("Leaders", 1)
wl["A1"] = "Season leaders by GAR"; wl["A1"].font = f_title
wl["A2"] = "Season (start year, 2024 = 2024-25)"; wl["B2"] = 2024
wl["A3"] = "Minimum games played"; wl["B3"] = 40
for a in ("A2", "A3"): wl[a].font = f_norm
for b in ("B2", "B3"): wl[b].font = f_in; wl[b].fill = fill_in
heads = ["Rank", "Player", "Team", "Pos", "GP", "GAR", "80% low", "80% high", "GAR per 82", "WAR", "Even-str. off.", "Even-str. def.", "PP off.", "PK def.", "Penalties", "Finishing"]
for j, h in enumerate(heads, 1):
    c = wl.cell(5, j, h); c.font = f_h; c.fill = fill_h; c.alignment = Alignment(wrap_text=True)
PD = lambda col: f"Player_Data!${col}$2:${col}${last}"
srcs = ["D", "E", "F", "G", "U", "W", "X", "Y", "Z", "I", "J", "K", "L", "M", "N"]
for k in range(1, 26):
    r = 5 + k
    wl.cell(r, 1, k).font = f_norm
    wl.cell(r, 17, f'=IFERROR(MATCH(LARGE({PD("AC")},A{r}),{PD("AC")},0),"")').font = f_note
    for j, col in enumerate(srcs, 2):
        c = wl.cell(r, j, f'=IF($Q{r}="","",INDEX({PD(col)},$Q{r}))'); c.font = f_norm
        if col not in ("D", "E", "F", "G"): c.number_format = "0.0"
wl["Q5"] = "row"; wl["Q5"].font = f_note
wl.column_dimensions["A"].width = 34; wl.column_dimensions["B"].width = 22
for col in range(3, 17): wl.column_dimensions[L(col)].width = 11
wl.column_dimensions["Q"].width = 6
wl["A32"] = "Intervals are for the player's true talent given this season's sample (components assumed independent). Not calibrated against outcomes yet."; wl["A32"].font = f_note

# ---------- Player_Lookup ----------
wp = wb.create_sheet("Player_Lookup", 2)
wp["A1"] = "Player lookup"; wp["A1"].font = f_title
wp["A2"] = "Player name (as in the data)"; wp["B2"] = "Leon Draisaitl"
wp["A3"] = "Season (start year)"; wp["B3"] = 2024
wp["C2"] = "Names with two players of the same name return the first match."; wp["C2"].font = f_note
for a in ("A2", "A3"): wp[a].font = f_norm
for b in ("B2", "B3"): wp[b].font = f_in; wp[b].fill = fill_in
wp["A4"] = "Row"; wp["B4"] = f'=IFERROR(MATCH(B2&"|"&B3,{PD("AB")},0),"not found")'
wp["A4"].font = f_note; wp["B4"].font = f_note
lab = [("Team", "E", None), ("Position", "F", None), ("Games played", "G", "0"), ("Hours of ice time (all strengths)", "H", "0.0"),
       ("Even-strength offense (goals)", "I", "0.0"), ("Even-strength defense (goals)", "J", "0.0"), ("Power-play offense (goals)", "K", "0.0"),
       ("Penalty-kill defense (goals)", "L", "0.0"), ("Net penalties (goals)", "M", "0.0"), ("Finishing (goals)", "N", "0.0"),
       ("GAR", "U", "0.0"), ("GAR low (80%)", "W", "0.0"), ("GAR high (80%)", "X", "0.0"), ("GAR per 82 games", "Y", "0.0"), ("WAR", "Z", "0.0"), ("GAR before shrinkage", "AA", "0.0"), ("Age (mid-season)", "AD", "0")]
for i, (a, col, nf) in enumerate(lab, 6):
    wp.cell(i, 1, a).font = f_norm
    c = wp.cell(i, 2, f'=IF(ISNUMBER($B$4),INDEX({PD(col)},$B$4),"")'); c.font = f_b if col == "U" else f_norm
    if nf: c.number_format = nf



# ---------- Aging_Curves ----------
wg = wb.create_sheet("Aging_Curves", 3)
wg["A1"] = "Aging and availability curves (delta method)"; wg["A1"].font = f_title
wg["A2"] = ("Delta = average change in unshrunk GAR per 82 games from age a to age a+1, for players with at least 40% of a full season's games in both seasons "
            "(9,080 consecutive-season pairs, 2007-08 to 2025-26). Age is age on Feb 1 of the season's second calendar year. 80% bands are from resampling players. "
            "Level is the cumulative curve anchored at age 25. Survivorship caveat: players who decline and leave the league drop out, so older-age declines are understated. [Likely]")
wg["A2"].font = f_note; wg["A2"].alignment = wrap; wg.merge_cells("A2:H2"); wg.row_dimensions[2].height = 66
gh = ["Position", "Age", "Pairs", "Delta (GAR/82)", "80% low", "80% high", "Games share change", "Level vs age 25"]
for j, h in enumerate(gh, 1):
    c = wg.cell(4, j, h); c.font = f_h; c.fill = fill_h; c.alignment = Alignment(wrap_text=True)
r0 = 5
agrow = {}
for pg in ["F", "D"]:
    g = adt[adt.pos_group == pg].sort_values("age")
    first = r0
    for _, x in g.iterrows():
        agrow[(pg, int(x.age))] = r0
        for j, v in enumerate([pg, int(x.age), int(x.n), round(float(x.delta), 3), round(float(x.lo), 3), round(float(x.hi), 3), round(float(x.dgp), 4)], 1):
            c = wg.cell(r0, j, v); c.font = f_norm
            if j in (4, 5, 6): c.number_format = "0.00"
            if j == 7: c.number_format = "0.000"
        r0 += 1
    last_r = r0 - 1
    # level formulas anchored at age 25 (younger: level_next - delta_this ; older: level_prev + delta_prev)
    for rr in range(first, last_r + 1):
        age = wg.cell(rr, 2).value
        if age == 25: f = 0
        elif age > 25: f = f"=H{rr-1}+D{rr-1}"
        else: f = None
        if f is not None: wg.cell(rr, 8, f)
    for rr in range(last_r, first - 1, -1):
        age = wg.cell(rr, 2).value
        if age < 25: wg.cell(rr, 8, f"=H{rr+1}-D{rr}")
    for rr in range(first, last_r + 1):
        c = wg.cell(rr, 8); c.font = f_norm; c.number_format = "0.0"
    r0 += 1
wg_last = r0
for col, wdt in zip("ABCDEFGH", [10, 8, 8, 16, 11, 11, 18, 16]): wg.column_dimensions[col].width = wdt
wg.freeze_panes = "A5"

wp.cell(23, 1, "Age adjustment for next season (GAR per 82)").font = f_norm
wp.cell(23, 2, '=IF(ISNUMBER($B$4),SUMIFS(Aging_Curves!$D$5:$D$%d,Aging_Curves!$A$5:$A$%d,IF($B$7="D","D","F"),Aging_Curves!$B$5:$B$%d,$B$22),"")' % (wg_last, wg_last, wg_last)).number_format = "0.00"
wp.cell(24, 1, "Projected GAR per 82 next season (shrunk + age adjustment)").font = f_b
wp.cell(24, 2, '=IF(ISNUMBER($B$4),$B$19+$B$23,"")').number_format = "0.0"
wp.cell(24, 2).font = f_b
wp["C23"] = "Zero when the age is outside the range with enough data. Backtest gain from the age adjustment is small (see Validation)."; wp["C23"].font = f_note
wp.column_dimensions["A"].width = 52; wp.column_dimensions["B"].width = 20

# ---------- Validation ----------
wv = wb.create_sheet("Validation")
wv["A1"] = "Validation so far"; wv["A1"].font = f_title
wv["A3"] = "Walk-forward test: predict next season's unshrunk GAR per 82 games. Players with 30+ games in both seasons. Lower error is better."; wv["A3"].font = f_note
for j, h in enumerate(["Predictor", "Error (RMSE, goals per 82)"], 1):
    c = wv.cell(4, j, h); c.font = f_h; c.fill = fill_h
rows = [("All-player average", wf["all-player average"]), ("Last season, unshrunk", wf["last season raw"]), ("This model (shrunk)", wf["shrunk (this model)"]),
        ("Average of last two seasons, unshrunk (smaller subset)", wf["avg of last 2 seasons raw (subset)"]), ("This model on the same subset", wf["shrunk, same subset"])]
for i, (a, v) in enumerate(rows, 5):
    wv.cell(i, 1, a).font = f_norm; c = wv.cell(i, 2, round(v, 3)); c.font = f_norm
wv.cell(11, 1, "Pairs tested (all / subset)").font = f_norm; wv.cell(11, 2, f"{wf['n_pairs']} / {wf['n_pairs_subset']}").font = f_norm
wv["A13"] = "Improvement of this model over last season, unshrunk"; wv["A13"].font = f_b
wv["B13"] = "=1-B7/B6"; wv["B13"].number_format = "0.0%"; wv["B13"].font = f_b
wv["A15"] = "Replacement-level sensitivity (2024-25 season)"; wv["A15"].font = f_b
for j, h in enumerate(["Alternative cut", "Rank correlation with base (Pearson on GAR)", "Average GAR shift (goals)"], 1):
    c = wv.cell(16, j, h); c.font = f_h; c.fill = fill_h; c.alignment = Alignment(wrap_text=True)
for i, (a, cr, sh) in enumerate(sens, 17):
    wv.cell(i, 1, a).font = f_norm; wv.cell(i, 2, round(cr, 4)).font = f_norm; wv.cell(i, 3, round(sh, 2)).font = f_norm
wv["A19"] = "Aging backtest: project next season's GAR per 82 (curve fitted without the target season)"; wv["A19"].font = f_b
for i, (a, v) in enumerate([("Last season, unshrunk", awf["last season raw"]), ("Shrunk", awf["shrunk"]), ("Shrunk + age curve", awf["shrunk + age curve"])], 20):
    wv.cell(i, 1, a).font = f_norm; wv.cell(i, 2, round(v, 3)).font = f_norm
wv["A24"] = "Gain from adding the age curve (share of error)"; wv["A24"].font = f_b; wv["B24"] = "=1-B22/B21"; wv["B24"].number_format = "0.0%"; wv["B24"].font = f_b
wv["A26"] = "Not yet done: calibration of the 80% intervals (needs the aging and availability projection), contract backtest (needs contract data)."; wv["A26"].font = f_note
wv.column_dimensions["A"].width = 58; wv.column_dimensions["B"].width = 30; wv.column_dimensions["C"].width = 26
wb.save("output/NHL_GAR_Baseline.xlsx")
print("saved", n, "rows")
json.dump({"wf": wf, "sens": [(a, float(b), float(c)) for a, b, c in sens], "gm": gm}, open("output/validation.json", "w"))
