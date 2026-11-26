import json
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
r = json.load(open("output/ice_time_results.json"))
F = "Arial"; fn = Font(name=F, size=10); fb = Font(name=F, size=10, bold=True); fh = Font(name=F, size=10, bold=True, color="FFFFFF")
ft = Font(name=F, size=14, bold=True); fnote = Font(name=F, size=9, italic=True, color="595959"); fin = Font(name=F, size=10, color="0000FF")
hf = PatternFill("solid", fgColor="1F3864"); yf = PatternFill("solid", fgColor="FFFF00")
wb = Workbook(); ws = wb.active; ws.title = "Results"
ws["A1"] = "Does per-minute even-strength impact fall as ice time rises?"; ws["A1"].font = ft
ws["A2"] = "Slope = change in 5-on-5 on-ice xG differential per 60 for each extra minute of 5-on-5 ice time in a game. Negative means diminishing returns."; ws["A2"].font = fnote
ws["A3"] = "z for confidence interval"; ws["B3"] = 1.96; ws["A3"].font = fn; ws["B3"].font = fin; ws["B3"].fill = yf
hdr = ["Group", "Method", "Slope per extra minute", "Std. error", "CI low", "CI high", "t-stat", "Game rows", "Player-seasons", "Reading"]
for j, h in enumerate(hdr, 1):
    c = ws.cell(5, j, h); c.font = fh; c.fill = hf; c.alignment = Alignment(wrap_text=True)
rows = [("F", "OLS-FE", "Within-player comparison only. Contaminated by selection: coaches play players more when they are playing well."),
        ("F", "placebo", "Next game's minutes cannot affect this game. A similar slope here shows the OLS slope is selection, not an effect."),
        ("F", "IV", "Main estimate: minutes moved by team 5-on-5 time (penalties), not by form."),
        ("D", "OLS-FE", "As above, defensemen."), ("D", "placebo", "As above, defensemen."), ("D", "IV", "Main estimate, defensemen.")]
names = {"F": "Forwards", "D": "Defensemen"}; meth = {"OLS-FE": "OLS with player fixed effects", "placebo": "Placebo: next-game minutes", "IV": "IV: team 5-on-5 minutes"}
for i, (g, m, note) in enumerate(rows, 6):
    b, s, n, p = r[f"{g}|{m}"]
    for j, v in enumerate([names[g], meth[m], round(b, 4), round(s, 4)], 1): ws.cell(i, j, v).font = fn
    ws.cell(i, 5, f"=C{i}-$B$3*D{i}"); ws.cell(i, 6, f"=C{i}+$B$3*D{i}"); ws.cell(i, 7, f"=C{i}/D{i}")
    ws.cell(i, 8, n); ws.cell(i, 9, p); ws.cell(i, 10, note)
    for j in (5, 6): ws.cell(i, j).number_format = "0.0000"
    ws.cell(i, 7).number_format = "0.0"; ws.cell(i, 8).number_format = "#,##0"
    for j in range(5, 11): ws.cell(i, j).font = fn
    ws.cell(i, 10).alignment = Alignment(wrap_text=True, vertical="top")
ws["A13"] = "Quadratic term (IV), forwards / defensemen"; ws["A13"].font = fb
ws["C13"] = round(r["F|quad"][0], 4); ws["D13"] = round(r["F|quad"][1], 4); ws["C14"] = round(r["D|quad"][0], 4); ws["D14"] = round(r["D|quad"][1], 4)
ws["A14"] = "(second row: defensemen)"; ws["A14"].font = fnote
for c in ("C13", "D13", "C14", "D14"): ws[c].font = fn
ws["A16"] = "Worst case for forwards: effect of five extra minutes (low end of CI times 5), xGD per 60"; ws["A16"].font = fb; ws["C16"] = "=E8*5"; ws["C16"].number_format = "0.000"; ws["C16"].font = fb
ws["A18"] = "Method"; ws["A18"].font = fb
notes = ["Data: MoneyPuck game-level skater files, 2018-19 to 2025-26 regular seasons, 5-on-5 rows, players with 3+ minutes in the game and 30+ games in the season.",
 "Outcome: score-and-venue-adjusted on-ice xG for minus against, per 60, per game. Weighted by minutes. Standard errors clustered by player.",
 "Controls: home or away and opponent team. Player-season fixed effects remove differences between players.",
 "Instrument: the team's total 5-on-5 minutes in that game. It moves with penalties, so it changes minutes without reflecting a player's form. First-stage within-player correlation for forwards is about 0.5.",
 "Limit: the instrument shifts forward minutes by roughly 1 to 2 minutes, so this speaks to small changes, not to moving a player from 12 to 20 minutes.",
 "Not tested: fatigue within a shift or across consecutive shifts (needs shift-level NHL data)."]
for i, t in enumerate(notes, 19):
    ws.cell(i, 1, t).font = fn; ws.merge_cells(start_row=i, start_column=1, end_row=i, end_column=10); ws.cell(i, 1).alignment = Alignment(wrap_text=True, vertical="top"); ws.row_dimensions[i].height = 30
for col, w in zip("ABCDEFGHIJ", [30, 30, 18, 12, 12, 12, 10, 12, 14, 70]): ws.column_dimensions[col].width = w
ws.row_dimensions[5].height = 30
wb.save("output/Ice_Time_Returns.xlsx")
