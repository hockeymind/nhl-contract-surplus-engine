"""Parse saved PuckPedia 'NHL Signings' pages (File > Save Page As, complete HTML) into one signings table.
Usage: python parse_puckpedia.py [folder with .html files] [output.csv]
With no arguments it reads the .html files in the same folder as this script and writes signings_parsed.csv there.
Reads the visible table of each saved page only; de-duplicates on PuckPedia's signing id, so overlapping pages are safe."""
import sys, re, glob, os
import pandas as pd
from bs4 import BeautifulSoup

# League salary cap by season start year (upper limit), used to express each cap hit as a share of the cap
CAP = {2013: 64.3e6, 2014: 69.0e6, 2015: 71.4e6, 2016: 73.0e6, 2017: 75.0e6, 2018: 79.5e6, 2019: 81.5e6, 2020: 81.5e6, 2021: 81.5e6,
       2022: 82.5e6, 2023: 83.5e6, 2024: 88.0e6, 2025: 95.5e6, 2026: 104.0e6}
need = ["Player", "Sign Team", "Cap Hit", "Term", "Total", "Date", "Start Yr", "Level", "Cap %", "POS", "Sign Age", "Sign Status", "Sign Expiry", "Structure", "Draft Year", "Draft Round"]
txt = lambda c: re.sub(r"\s+", " ", c.get_text(" ", strip=True))

def parse_file(path):
    soup = BeautifulSoup(open(path, encoding="utf-8", errors="ignore").read(), "html.parser")
    out = []
    for t in soup.find_all("table"):
        trs = t.find_all("tr")
        head = [txt(c) for c in trs[0].find_all("th")] if trs else []
        if not all(h in head for h in need): continue
        for tr in trs[1:]:
            tds = tr.find_all("td")
            if len(tds) != len(head): continue
            r = dict(zip(head, [txt(c) for c in tds]))
            a = tds[head.index("Player")].find("a", href=re.compile(r"/signing/\d+"))
            r["signing_id"] = int(re.search(r"/signing/(\d+)", a["href"]).group(1)) if a else None
            ta = tds[head.index("Sign Team")].find("a", href=re.compile(r"/team/"))
            r["team"] = ta["href"].rstrip("/").split("/")[-1] if ta else None
            out.append(r)
    m = re.search(r"Showing (\d+) - (\d+)", soup.get_text(" ", strip=True))
    return out, m.groups() if m else None

HERE = os.path.dirname(os.path.abspath(__file__))

def main(folder=HERE, out_csv=None):
    out_csv = out_csv or os.path.join(folder, "signings_parsed.csv")
    recs = []
    for f in sorted(glob.glob(os.path.join(folder, "*.htm*"))):
        rows, info = parse_file(f)
        print(os.path.basename(f), "rows:", len(rows), "| page says showing", info)
        if info and int(info[1]) - int(info[0]) + 1 != len(rows): print("  WARNING: parsed rows differ from the page's own count")
        recs += rows
    if not recs: sys.exit("No signings tables found in " + folder + " (looked for .htm/.html files with the PuckPedia signings table).")
    d = pd.DataFrame(recs)
    n0 = len(d); d = d.drop_duplicates("signing_id"); print("duplicates removed:", n0 - len(d))
    num = lambda s: float(re.sub(r"[^0-9.]", "", s)) if re.search(r"\d", str(s)) else None
    o = pd.DataFrame({
        "signing_id": d.signing_id, "player_name": d.Player, "team": d.team,
        "position": d.POS, "age_at_signing": pd.to_numeric(d["Sign Age"], errors="coerce"),
        "sign_date": pd.to_datetime(d.Date.str.extract(r"^([A-Za-z]{3} \d{1,2}, \d{4})")[0], format="%b %d, %Y", errors="coerce").dt.date,
        "first_season": d["Start Yr"].str[:4].astype(int), "term_years": d.Term.str.extract(r"(\d+)")[0].astype(int),
        "cap_hit_usd": d["Cap Hit"].map(num), "level": d.Level, "sign_status": d["Sign Status"], "sign_expiry": d["Sign Expiry"],
        "structure": d.Structure, "draft_year": pd.to_numeric(d["Draft Year"], errors="coerce"), "draft_round": pd.to_numeric(d["Draft Round"], errors="coerce"),
        "cap_pct_listed": d["Cap %"].str.rstrip("%").astype(float) / 100})
    o["total_usd"] = o.cap_hit_usd * o.term_years
    o["cap_ceiling_usd"] = o.first_season.map(CAP)
    o["cap_share"] = o.cap_hit_usd / o.cap_ceiling_usd
    bad = (o.cap_share - o.cap_pct_listed).abs() > 0.006
    if bad.any(): print("WARNING: computed cap share differs from the page for", bad.sum(), "rows (check the cap table)")
    o["is_goalie"] = o.position.eq("G"); o["buys_ufa_years"] = o.sign_status.str.startswith("RFA") & o.sign_expiry.str.startswith("UFA") & ~o.sign_expiry.str.contains("No QO")
    o["league_year"] = (pd.to_datetime(o.sign_date) - pd.Timedelta(days=181)).dt.year   # Jul 1 - Jun 30, labelled by its first calendar year
    rep = o[~o.is_goalie].groupby("league_year").agg(signings=("signing_id", "size"), at_1m_plus=("cap_hit_usd", lambda s: int((s >= 1e6).sum())),
                                                    biggest_cap_hit=("cap_hit_usd", "max"))
    print("\nCoverage by league year (Jul 1 - Jun 30). The biggest cap hit should be that year's largest skater contract:"); print(rep.to_string())
    o.to_csv(out_csv, index=False); print(len(o), "signings ->", out_csv); return o

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE, sys.argv[2] if len(sys.argv) > 2 else None)
