"""Fetch birth dates for every player in the MoneyPuck skater files from the NHL's public web API.

Run this ONCE on your own computer (the cloud sandbox cannot reach api-web.nhle.com):

    pip install requests pandas
    python src/fetch_birthdates.py

Paths are relative to the project folder, not the terminal's folder. The script prints
the full path it reads from and writes to when it starts.

It writes data/birthdates.csv (playerId, birthDate, name) and resumes if interrupted.
The NHL web API is unofficial and undocumented; be polite (the script waits between calls)
and check the NHL's terms before redistributing the data. Upload data/birthdates.csv back to Claude.
"""
import csv, os, sys, time
import pandas as pd
import requests

# Paths are anchored to the project folder (the folder that contains src/), not to the
# terminal's current directory, so the output lands in the same place wherever you run it.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.environ.get("NHL_RAW_DIR", os.path.join(ROOT, "data", "raw"))   # folder with your 'skaters YYYY.csv' files
OUT = os.environ.get("NHL_OUT_FILE", os.path.join(ROOT, "data", "birthdates.csv"))
URL = "https://api-web.nhle.com/v1/player/{pid}/landing"
PAUSE = 0.25          # seconds between requests
RETRIES = 4

def player_ids():
    ids = {}
    for f in sorted(os.listdir(RAW)):
        if f.startswith("skaters ") and f.endswith(".csv"):
            d = pd.read_csv(os.path.join(RAW, f), usecols=["playerId", "name", "situation"])
            for pid, name in d[d.situation == "all"][["playerId", "name"]].itertuples(index=False):
                ids[int(pid)] = name
    return ids

def main():
    if not os.path.isdir(RAW):
        sys.exit(f"Skater files folder not found: {RAW}\nPut the 'skaters YYYY.csv' files there, or set NHL_RAW_DIR.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print("Reading skater files from:", RAW)
    print("Writing birth dates to:   ", OUT)
    ids = player_ids()
    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT).playerId.astype(int))
    todo = [p for p in ids if p not in done]
    print(f"{len(ids)} players, {len(done)} already fetched, {len(todo)} to go")
    new_file = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as fh:
        wr = csv.writer(fh)
        if new_file:
            wr.writerow(["playerId", "birthDate", "name"])
        for i, pid in enumerate(todo, 1):
            birth = ""
            for attempt in range(RETRIES):
                try:
                    r = requests.get(URL.format(pid=pid), timeout=20)
                    if r.status_code == 200:
                        birth = r.json().get("birthDate", "") or ""
                        break
                    if r.status_code == 404:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2 ** attempt)
            wr.writerow([pid, birth, ids[pid]])
            fh.flush()
            if i % 100 == 0:
                print(f"{i}/{len(todo)}")
            time.sleep(PAUSE)
    print("done ->", OUT)

if __name__ == "__main__":
    sys.exit(main())
