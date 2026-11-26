"""Load every MoneyPuck skater season file into one table."""
import glob, os
import pandas as pd

RENAME = {"penalities": "penalties", "penalitiesDrawn": "penaltiesDrawn"}  # 2007 file misspells these

def load_skaters(raw_dir):
    frames = []
    for f in sorted(glob.glob(os.path.join(raw_dir, "skaters *.csv"))):
        d = pd.read_csv(f)
        d = d.drop(columns=[c for c in ("I_F_penalities",) if c in d.columns]).rename(columns=RENAME)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["pos_group"] = d["position"].map(lambda p: "D" if p == "D" else "F")
    return d
