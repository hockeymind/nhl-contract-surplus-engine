"""Linemate-adjusted even-strength effects from MoneyPuck line and pairing game logs (5on5 only).

Each row is one forward line (3 players) or defense pairing (2 players) in one game, with its ice time and
expected goals for and against. lineId is the player IDs joined (7 digits each).
Model per unit type (forwards, defense), separately for offense (xGF/60) and defense (xGA/60):
    rate = team effect + opponent-team effect + sum of the effects of the players on the unit
fitted by weighted ridge regression (weights = ice time); only the player effects are penalised.
Player effects are therefore measured against teammates, not just against the team average.
"""
import numpy as np, pandas as pd
from scipy import sparse

def load_lines(path):
    d = pd.read_csv(path, dtype={"lineId": str})
    d = d[(d.situation == "5on5") & (d.icetime > 0)].copy()
    d["n"] = d.lineId.str.len() // 7
    ok = (d.position.eq("line") & d.n.eq(3)) | (d.position.eq("pairing") & d.n.eq(2))
    d = d[ok].copy()
    d["players"] = d.lineId.map(lambda s: [int(s[i:i + 7]) for i in range(0, len(s), 7)])
    return d

def design(d, pid_index, team_index):
    rows, cols = [], []
    for r, pl in enumerate(d.players.values):
        for p in pl:
            rows.append(r); cols.append(pid_index[p])
    P = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(d), len(pid_index)))
    nt = len(team_index)
    T = sparse.csr_matrix((np.ones(len(d)), (np.arange(len(d)), d.playerTeam.map(team_index).values)), shape=(len(d), nt))
    O = sparse.csr_matrix((np.ones(len(d)), (np.arange(len(d)), d.opposingTeam.map(team_index).values)), shape=(len(d), nt))
    return sparse.hstack([P, T, O]).tocsr()

def ridge(X, y, w, n_pen, lam):
    W = sparse.diags(w)
    A = (X.T @ W @ X).toarray()
    A[np.arange(n_pen), np.arange(n_pen)] += lam
    A[np.arange(n_pen, A.shape[0]), np.arange(n_pen, A.shape[0])] += 1e-6   # tiny ridge for the collinear team columns
    b = X.T @ (w * y)
    return np.linalg.solve(A, b)

def fit_unit(d, target, lam):
    """Returns {playerId: effect per 60} for one unit type and one target ('xGoalsFor' or 'xGoalsAgainst')."""
    pids = sorted({p for pl in d.players for p in pl}); pid_index = {p: i for i, p in enumerate(pids)}
    teams = sorted(set(d.playerTeam) | set(d.opposingTeam)); team_index = {t: i for i, t in enumerate(teams)}
    X = design(d, pid_index, team_index)
    hours = d.icetime.values / 3600.0
    y = d[target].values / hours
    coef = ridge(X, y, hours, len(pids), lam)
    return pd.Series(coef[:len(pids)], index=pids)

def predict_rmse(d_train, d_test, target, lam):
    pids = sorted({p for pl in d_train.players for p in pl}); pid_index = {p: i for i, p in enumerate(pids)}
    teams = sorted(set(d_train.playerTeam) | set(d_train.opposingTeam)); team_index = {t: i for i, t in enumerate(teams)}
    X = design(d_train, pid_index, team_index); hours = d_train.icetime.values / 3600.0
    coef = ridge(X, d_train[target].values / hours, hours, len(pids), lam)
    dt = d_test[d_test.players.map(lambda pl: all(p in pid_index for p in pl))]
    Xt = design(dt, pid_index, team_index); ht = dt.icetime.values / 3600.0
    return float(np.sqrt(np.average((dt[target].values / ht - Xt @ coef) ** 2, weights=ht))), len(dt)
