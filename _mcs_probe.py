import time, numpy as np, pandas as pd
from arch.bootstrap import MCS

data = pd.read_parquet("merged_RV_GVZ_with_macro_event.parquet")
rv = data["RV_gold"].astype(float)
TRADING_DAYS = 252
WINDOW_YEARS = np.arange(1.0, 7.001, 0.25)
WINDOWS = [int(round(yr * TRADING_DAYS)) for yr in WINDOW_YEARS]
DELTAS = np.arange(1.0, 0.987, -0.001)
EPS = 1e-6
x = np.log(rv)

def build_log_design(extra_cols):
    df = pd.DataFrame(index=rv.index)
    df["x_d"] = x; df["x_w"] = x.rolling(5).mean(); df["x_m"] = x.rolling(22).mean()
    for name, s in extra_cols.items(): df[name] = s.reindex(rv.index)
    df["y_log"] = x.shift(-1); df["y_level"] = rv.shift(-1)
    return df.dropna()

log_gvz = np.log(data["GVZ_close"]); log_spx = np.log(data["RV_ES"]); log_crude = np.log(data["RV_crude"])
macro = data["macro_event"].shift(-1).astype(float)
d_gvz = build_log_design({"log_GVZ": log_gvz, "macro": macro})
d_crude = build_log_design({"log_GVZ": log_gvz, "log_RV_crude": log_crude, "macro": macro})
d_spx = build_log_design({"log_GVZ": log_gvz, "log_RV_ES": log_spx, "macro": macro})
START_DATE = d_gvz.index[max(WINDOWS)]
specs = [("R18", d_gvz, ["x_d","x_w","x_m","log_GVZ","macro"]),
         ("R20", d_crude, ["x_d","x_w","x_m","log_GVZ","log_RV_crude","macro"]),
         ("R19", d_spx, ["x_d","x_w","x_m","log_GVZ","log_RV_ES","macro"])]

def _rw(n, delta):
    if delta >= 1.0: return np.ones(n)
    ages = np.arange(n)[::-1]; w = delta**ages; return w*(n/w.sum())
def _qlike(a, f, eps=EPS):
    f = np.maximum(f, eps); r = a/f; return r - np.log(r) - 1.0
def loss_series(design, feats, window, delta):
    X = np.column_stack([np.ones(len(design)), design[feats].to_numpy()])
    yl = design["y_log"].to_numpy(); lvl = design["y_level"].to_numpy(); idx = design.index
    N = len(X); t_all = np.arange(window, N); t_oos = t_all[idx[window:] >= START_DATE]; starts = t_oos - window
    Xwins = np.lib.stride_tricks.sliding_window_view(X, window, axis=0)[starts].transpose(0,2,1)
    ywins = np.lib.stride_tricks.sliding_window_view(yl, window)[starts]
    w = _rw(window, delta); sw = np.sqrt(w)
    Xs = Xwins*sw[None,:,None]; ys = ywins*sw[None,:]
    A = np.einsum("nwi,nwj->nij", Xs, Xs); b = np.einsum("nwi,nw->ni", Xs, ys); beta = np.linalg.solve(A,b)
    fitted = np.einsum("nwp,np->nw", Xwins, beta); smear = np.einsum("nw,w->n", np.exp(ywins-fitted), w)/w.sum()
    fc = np.exp(np.einsum("np,np->n", X[t_oos], beta))*smear
    return pd.Series(_qlike(lvl[t_oos], fc), index=idx[t_oos])

t0 = time.time()
cols = {}
for lbl, d, f in specs:
    for yr, wn in zip(WINDOW_YEARS, WINDOWS):
        for de in DELTAS:
            cols[f"{lbl}|{yr:.2f}|{de:.3f}"] = loss_series(d, f, wn, de)
losses = pd.DataFrame(cols).dropna()
losses.to_parquet("_losses_cache.parquet")
print(f"loss matrix build: {time.time()-t0:.1f}s  shape={losses.shape} -> cached")

t0 = time.time()
m = MCS(losses, size=0.05, reps=10000, block_size=None, method="max", bootstrap="stationary", seed=42)
m.compute()
inc = (m.pvalues["Pvalue"] > 0.05).sum()
print(f"method=max reps=10000: {time.time()-t0:.1f}s  included={inc}/{losses.shape[1]}")
