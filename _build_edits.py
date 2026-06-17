import json

WIN_OLD = 'WINDOW_YEARS = np.array([1.0, 1.5, 2.0, 2.5, 3.0])            # reduced grid for the 3D sweep'
WIN_NEW = 'WINDOW_YEARS = np.arange(1.0, 7.001, 0.25)                   # 1.0y .. 7.0y in 0.25y steps'
DELTAS_OLD = 'DELTAS = [1.0, 0.999, 0.995, 0.99, 0.97]'
DELTAS_NEW = 'DELTAS = np.arange(1.0, 0.987, -0.001)                        # 1.000 .. 0.987 in -0.001 steps'
IDX_OLD = 'ax.scatter([yr_star], [DELTAS.index(d_star)], [q_star], color="red", s=60,'
IDX_NEW = 'ax.scatter([yr_star], [int(df.columns.get_loc(d_star))], [q_star], color="red", s=60,'


def edit_cell(cell, old, new, required=True):
    s = "".join(cell["source"])
    if old not in s:
        if required:
            raise SystemExit(f"NOT FOUND: {old[:60]!r}")
        return
    s = s.replace(old, new)
    cell["source"] = s.splitlines(keepends=True)


def clear(cell):
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def apply_grid(path, is_macro):
    nb = json.load(open(path))
    cells = nb["cells"]
    for c in cells:
        edit_cell(c, WIN_OLD, WIN_NEW, required=False)
        edit_cell(c, DELTAS_OLD, DELTAS_NEW, required=False)
        edit_cell(c, IDX_OLD, IDX_NEW, required=False)
        clear(c)
    # sanity: each replacement should now be present somewhere
    blob = "\n".join("".join(c["source"]) for c in cells)
    assert WIN_NEW in blob and DELTAS_NEW in blob and IDX_NEW in blob, f"grid edits incomplete in {path}"
    if is_macro:
        add_macro_bits(cells)
    json.dump(nb, open(path, "w"), indent=1)
    print("edited", path)


HELPER = '''

# --- Per-day QLIKE loss series (date-indexed) for the MCS test ------------------
# Same vectorised WLS machinery as rolling_log_ols_eval, but returns the per-forecast
# -day QLIKE loss as a date-indexed Series (forecast-origin dates) instead of the mean.
# The MCS test (arch.bootstrap.MCS) needs aligned per-day loss series, one per model.
def rolling_log_ols_loss_series(design, feat_cols, window, start_date=None, delta=1.0):
    if start_date is None:
        start_date = START_DATE
    X = np.column_stack([np.ones(len(design)), design[feat_cols].to_numpy()])
    yl  = design["y_log"].to_numpy()
    lvl = design["y_level"].to_numpy()
    idx = design.index
    N, p = X.shape
    t_all = np.arange(window, N)
    t_oos = t_all[idx[window:] >= start_date]
    starts = t_oos - window
    Xwins = np.lib.stride_tricks.sliding_window_view(X, window, axis=0)[starts].transpose(0, 2, 1)
    ywins = np.lib.stride_tricks.sliding_window_view(yl, window)[starts]
    w  = _recency_weights(window, delta); sw = np.sqrt(w)
    Xs = Xwins * sw[None, :, None]; ys = ywins * sw[None, :]
    A = np.einsum("nwi,nwj->nij", Xs, Xs)
    b = np.einsum("nwi,nw->ni", Xs, ys)
    beta = np.linalg.solve(A, b)
    fitted = np.einsum("nwp,np->nw", Xwins, beta)
    smear = np.einsum("nw,w->n", np.exp(ywins - fitted), w) / w.sum()
    x_pred = np.einsum("np,np->n", X[t_oos], beta)
    fc = np.exp(x_pred) * smear
    ac = lvl[t_oos]
    q, _ = _qlike(ac, fc)
    return pd.Series(q, index=idx[t_oos], name="qlike")

'''

MCS_CELL = '''# ===========================================================================
# Cell 8 — Combined Model Confidence Set (MCS) over all (spec, window, delta) models
# ===========================================================================
# Build one per-day QLIKE loss series per model (3 specs x 25 windows x 14 deltas =
# 1050 models), align them on the common OOS dates, and run the Model Confidence Set
# (Hansen, Lunde & Nason 2011) via arch.bootstrap.MCS. The MCS p-value of a model is
# the probability it belongs to the set of models statistically indistinguishable from
# the best: p > 0.05 => kept in the 5% confidence set; p <= 0.05 => significantly
# inferior. Results are saved to a tidy dataframe (spec, rolling-window length,
# recency-decay delta, p-value) for easy visualisation.
#
# method="max" (not "R"): the R-test materialises a (reps x k x k) array which, at
# k=1050 models, needs ~17 GB and is OOM-killed. The max-test is O(reps x k) in memory
# and runs in ~1.5 min at reps=10000 while giving the same included/excluded decisions.
from arch.bootstrap import MCS

# One column per (spec, window, delta) model. Every model shares the same common-OOS
# dates (identical design indices + the same START_DATE gate), so the inner-join
# dropna() leaves the full matrix intact.
loss_cols, model_key = {}, {}
for label, design, feats in specs:
    for yr, w in zip(WINDOW_YEARS, WINDOWS):
        for delta in DELTAS:
            col = f"{label} | win={yr:.2f}y | delta={delta:.3f}"
            loss_cols[col] = rolling_log_ols_loss_series(design, feats, w, delta=delta)
            model_key[col] = (label, round(float(yr), 2), round(float(delta), 3))
losses = pd.DataFrame(loss_cols).dropna()
assert losses.notna().all().all() and len(losses) > 0
print(f"MCS loss matrix: {losses.shape[0]} OOS days x {losses.shape[1]} models")

mcs = MCS(losses, size=0.05, reps=10000, block_size=None,
          method="max", bootstrap="stationary", seed=42)
mcs.compute()

# Tidy results: spec, rolling-window length (years), recency-decay delta, MCS p-value.
pv = mcs.pvalues["Pvalue"]
mcs_results = pd.DataFrame(
    [(*model_key[c], float(pv[c]), bool(pv[c] > 0.05)) for c in losses.columns],
    columns=["spec", "window_years", "delta", "pvalue", "in_mcs"],
).sort_values(["pvalue", "spec", "window_years", "delta"],
              ascending=[False, True, True, True]).reset_index(drop=True)
mcs_results.to_parquet("mcs_results_with_macro.parquet")   # saved for visualisation

pd.set_option("display.float_format", lambda v: f"{v:.4f}")
print(f"\\n{int(mcs_results['in_mcs'].sum())} models in the 5% MCS; "
      f"{int((~mcs_results['in_mcs']).sum())} excluded (p<=0.05).")
print(f"Lowest mean-QLIKE model: {losses.mean().idxmin()}  ({losses.mean().min():.6f})")
print("\\nMCS p-value range:", f"{pv.min():.4f} .. {pv.max():.4f}",
      f"(median {pv.median():.4f})")
print("\\nLeast-supported models (lowest MCS p-value):")
print(mcs_results.tail(10).to_string())
mcs_results'''


def add_macro_bits(cells):
    # insert helper into cell 3 (after rolling_log_ols_eval return, before the prints)
    c3 = cells[3]
    s = "".join(c3["source"])
    anchor = '    return q.mean(), len(q), clip\n\nprint(f"Common OOS start:'
    assert anchor in s, "cell3 anchor not found"
    s = s.replace('    return q.mean(), len(q), clip\n\n',
                  '    return q.mean(), len(q), clip\n' + HELPER + '\n', 1)
    c3["source"] = s.splitlines(keepends=True)
    clear(c3)
    # append MCS cell
    new_cell = {
        "cell_type": "code",
        "id": "mcs8",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": MCS_CELL.splitlines(keepends=True),
    }
    cells.append(new_cell)


apply_grid("HAR_simpleOLS_3d.ipynb", is_macro=False)
apply_grid("HAR_simpleOLS_3d_with_macro.ipynb", is_macro=True)
print("done")
