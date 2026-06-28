# Gold RV forecasting — ML (XGBoost/LightGBM) vs HAR: findings & file map

Notes for future work (and future Claude Code sessions) so results don't have to be re-derived from the
notebooks. Last updated 2026-06-28.

## TL;DR
For 1-day-ahead gold realized-vol, **the linear log-HAR with implied vol (GVZ) + macro dummy is the model
to beat, and nothing tried beats it.** XGBoost — across log/level targets, custom-QLIKE vs squared-error
objectives, CV-ensembling, full Optuna tuning, weekly vs bi-weekly refit, wide L1/L2, and a regime-switching
blend — is always **≈8–11% worse on mean QLIKE** and is *worst of all in the high-vol regime* because trees
**cannot extrapolate** beyond their training range. Parsimony wins on every axis (features and model).

## Shared protocol (all experiments)
- **Target:** `RV_gold[t+1]` (next-day realized vol), forecast from info at day `t`. Data:
  `merged_RV_GVZ_with_macro_event.parquet` (cols `RV_gold, RV_crude, RV_ES, GVZ_close, macro_event`).
- **Features (matched / "Run 18"):** `x_d=log(RV)`, `x_w=mean(log RV,5)`, `x_m=mean(log RV,22)`,
  `log_GVZ`, `macro` (= `macro_event.shift(-1)`, scheduled releases, known in advance). "rich" adds
  `log_RV_crude`, `log_RV_ES`.
- **Window:** 6-year rolling = 1512 trading days. **Recency:** geometric `δ=0.999` (mean-1 normalised).
- **OOS:** common dates 2017-07-11 → 2026-05-28 (**2229 days**), gated so every model shares them.
- **Metric:** QLIKE in levels `y/f - log(y/f) - 1` (Duan smearing for log models). **Test:**
  `arch.bootstrap.MCS` (size 0.05, reps 10000, stationary bootstrap, method "R", seed 42). p>0.05 ⇒ in 5% MCS.

## Champion: log-HAR (OLS) — daily refit, δ=0.999
| model | mean QLIKE | MCS p |
|---|---|---|
| har_run20 (HAR + crude + GVZ + macro) | **0.027481** | 1.00 |
| har_run19 (HAR + SPX + GVZ + macro) | 0.027546 | 0.55 |
| har_run18 (HAR + GVZ + macro) | 0.027551 | 0.55 |

Cross-asset RVs (crude/SPX) add ~nothing beyond GVZ+macro (4th-5th decimal). GVZ + macro are the useful
extras; the RV lags + GVZ carry the model.

## XGBoost experiments (all lose to HAR on mean QLIKE)
Protocol changes only move **MCS membership** (a variance effect on the loss *difference*), never accuracy.
- **v1 — log target + squared error + Duan smear + fixed params + daily refit:** xgb_matched 0.029005
  (p=0.027), xgb_rich 0.028906 (p=0.0029) → **excluded** from 5% MCS. *(superseded in-notebook.)*
- **v2 — level target + custom-QLIKE objective + 4-fold CV-ensemble + early stopping + bi-weekly refit:**
  xgb_matched 0.030028, xgb_rich 0.029950 (p≈0.072) → in MCS, but worse mean QLIKE (MCS inclusion ≠ better).
- **No-CV ablation (single fixed default `max_depth=3, lr=0.05`, no grid, no ensemble):** xgb_matched_nocv
  **0.029671**, xgb_rich_nocv 0.029899 → **beats the CV-ensemble** and ~40× faster. CV+ensemble buys nothing.
- **Optuna tuning (nested walk-forward CV, annual re-tune ×9, 30 TPE trials, 8-param search, max_depth≤4):**
  best ~0.0299–0.0307 depending on variant — **does not beat the simple no-CV default**, and the tuner almost
  always picks **`max_depth=2`** (shallowest).
- **Fold-weighting in tuning:** recency-weighted folds [0.14,0.20,0.29,0.41] narrowly beat equal folds
  [0.25×4] (0.030503 vs 0.030682). Per-observation δ=0.999 weight: small, protocol-dependent effect.
- **Weekly (5d) vs bi-weekly (10d) refit:** no meaningful difference (5th decimal).
- **Regularisation to fight collinearity:** even with `reg_lambda, reg_alpha ∈ [1e-3, 50]`, the tuner picks
  **small** L1/L2 (λ≈0.06–0.08, α≈0.01–0.02) and instead uses `colsample_bytree≈0.66` + shallow trees.
  Trees handle the collinearity *structurally* (column subsampling); heavy L1/L2 isn't wanted.

## Collinearity (matched features)
VIF: `x_w 6.97, log_GVZ 6.87, x_m 5.68, x_d 3.42, macro 1.00`. Moderate (all <10); `x_d/x_w/x_m/log_GVZ`
correlated 0.70–0.89, `macro` independent. Not severe enough to hurt trees.

## Feature importance (mean gain, stable across all XGB variants)
`log_GVZ` (~0.30–0.45) and `x_w`/weekly lag (~0.28–0.34) dominate; `x_d` ~0.13–0.22; `macro` ~0.07;
`x_m` and cross-asset RVs minor. Trees lean on the same GVZ + RV-lag signal as the HAR, less efficiently.

## Regime-aware ensemble (75/25 XGB/OLS by vol regime) — the key diagnostic
Blend of `xgb_optuna_recency` (XGB) and `har_run18` (OLS); regime = current RV vs **expanding 95th pct**
(no look-ahead). High-vol = 125/2229 days (5.6%). High→75% XGB, low→75% OLS.
- **regime_ensemble mean QLIKE = 0.028318** (p=0.319, in MCS) — better than pure XGB but **worse than pure
  OLS** (0.0275). Pure XGB falls **out** of this MCS (p≈0.025).
- **Per-regime mean QLIKE (the punchline):**
  | | high-vol (n=125) | low-vol (n=2104) |
  |---|---|---|
  | har_run18 (OLS) | **0.0521** | 0.0261 |
  | xgb_optuna_recency | 0.0744 | 0.0279 |
  | regime_ensemble | 0.0648 | 0.0262 |
  XGBoost is **worst exactly in high-vol** (0.0744 vs OLS 0.0521, ~43% worse) → up-weighting it there *hurts*.
- **Mechanism:** trees output the (weighted) mean of training points in a leaf, so forecasts are **capped at
  the max seen in training**. When vol prints a new high (>95th pct, often a fresh high), XGBoost
  under-predicts the spike; the log-linear OLS extrapolates through it. **Tree inability to extrapolate is
  disqualifying precisely at the extremes.**

## What works best when vol jumps outside its historical range / at extremes
Need two properties: **(1) extrapolation** (parametric/log-linear form, not interpolating trees/kNN) and
**(2) forward-looking inputs** (implied vol GVZ/VIX/OVX, variance risk premium, option-implied skew, macro
calendar). Recommended families (all keep both):
- **HARX (HAR + GVZ + macro)** — current champion.
- **HARQ / HAR-J / CHAR** — add jump / measurement-precision response at extremes (cheap, linear). *Natural
  next experiment; reuses existing machinery.*
- **TVP-HARX via Kalman filter** — coefficients adapt during regime shifts (the "Kalman" idea; alternative
  to HAR, not a component of XGBoost).
- **GARCH-X / GARCH-MIDAS / Realized-GARCH with Student-t + asymmetry (GJR/EGARCH)** — fat tails + exog IV.
- **Markov regime-switching HARX** — explicit high-vol state (principled version of the regime ensemble).
- **Avoid pure tree ensembles for the extreme regime** (no extrapolation). NN with linear output / TFT *can*
  extrapolate and take known-future inputs but overfit on this sample size — exploratory only.

## File map (so results needn't be recomputed)
- `XGBoost_vs_HAR_MCS_6y.ipynb` — 4 XGB variants (CV-ensemble + no-CV, matched + rich) vs 3 HAR; level/QLIKE
  objective, bi-weekly, max_depth≤4. → `xgb_vs_har_losses_6y.parquet` (7-model per-day QLIKE, **canonical
  cache** reused by other notebooks), `mcs_xgb_vs_har_6y.parquet`, `xgb_feature_importance_6y.png`,
  `xgb_vs_har_cum_qlike_6y.png`.
- `XGBoost_optuna_tuned_matched_6y.ipynb` — Optuna nested WFCV (weekly refit, wide L1/L2), recency vs equal
  fold tuning, **regime-aware ensemble**, collinearity (VIF), param trajectories, "Why it matters" (shallow
  trees) writeup. → `xgb_optuna_regime_losses_6y.parquet`, `mcs_xgb_optuna_regime_6y.parquet`,
  `xgb_optuna_regime_cumqlike.png`, `xgb_optuna_weekly_params.png` (+ earlier `*_foldweight_*`, `*_weekly_*`).
- HAR source of truth: `HAR_simpleOLS_3d_with_macro.ipynb` (recency weights, QLIKE, Duan smearing, 3D
  surfaces, full MCS `mcs_results_with_macro.parquet`).
- **Not authored/verified in this analysis** (parallel IDE work, results not summarised here):
  `LightGBM_vs_HAR_MCS_6y.ipynb`, `ExpandingML_vs_HAR_MCS_6y.ipynb`,
  `ExpandingML_recency_compare_6y.ipynb`, `Ensemble_LGB_XGB_HAR_MCS_6y.ipynb` and their `lgb_*`,
  `expandingml_*`, `ensemble_*` outputs.

## Environment / repro
- Interpreter: `/Library/Frameworks/Python.framework/Versions/3.10/bin/python3` (has pandas/numpy/sklearn/
  arch/xgboost/optuna/pyarrow/matplotlib; `python`/`jupyter` not on PATH — use `python3 -m jupyter`).
- **xgboost OpenMP fix (one-time):** the wheel needs `libomp.dylib`; pointed at scikit-learn's bundled copy
  via `install_name_tool -add_rpath <sklearn>/.dylibs <xgboost>/lib/libxgboost.dylib`. After this
  `import xgboost` works with no env vars.
- XGBoost runs use `tree_method="hist"`, `nthread=2`. Run a notebook headless:
  `python3 -m jupyter nbconvert --to notebook --execute --inplace <nb>.ipynb`.
