
"""
Clustered default simulation for risky zero-coupon bonds.

Model:
- All bonds are zero-coupon bonds with face value F.
- Risk-free zero price: P_rf(t,T) = F * exp(-r * tau)
- Risky pre-default price: P_i(t,T) = F * exp(-(r + s_i(t)) * tau)
- Credit spread follows a log-spread diffusion:
    d log s_i(t) = kappa_i * (log(theta_i) - log s_i(t)) dt
                  + beta_i * dX_t
                  + sigma_i * dW_i(t)
- Defaults are generated from a reduced-form default intensity:
    lambda_i(t) = lambda_0 + alpha_s * s_i(t)
                  + gamma_cum * cumulative_default_share
                  + eta_recent * recent_default_share
- Monthly default probability:
    PD_i(t,t+dt) = 1 - exp(-lambda_i(t) * dt)
- On default, price jumps to recovery * face value.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def simulate_clustered_zero_coupon_bonds(
    n_bonds=50,
    n_months=60,
    seed=42,
    face=100.0,
    recovery_rate=0.30,
    risk_free_rate=0.035,
    maturity_years=5.0,
    # initial spreads, in decimal form: 0.08 = 800 bps
    initial_spread_low=0.04,
    initial_spread_high=0.18,
    # long-run spread levels, also in decimal form
    long_run_spread_low=0.05,
    long_run_spread_high=0.12,
    # log-spread diffusion parameters
    kappa_low=0.35,
    kappa_high=0.85,
    spread_vol_low=0.20,
    spread_vol_high=0.40,
    beta_common_low=0.30,
    beta_common_high=0.75,
    # common sector stress factor
    sector_vol=0.25,
    default_stress_jump=0.35,
    sector_mean_reversion=0.60,
    # default intensity parameters, annualized
    lambda_0=0.005,
    alpha_spread=1.20,
    gamma_cumulative=0.35,
    eta_recent=0.60,
):
    rng = np.random.default_rng(seed)
    dt = 1.0 / 12.0

    bond_ids = [f"Bond_{i:02d}" for i in range(1, n_bonds + 1)]

    # Bond-specific parameters
    initial_spreads = rng.uniform(initial_spread_low, initial_spread_high, n_bonds)
    theta_spreads = rng.uniform(long_run_spread_low, long_run_spread_high, n_bonds)
    kappas = rng.uniform(kappa_low, kappa_high, n_bonds)
    sigmas = rng.uniform(spread_vol_low, spread_vol_high, n_bonds)
    betas = rng.uniform(beta_common_low, beta_common_high, n_bonds)

    log_spreads = np.log(initial_spreads)
    log_theta = np.log(theta_spreads)

    defaulted = np.zeros(n_bonds, dtype=bool)
    default_month = np.full(n_bonds, np.nan)

    # Storage
    records = []
    monthly_records = []

    # Sector stress state; positive values widen spreads through the common factor
    X = 0.0
    recent_default_share = 0.0

    for month in range(n_months + 1):
        t_years = month * dt
        tau = max(maturity_years - t_years, 0.0)

        spreads = np.exp(log_spreads)
        risk_free_price = face * np.exp(-risk_free_rate * tau)
        pre_default_prices = face * np.exp(-(risk_free_rate + spreads) * tau)
        prices = np.where(defaulted, recovery_rate * face, pre_default_prices)

        cumulative_default_share = defaulted.mean()

        for i in range(n_bonds):
            spread_discount_vs_rf = 1.0 - (pre_default_prices[i] / risk_free_price) if risk_free_price > 0 else np.nan

            records.append({
                "month": month,
                "year": round(t_years, 4),
                "bond_id": bond_ids[i],
                "tau_years": round(tau, 4),
                "risk_free_price": round(risk_free_price, 4),
                "spread_decimal": round(spreads[i], 6),
                "spread_bps": round(spreads[i] * 10000, 2),
                "pre_default_price": round(pre_default_prices[i], 4),
                "price": round(prices[i], 4),
                "spread_discount_vs_rf": round(spread_discount_vs_rf, 6),
                "defaulted": bool(defaulted[i]),
                "default_month": None if np.isnan(default_month[i]) else int(default_month[i]),
                "sector_stress": round(X, 6),
                "cumulative_default_share": round(cumulative_default_share, 6),
                "recent_default_share": round(recent_default_share, 6),
            })

        monthly_records.append({
            "month": month,
            "year": round(t_years, 4),
            "risk_free_price": round(risk_free_price, 4),
            "avg_price": round(float(np.mean(prices)), 4),
            "avg_spread_bps_alive": round(float(np.mean(spreads[~defaulted]) * 10000), 2) if np.any(~defaulted) else np.nan,
            "defaulted_bonds": int(defaulted.sum()),
            "cumulative_default_share": round(cumulative_default_share, 6),
            "sector_stress": round(X, 6),
        })

        if month == n_months or tau <= 0:
            break

        # 1) Draw defaults using current spreads and contagion terms
        defaults_this_month = np.zeros(n_bonds, dtype=bool)

        for i in range(n_bonds):
            if defaulted[i]:
                continue

            lambda_i = (
                lambda_0
                + alpha_spread * spreads[i]
                + gamma_cumulative * cumulative_default_share
                + eta_recent * recent_default_share
            )

            # Convert annualized intensity to period default probability.
            pd_i = 1.0 - np.exp(-lambda_i * dt)

            if rng.random() < pd_i:
                defaults_this_month[i] = True

        new_default_count = int(defaults_this_month.sum())
        new_default_share = new_default_count / n_bonds

        for i in np.where(defaults_this_month)[0]:
            defaulted[i] = True
            default_month[i] = month + 1

        # 2) Update the sector stress factor.
        # Defaults create a positive jump in X, which widens spreads for surviving bonds.
        dW_common = rng.normal(0.0, np.sqrt(dt))
        dX = (
            -sector_mean_reversion * X * dt
            + sector_vol * dW_common
            + default_stress_jump * new_default_share
        )
        X = X + dX

        # 3) Update log credit spreads for surviving bonds.
        # Defaulted bonds no longer need spread evolution because their price is fixed at recovery.
        eps_idio = rng.normal(0.0, 1.0, n_bonds)

        for i in range(n_bonds):
            if defaulted[i]:
                continue

            dlog_s = (
                kappas[i] * (log_theta[i] - log_spreads[i]) * dt
                + betas[i] * dX
                + sigmas[i] * np.sqrt(dt) * eps_idio[i]
            )

            log_spreads[i] = log_spreads[i] + dlog_s

            # Optional numerical guardrail: spreads between 1 bp and 5000 bps.
            log_spreads[i] = np.clip(log_spreads[i], np.log(0.0001), np.log(0.50))

        recent_default_share = new_default_share

    long_df = pd.DataFrame(records)
    monthly_df = pd.DataFrame(monthly_records)

    final_df = (
        long_df[long_df["month"] == long_df["month"].max()]
        [["bond_id", "price", "defaulted", "default_month", "spread_bps"]]
        .merge(
            long_df[long_df["month"] == 0][["bond_id", "price", "spread_bps"]]
            .rename(columns={"price": "initial_price", "spread_bps": "initial_spread_bps"}),
            on="bond_id",
            how="left"
        )
        .rename(columns={"price": "final_price", "spread_bps": "final_spread_bps"})
        [["bond_id", "initial_price", "initial_spread_bps", "final_price", "final_spread_bps", "defaulted", "default_month"]]
    )

    return long_df, monthly_df, final_df


if __name__ == "__main__":
    long_df, monthly_df, final_df = simulate_clustered_zero_coupon_bonds()

    long_df.to_csv("zero_coupon_credit_spread_paths.csv", index=False)
    monthly_df.to_csv("zero_coupon_monthly_default_summary.csv", index=False)
    final_df.to_csv("zero_coupon_final_bond_summary.csv", index=False)

    print(final_df)
    print()
    print(monthly_df.tail())

    plt.figure(figsize=(9, 5))
    plt.plot(monthly_df["month"], monthly_df["defaulted_bonds"], marker="o")
    plt.xlabel("Month")
    plt.ylabel("Cumulative defaulted bonds")
    plt.title("Clustered defaults from log credit-spread diffusion")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("zero_coupon_clustered_defaults.png", dpi=160)
    plt.show()
