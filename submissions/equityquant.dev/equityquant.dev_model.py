import numpy as np
import pandas as pd

# =====================================================
# Official horizons: 1–26 weeks
# =====================================================
HORIZONS = 5 * (np.arange(26) + 1)


# =====================================================
# Fast-mixing Tau–Poisson Gaussian mixture
# =====================================================
def simulate_price_series(
    n_days=300_000,
    s0=100.0,

    # ----------- free parameters (≤ 3) -----------
    sigma0=0.25,     # overall volatility scale
    c_int=10.0,      # Poisson intensity scale
    kappa=0.50,      # speed of mean reversion (fast mixing)
    # ---------------------------------------------

    seed=1,

    # fixed (do not tune)
    a_shape=1.5,
    lam_cap=500.0,
    burn_in=2000
):
    """
    Fast-mixing CIR-like precision tau -> Poisson N -> Gaussian sqrt(N)

    Key properties:
    - q-variance emerges structurally
    - time-invariant
    - robust to simulation length
    - no skew hard-coded
    """

    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    sqrt_dt = np.sqrt(dt)

    # --- CIR stationary target for tau
    a = a_shape
    beta = sigma0**2
    theta = a / beta
    eta = np.sqrt(2.0 * kappa * theta / a)

    # start tau at stationary mean
    tau = theta

    # stationary-mean lambda
    lam_bar = min(lam_cap, c_int / theta)
    lam_bar = max(lam_bar, 1e-8)

    # fixed mixture variance scale
    s_unit2 = (sigma0**2 / 252.0) / lam_bar
    s_unit = np.sqrt(s_unit2)

    logP = np.empty(n_days + burn_in)
    logP[0] = np.log(s0)

    for t in range(1, n_days + burn_in):
        # --- fast-mixing tau step
        tau_pos = max(tau, 0.0)
        tau = (
            tau
            + kappa * (theta - tau_pos) * dt
            + eta * np.sqrt(tau_pos) * rng.standard_normal() * sqrt_dt
        )
        tau = max(tau, 1e-10)

        # --- Poisson intensity
        lam = c_int / tau
        lam = min(lam, lam_cap)

        # --- Poisson–Gaussian mixture
        N = rng.poisson(lam)
        r_t = rng.normal(0.0, s_unit * np.sqrt(N)) if N > 0 else 0.0

        logP[t] = logP[t - 1] + r_t

    # drop burn-in
    logP = logP[burn_in:]
    prices = np.exp(logP)

    return prices


# =====================================================
# Q-variance dataset builder (OFFICIAL, OVERLAPPING)
# =====================================================
def build_qvariance_dataset_from_prices(prices, ticker="DRAGON"):
    ret = np.diff(np.log(prices))
    scale = np.sqrt(252.0)

    rows = []
    for T in HORIZONS:
        for i in range(0, len(ret) - T + 1):   # OVERLAPPING
            window = ret[i:i + T]

            x = window.sum()
            sigma = np.std(window, ddof=0) * scale
            z_raw = x / np.sqrt(T / 252.0)

            if not (np.isfinite(sigma) and sigma > 0 and np.isfinite(z_raw)):
                continue

            rows.append({
                "ticker": ticker,
                "date": int(i),
                "T": int(T),
                "z_raw": float(z_raw),
                "sigma": float(sigma),
            })

    df = pd.DataFrame(rows)

    # Official de-meaning step
    df["z"] = df.groupby("T")["z_raw"].transform(lambda g: g - g.mean())
    df = df.drop(columns="z_raw").dropna().reset_index(drop=True)

    return df[["ticker", "date", "T", "z", "sigma"]]


# =====================================================
# Main (submission-style outputs)
# =====================================================
def main():
    prices = simulate_price_series(
        n_days=100_000,
        sigma0=0.25,
        c_int=10.0,
        kappa=0.50,
        seed=6
    )

    # -------------------------------------------------
    # CSV requested by competition (first 100K points)
    # -------------------------------------------------
    n_out = 100_000
    df_out = pd.DataFrame({
        "Day": np.arange(n_out),
        "Price": prices[:n_out]
    })
    df_out.to_csv("qvariance_simulation_100k.csv", index=False)
    print("Saved qvariance_simulation_100k.csv")

    # -------------------------------------------------
    # Q-variance dataset
    # -------------------------------------------------
    df_qv = build_qvariance_dataset_from_prices(prices)
    df_qv.to_parquet("dataset.parquet", index=False)
    print(f"Saved dataset.parquet with {len(df_qv):,} rows")


if __name__ == "__main__":
    main()
