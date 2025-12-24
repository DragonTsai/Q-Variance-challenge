import numpy as np
import pandas as pd

# =====================================================
# Official horizons: 1–26 weeks
# =====================================================
HORIZONS = 5 * (np.arange(26) + 1)


# =====================================================
# CIR precision -> Poisson-weighted Gaussian mixture
# =====================================================
def simulate_price_series(
    n_days=300_000,
    s0=100.0,
    sigma0=0.25,
    kappa=0.02,
    c_int=15.0,
    seed=1,
    a_shape=1.5,
    lam_cap=500.0
):
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    sqrt_dt = np.sqrt(dt)

    # --- CIR target
    a = a_shape
    beta = sigma0**2
    theta = a / beta
    eta = np.sqrt(2.0 * kappa * theta / a)

    tau = theta
    logP = np.empty(n_days)
    logP[0] = np.log(s0)
    r = np.empty(n_days - 1)

    # Typical lambda at long-run tau
    lam_typ = min(lam_cap, c_int / theta)
    lam_typ = max(lam_typ, 1e-8)

    # Calibrate mixture variance
    s_unit2 = (sigma0**2 / 252.0) / lam_typ
    s_unit = np.sqrt(s_unit2)

    for t in range(1, n_days):
        # CIR step
        dW_tau = rng.standard_normal() * sqrt_dt
        tau_pos = max(tau, 0.0)
        tau = tau + kappa * (theta - tau_pos) * dt + eta * np.sqrt(tau_pos) * dW_tau
        tau = max(tau, 1e-10)

        # Shock intensity
        lam = c_int / tau
        if lam > lam_cap:
            lam = lam_cap

        # Poisson-weighted Gaussian mixture
        N = rng.poisson(lam)
        r_t = rng.normal(0.0, s_unit * np.sqrt(N)) if N > 0 else 0.0

        r[t - 1] = r_t
        logP[t] = logP[t - 1] + r_t

    prices = np.exp(logP)
    return prices, r


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
# Main
# =====================================================
def main():
    prices, returns = simulate_price_series(
        n_days=120_000,
        sigma0=0.25,
        kappa=0.02,
        c_int=10.0,
        seed=3
    )

    # -------------------------------------------------
    # CSV (first 100K points)
    # -------------------------------------------------
    n_out = 100_000

    df_out = pd.DataFrame({
        "Day": np.arange(n_out),
        "Price": prices[:n_out],
        "y": np.concatenate([[0.0], returns])[:n_out]
    })

    df_out.to_csv("qvariance_simulation_100k.csv", index=False)
    print("Saved qvariance_simulation_100k.csv")

    # -------------------------------------------------
    # Original outputs
    # -------------------------------------------------
    pd.Series(prices, name="Price").to_csv("prices.csv", index=False)

    df_qv = build_qvariance_dataset_from_prices(prices)
    df_qv.to_parquet("dataset.parquet", index=False)
    print(f"Saved dataset.parquet with {len(df_qv):,} rows")


if __name__ == "__main__":
    main()
