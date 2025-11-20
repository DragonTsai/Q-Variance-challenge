# baseline_fit.py
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

df = pd.read_parquet("prize_dataset.parquet")   # load the parquet file from data_loader.py

# Select S&P 500, T=5
# data = df[(df["ticker"] == "^GSPC") & (df["T"] == 5)].copy()
# data = df[(df["ticker"] == "^GSPC") ].copy()
data = df.copy()

#print(f"S&P 500 T=5: {len(data)} windows")
print(f"z has NaNs: {data['z'].isna().sum()}")  # → 0

bins = np.linspace(-0.5, 0.5, 25)         # fixed bins
# create data frame with e.g. zbin = (-0.601, -0.55], z_mid, sigma
binned = (data.assign(z_bin=pd.cut(data.z, bins=bins, include_lowest=True))
               .groupby('z_bin')
               .agg(z_mid=('z', 'mean'), sigma=('sigma', 'mean'))
               .dropna())

def qvar(z, s0, zoff):    # define q-variance function, parameter is minimal volatility s0
    return np.sqrt(s0**2 + (z - zoff)**2 / 2)

# curve_fit returns a value popt and a covariance pcov
popt, _ = curve_fit(qvar, binned.z_mid, binned.sigma, p0=[0.12, 0])
##popt[0] = np.sqrt(np.mean(data['sigma']**2 - data['z']**2/2) )
fitted = qvar(binned.z_mid, popt[0], popt[1])  # cols are z_bin, which is a range like (-0.601, -0.55], and qvar
r2 = 1 - np.sum((binned.sigma - fitted)**2) / np.sum((binned.sigma - binned.sigma.mean())**2)

print(f"\nQ-VARIANCE — WILMOTT & ORRELL, JULY 2025")
print(f"σ₀ = {popt[0]:.4f}  zoff = {popt[1]:.4f}  R² = {r2:.4f}")

# The plot from your paper — exact
plt.figure(figsize=(10,7))
plt.scatter(data.z, data.sigma, c='steelblue', alpha=0.5, s=5, edgecolor='none')
plt.plot(binned.z_mid, binned['sigma'], 'b-', lw=3)     # label='binned'
plt.plot(binned.z_mid, fitted, 'red', lw=4, label=f'σ₀ = {popt[0]:.3f}, zoff = {popt[1]:.3f}, R² = {r2:.3f}')
plt.xlabel('z (scaled log return)', fontsize=12)
plt.ylabel('Annualised realised volatility', fontsize=12)
plt.title('All data T=1 to 26 weeks – Q-Variance', fontsize=14)

plt.xlim(-0.5, 0.5)  
plt.ylim(0.0, 0.6)

plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
#plt.show()

# now do a 4×2 panel figure for all 8 assets (Wilmott-style)
# The 8 tickers in the order you want them to appear
TICKERS = ["^GSPC", "^DJI", "^FTSE", "AAPL", "MSFT", "AMZN", "JPM","BTC-USD"]
# T = 20                                      # fixed horizon

# Set up the 4×2 grid
fig, axes = plt.subplots(4, 2, figsize=(6.5, 10), sharex=True, sharey=True)
axes = axes.flatten()                       # makes indexing easy: axes[0] … axes[7]

for idx, ticker in enumerate(TICKERS):
    ax = axes[idx]
    data = df[(df.ticker == ticker)].copy()

    # Create data frame with e.g. zbin = (-0.601, -0.55], z_mid, sigma
    binned = (data.assign(z_bin=pd.cut(data.z, bins=bins, include_lowest=True))
                  .groupby('z_bin')
                  .agg(z_mid=('z', 'mean'), sigma=('sigma', 'mean'))
                  .dropna())

    # Curve_fit returns a value popt and a covariance pcov, the _ means we ignore the pcov
    popt, _ = curve_fit(qvar, binned.z_mid, binned.sigma, p0=[0.12, 0])     #  maxfev=2000
    fitted = qvar(binned.z_mid, popt[0], popt[1])
    
    r2 = 1 - np.sum((binned.sigma - fitted)**2) / np.sum((binned.sigma - binned.sigma.mean())**2)

    # Plot scatter + fit
    ax.scatter(data.z, data.sigma, c='steelblue', alpha=0.5, s=5, edgecolor='none')
    ax.plot(binned.z_mid, binned['sigma'], 'b-', lw=3)
    ax.plot(binned.z_mid, fitted, 'red', lw=3, label=f'σ₀ = {popt[0]:.3f}, zoff = {popt[1]:.4f}, R² = {r2:.3f}')

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0.0, 0.6)
    ax.set_title(ticker, fontsize=10, pad=15)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

# Shared labels
fig.supxlabel('z (scaled log return)', fontsize=10, y=0.04)
fig.supylabel('Annualised realised volatility', fontsize=10, x=0.04)

# Big main title
plt.suptitle('Non-overlapping windows across 8 major assets', fontsize=12, y=0.96, weight='bold')

plt.tight_layout(rect=[0.05, 0.05, 1, 0.94])   # makes room for suptitle
plt.show()

# Optional: save as high-res PNG/PDF for the announcement
# plt.savefig("q_variance_8_assets.png", dpi=300, bbox_inches='tight')
# plt.savefig("q_variance_8_assets.pdf", bbox_inches='tight')
