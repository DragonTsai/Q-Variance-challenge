# baseline_fit.py
# comment out lines to read files in 3 parts or 1 part
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# load the parquet files from data_loader.py
###df = pd.concat([pd.read_parquet("dataset_part1.parquet"),pd.read_parquet("dataset_part2.parquet"),pd.read_parquet("dataset_part3.parquet")])


df = pd.read_parquet("dataset.parquet")  # READ TEST DATA

# Select S&P 500, T=5
# data = df[(df["ticker"] == "^GSPC") & (df["T"] == 5)].copy()
# data = df[(df["ticker"] == "^GSPC") ].copy()
data = df.copy()
data["var"] = data.sigma**2

#print(f"S&P 500 T=5: {len(data)} windows")
print(f"{len(data)} windows")
# print(f"mean z = {data["z"].mean()}")  # mean of z is zero
print(f"z has NaNs: {data['z'].isna().sum()}")  # → 0


zmax = 0.6
delz = 0.025*2
nbins = int(2*zmax/delz + 1)
#bins = np.linspace(-0.5, 0.5, 41)         # fixed bins
bins = np.linspace(-zmax, zmax, nbins)         # fixed bins

# create data frame with e.g. zbin = (-0.601, -0.55], z_mid, sigma
binned = (data.assign(z_bin=pd.cut(data.z, bins=bins, include_lowest=True))
               .groupby('z_bin',observed=False)
               .agg(z_mid=('z', 'mean'), var=('var', 'mean'))
               .dropna())

# zmid = (bins[0:(nbins-1)] + bins[1:(nbins)])/2

def qvar(z, s0, zoff):    # define q-variance function, parameter is minimal volatility s0
    return (s0**2 + (z - zoff)**2 / 2)

# curve_fit returns a value popt and a covariance pcov, the _ means we ignore the pcov
###popt, _ = curve_fit(qvar, binned.z_mid, binned["var"], p0=[0.02, 0])
popt = [0.255, 0.020]  # same as optimized fit to data

fitted = qvar(binned.z_mid, popt[0], popt[1])  # cols are z_bin, which is a range like (-0.601, -0.55], and qvar
r2 = 1 - np.sum((binned["var"] - fitted)**2) / np.sum((binned["var"] - binned["var"].mean())**2)

print(f"σ₀ = {popt[0]:.4f}  zoff = {popt[1]:.4f}  R² = {r2:.4f}")

# plot of all stocks
markfac = 1  # default is 1, can increase to 3 if less data points
plt.figure(figsize=(9,7))
plt.scatter(data.z, data['var'], c='steelblue', alpha=markfac*0.1, s=markfac*1, edgecolor='none')
numeric_array = (1 - data["T"]/130)
string_array = [str(x) for x in numeric_array]
#plt.scatter(data.z, data['var'], c='steelblue', alpha=numeric_array, s=1, edgecolor='none')
#plt.scatter(data.z, data['var'], c=string_array, alpha=0.1, s=1, edgecolor='none')
plt.plot(binned.z_mid, binned['var'], 'b-', lw=3)     # label='binned'
plt.plot(binned.z_mid, fitted, 'red', lw=3, label=f'σ₀ = {popt[0]:.3f}, zoff = {popt[1]:.3f}, R² = {r2:.3f}')

plt.xlabel('z (scaled log return)', fontsize=12)
plt.ylabel('Annualised variance', fontsize=12)
plt.title('All data T=1 to 26 weeks – Q-Variance', fontsize=14)

plt.xlim(-zmax, zmax) 
plt.ylim(0.0, 0.35)

plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
