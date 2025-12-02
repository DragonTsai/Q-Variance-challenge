# data_loader.py
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

HORIZONS = 5*(np.arange(26)+1)   # does 1 to 26 weeks, can also do [5, 10, 20, 40, 80, 160]

TICKERS = ["A" , "AAPL", "ABT", "ACGL","ADBE" , "ADI" , "ADM" , "ADP" , "ADSK" , "AEE" , "AEP" ,
"AES" , "AFL" , "AIG" , "AJG" , "AKAM" , "ALB" , "ALL" , "AMAT" , "AMD" , "AME" , "AMGN",
 "AMT" , "AMZN" , "AON" , "AOS" , "APA" , "APD" , "APH" , "ARE" , "ATO" , "AVB",  
"AVY" , "AXP" , "AZO" , "BA" , "BAC" , "BALL" , "BAX" , "BBY" , "BDX" , "BEN" , "BIIB",
"BK" , "BKNG" , "BKR" , "BLK" , "BMY" , "BRO" , "BSX" , "BXP" , "C" , "CAG" , "CAH",
"CAT" , "CB" , "CCI" , "CCL" , "CDNS" , "CHD" , "CHRW" , "CI" , "CINF" , "CL" , "CLX",  
"CMCSA" , "CMI" , "CMS" , "CNP" , "COF" , "COO" , "COP" , "COR" , "COST" , "CPB" , "CPRT",
"CPT" , "CSCO" , "CSGP" , "CSX" , "CTAS" , "CTRA" , "CTSH" , "CVS" , "CVX" , "D" , "DD",
"DE" , "DECK" , "DGX" , "DHI" , "DHR" , "DIS" , "DLTR" , "DOC" , "DOV" , "DRI" , "DTE",
"DUK" , "DVA" , "DVN" , "EA" , "EBAY" , "ECL" , "ED" , "EFX" , "EG" , "EIX" , "EL",
"EMN" , "EMR" , "EOG" , "EQR" , "EQT" , "ERIE" , "ES" , "ESS" , "ETN" , "ETR" , "EVRG",
"EW" , "EXC" , "EXPD" , "F" , "FAST" , "FCX" , "FDS" , "FDX" , "FE" , "FFIV" , "FI",
"FICO" , "FITB" , "FRT" , "GD" , "GE" , "GEN" , "GILD" , "GIS" , "GL" , "GLW" , "GPC", 
"GS" , "GWW" , "HAL" , "HAS" , "HBAN" , "HD"  , "HIG" , "HOLX" , "HON" , "HPQ",  
"HRL" , "HSIC" , "HST" , "HSY" , "HUBB" , "HUM" , "IBM" , "IDXX" , "IEX" , "IFF" , "INCY",
"INTC" , "INTU" , "IP" , "IPG" , "IRM" , "IT" , "ITW" , "IVZ" , "J" , "JBHT" , "JBL",
 "JCI" , "JKHY" , "JNJ" , "JPM" , "K" , "KEY" , "KIM" , "KLAC" , "KMB" , "KMX",  
 "KO" , "KR" , "L" , "LEN" , "LH" , "LHX" , "LII" , "LIN" , "LLY" , "LMT" , "LNT",
 "LOW" , "LRCX" , "LUV" , "MAA" , "MAR" , "MAS" , "MCD" , "MCHP" , "MCK" , "MCO" , "MDT",
 "MET" , "MGM" , "MHK" , "MKC" , "MLM" , "MMC" , "MMM" , "MNST" , "MO" , "MOS" , "MRK",
"MS" , "MSFT" , "MSI" , "MTB" , "MTCH" , "MTD" , "MU" , "NDSN" , "NEE" , "NEM" , "NI",
"NKE" , "NOC" , "NSC" , "NTAP" , "NTRS" , "NUE" , "NVDA" , "NVR" , "O" , "ODFL" , "OKE",
"OMC" , "ORCL" , "ORLY" , "OXY" , "PAYX" , "PCAR" , "PCG" , "PEG" , "PEP" , "PFE" , "PG", 
 "PGR" , "PH" , "PHM" , "PKG" , "PLD" , "PNC" , "PNR" , "PNW" , "POOL" , "PPG" , "PPL",
 "PSA" , "PTC" , "PWR" , "QCOM" , "RCL" , "REG" , "REGN" , "RF" , "RJF" , "RL" , "RMD",
"ROK" , "ROL" , "ROP" , "ROST" , "RSG" , "RTX" , "RVTY" , "SBAC" , "SBUX" , "SCHW" , "SHW",
 "SJM" , "SLB" , "SNA" , "SNPS" , "SO" , "SPG" , "SPGI" , "SRE" , "STE" , "STLD" , "STT",
"STZ" , "SWK" , "SWKS" , "SYK" , "SYY" , "T" , "TAP" , "TDY" , "TECH" , "TER" , "TFC",
 "TGT" , "TJX" , "TKO" , "TMO" , "TPL" , "TRMB" , "TROW" , "TRV" , "TSCO" , "TSN" , "TT", 
"TTWO" , "TXN" , "TXT" , "TYL" , "UDR" , "UHS" , "UNH" , "UNP" , "UPS" , "URI" , "USB",
"VLO" , "VMC" , "VRSN" , "VRTX" , "VTR" , "VTRS" , "VZ" , "WAB" , "WAT" , "WDC",   
"WEC" , "WELL" , "WFC" , "WM" , "WMB" , "WMT" , "WRB" , "WSM" , "WST" , "WY" , "XEL",
"XOM" , "YUM" , "ZBRA"]
ntick = len(TICKERS)

# Path("cache").mkdir(exist_ok=True)
all_data = []

print("Generating Q-Variance Challenge Dataset...")

for ticker in TICKERS:
    print(f"→ {ticker}", end="")
    price = yf.download(ticker, period="max", progress=False, auto_adjust=True)["Close"]
    
    ret = np.log(price).diff().dropna().values

    rows = []
    scale = np.sqrt(252)

    for T in HORIZONS:
        i = 0
        while i + T <= len(ret):
            window = ret[i:i+T]     # T points
            if len(window) < T * 0.8:
                break

            x = window.sum()   # total price change over the period
            sigma = np.std(window, ddof=0) * scale  # use np.std to get std over period, ddof=1 means divisor is N-1
            z_raw = x / np.sqrt(T / 252.0)

            # REJECT BAD WINDOWS
            if not (np.isfinite(sigma) and sigma > 0 and np.isfinite(z_raw)):
                i += T
                continue

            rows.append({          # append row of data for this period
                "ticker": ticker,
                "date": price.index[i + T - 1].date(),
                "T": T,
                "z_raw": float(z_raw),
                "sigma": float(sigma)
            })
            i += T

    if not rows:
        print(" [no data]")
        continue

    df = pd.DataFrame(rows)

    # CLEAN BEFORE DE-MEANING 
    df = df[np.isfinite(df['z_raw']) & np.isfinite(df['sigma']) & (df['sigma'] > 0)]

    # NOW de-mean safely, this step groups by ticker and T, and subtracts the group mean 
    df["z"] = df.groupby(["ticker", "T"])["z_raw"].transform(lambda g: g - g.mean())

    df = df.drop(columns="z_raw")
    df = df.dropna().reset_index(drop=True)  # Final clean

    print(f" → {len(df)} clean windows")
    all_data.append(df)

full = pd.concat(all_data, ignore_index=True)

n = len(full) // 3
part1 = full.iloc[:n]
part2 = full.iloc[n:2*n]
part3 = full.iloc[2*n:]

# Save 3 small files
part1.to_parquet("dataset_part1.parquet", compression=None)
part2.to_parquet("dataset_part2.parquet", compression=None)
part3.to_parquet("dataset_part3.parquet", compression=None)

print("Done! 3 files created — each <25 MB")
