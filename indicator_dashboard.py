import pandas as pd
import matplotlib.pyplot as plt
import ta  # technischer Indikatoren-Support (pip install ta)
import os

# === Daten laden ===
csv_path = "data/btc_bitget_7days.csv"
df = pd.read_csv(csv_path)
df["datetime"] = pd.to_datetime(df["datetime"])

# === Indikatoren berechnen ===
df["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
df["ema_20"] = ta.trend.ema_indicator(df["close"], window=20)
df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
macd = ta.trend.macd(df["close"])
df["macd"] = macd.macd()
df["macd_signal"] = macd.macd_signal()
bb = ta.volatility.BollingerBands(df["close"], window=20)
df["boll_upper"] = bb.bollinger_hband()
df["boll_lower"] = bb.bollinger_lband()
df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])

# === Plotten ===
fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

# 1 – Preis + SMA/EMA + Bollinger
axs[0].plot(df["datetime"], df["close"], label="Close", color="black")
axs[0].plot(df["datetime"], df["sma_20"], label="SMA 20", linestyle="--")
axs[0].plot(df["datetime"], df["ema_20"], label="EMA 20", linestyle=":")
axs[0].plot(df["datetime"], df["boll_upper"], label="Bollinger Upper", alpha=0.4)
axs[0].plot(df["datetime"], df["boll_lower"], label="Bollinger Lower", alpha=0.4)
axs[0].set_title("Preis, SMA, EMA, Bollinger")
axs[0].legend()

# 2 – RSI
axs[1].plot(df["datetime"], df["rsi_14"], label="RSI 14", color="purple")
axs[1].axhline(70, color="red", linestyle="--")
axs[1].axhline(30, color="green", linestyle="--")
axs[1].set_title("RSI")

# 3 – MACD
axs[2].plot(df["datetime"], df["macd"], label="MACD", color="blue")
axs[2].plot(df["datetime"], df["macd_signal"], label="Signal", color="orange")
axs[2].set_title("MACD & Signal")
axs[2].legend()

# 4 – OBV
axs[3].plot(df["datetime"], df["obv"], label="OBV", color="brown")
axs[3].set_title("On-Balance Volume (OBV)")

plt.tight_layout()
plt.show()