import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.subheader("📊 Technische Indikatoren")

# Daten laden
csv_path = "data/btc_bitget_7days.csv"
df = pd.read_csv(csv_path)
df["datetime"] = pd.to_datetime(df["datetime"])

# Indikatoren berechnen
df["sma_20"] = df["close"].rolling(window=20).mean()
df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

# RSI 14
delta = df["close"].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
rs = gain / loss
df["rsi_14"] = 100 - (100 / (1 + rs))

# OBV
df["obv"] = 0
df.loc[1:, "obv"] = np.where(df["close"].diff().iloc[1:] > 0,
                             df["volume"].iloc[1:], 
                             -df["volume"].iloc[1:])
df["obv"] = df["obv"].cumsum()

# Diagramme vorbereiten
fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

# Preis mit SMA/EMA
axs[0].plot(df["datetime"], df["close"], label="Close", color="black")
axs[0].plot(df["datetime"], df["sma_20"], label="SMA 20", linestyle="--", color="green")
axs[0].plot(df["datetime"], df["ema_20"], label="EMA 20", linestyle=":", color="orange")
axs[0].set_title("Preis + SMA/EMA")
axs[0].legend()

# RSI
axs[1].plot(df["datetime"], df["rsi_14"], label="RSI 14", color="purple")
axs[1].axhline(70, color="red", linestyle="--")
axs[1].axhline(30, color="green", linestyle="--")
axs[1].set_title("RSI")

# OBV
axs[2].plot(df["datetime"], df["obv"], label="OBV", color="brown")
axs[2].set_title("On-Balance Volume (OBV)")

# Volumen
axs[3].bar(df["datetime"], df["volume"], label="Volume", color="gray")
axs[3].set_title("Volumen")

plt.tight_layout()

# In Streamlit anzeigen
st.pyplot(fig)