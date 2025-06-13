import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests
import time
import datetime
import csv
import os
import numpy as np
from chart_utils import draw_price_chart

# === API-Funktion ====
def fetch_bitget_spot_data_and_save(symbol="BTCUSDT", granularity="1h", filename="btc_bitget_7days.csv"):
    url = "https://api.bitget.com/api/v2/spot/market/candles"
    params = {
        "symbol": symbol,
        "granularity": granularity,
        "limit": "168"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data["code"] != "00000":
            raise Exception(f"Bitget API Fehler: {data['msg']}")
        candles = sorted(data["data"], key=lambda x: int(x[0]))
    except Exception as e:
        raise Exception(f"API-Zugriff fehlgeschlagen: {e}")

    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", filename)

    with open(path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "datetime", "open", "high", "low", "close", "volume"])
        for c in candles:
            ts = int(c[0])
            dt = datetime.datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([ts, dt] + c[1:])

    return path


# === Streamlit App ===
st.title("📊 DaVinci Trading App – BTC 1H Analyse")

if "csv_created" not in st.session_state:
    st.session_state.csv_created = False

# 📥 API-Daten abrufen
if st.button("📥 API-Daten abrufen und CSV erstellen"):
    try:
        csv_path = fetch_bitget_spot_data_and_save()
        st.session_state.csv_created = True
        st.success(f"✅ CSV erfolgreich erstellt: {csv_path}")
    except Exception as e:
        st.session_state.csv_created = False
        st.error(f"❌ Fehler: {e}")
        st.stop()

# 📊 Visualisierung nur wenn CSV vorhanden
csv_path = "data/btc_bitget_7days.csv"
if st.session_state.csv_created and os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    st.subheader("📄 Rohdaten")
    st.dataframe(df)

    # === Technische Indikatoren ===
    st.subheader("📐 Technische Indikatoren")
    df["ma_8"] = df["close"].rolling(window=8).mean()
    df["ma_14"] = df["close"].rolling(window=14).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["obv"] = np.where(df["close"].diff() > 0, df["volume"], -df["volume"]).cumsum()
    
    # === Chart-Visualisierung ===
    fig = draw_price_chart(df)
    st.pyplot(fig)
    
    
    
else:
    st.info("⬆️ Bitte zuerst auf 'API-Daten abrufen' klicken.")
