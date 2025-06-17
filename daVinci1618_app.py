import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests
import time
import datetime
import csv
import numpy as np
from utils.chart_utils import draw_price_chart
from utils.ml_model import BTCModelTrainer

st.title("DaVinci 1.618 CryptoTrader")

# 1. API-Daten abrufen
if st.button("📡 Aktuelle Daten von Bitget laden"):
    try:
        path = fetch_bitget_spot_data_and_save()
        st.success(f"✅ Daten gespeichert unter: {path}")
    except Exception as e:
        st.error(f"❌ Fehler beim Abrufen der Daten: {e}")

# 2. Datenvorschau (nach API oder bei vorhandenem CSV)
trainer = BTCModelTrainer(csv_path=path)
if st.button("🔍 Vorschau auf Trainingsdaten"):
    preview = trainer.preview_model_data()
    if not preview.empty:
        st.dataframe(preview)
    else:
        st.warning("⚠️ Keine Daten zum Anzeigen.")

# 3. Modell trainieren
if st.button("🤖 Modell trainieren"):
    model_path, info, fig = trainer.train_model()
    st.success(info)
    if fig:
        st.pyplot(fig)



# === API-Funktion ===
def fetch_bitget_spot_data_and_save(symbol="BTCUSDT", granularity="1h", filename="btc_bitget_7days.csv"):
    url = "https://api.bitget.com/api/v2/spot/market/candles"
    params = {"symbol": symbol, "granularity": granularity, "limit": "168"}

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
            writer.writerow([
                ts, dt,
                float(c[1]),  # open
                float(c[2]),  # high
                float(c[3]),  # low
                float(c[4]),  # close
                float(c[5])   # volume
            ])
    return path

trainer = BTCModelTrainer(csv_path=csv_path)
