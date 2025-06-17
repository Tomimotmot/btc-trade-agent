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

# 1. API-Daten abrufen
if st.button("📥 API-Daten abrufen und CSV erstellen"):
    try:
        path = fetch_bitget_spot_data_and_save()
        st.session_state.csv_created = True
        st.session_state.csv_path = path  # neuen Pfad merken
        st.success(f"✅ CSV erstellt: {path}")
    except Exception as e:
        st.session_state.csv_created = False
        st.error(f"❌ Fehler: {e}")
        st.stop()

# 2. Datenvorschau (nach API oder bei vorhandenem CSV)
csv_path = st.session_state.get("csv_path", "data/btc_bitget_7days.csv")
trainer = BTCModelTrainer(csv_path=csv_path)

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


# 4. Prognose anzeigen
if st.button("🔮 Nächste 3h prognostizieren"):
    if not os.path.exists(trainer.model_path):
        st.warning("⚠️ Bitte zuerst das Modell trainieren.")
    else:
        # Volle verarbeitete Daten holen (inkl. technischer Indikatoren)
        processed_df = trainer.preview_model_data(return_full=True)
        if processed_df.empty:
            st.error("❌ Keine gültigen Daten für Prognose.")
            st.stop()

        last_df = processed_df.tail(50).copy()
        forecast = trainer.predict_next_3h(last_df)

        current_price = last_df["close"].iloc[-1]
        last_time = last_df["datetime"].iloc[-1]
        future_times = [last_time + pd.Timedelta(hours=i+1) for i in range(3)]

        final_forecast = forecast[-1]
        delta_pct = ((final_forecast - current_price) / current_price) * 100

        delta_color = "green" if delta_pct > 0 else "red"
        delta_arrow = "🔺" if delta_pct > 0 else "🔻"

        # Plot anzeigen
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(last_df["datetime"], last_df["close"], label="Echt", color="gray")
        ax.plot(future_times, forecast, label="Prognose", linestyle="dashed", color="orange")
        ax.set_title("BTC-Kurs: Rückblick & Prognose (nächste 3h)")
        ax.set_ylabel("Preis (USDT)")
        ax.legend()
        st.pyplot(fig)

        # Prognose-Text
        st.markdown(f"""
            <h4>📉 Prognose für in 3 Stunden:</h4>
            <p style='font-size:24px; color:{delta_color};'>
            {delta_arrow} {final_forecast:,.2f} USDT <br>
            ({delta_pct:+.2f}% ggü. aktuell)
            </p>
        """, unsafe_allow_html=True)
