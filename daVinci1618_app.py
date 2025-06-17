import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests
import datetime
import csv
from utils.chart_utils import draw_price_chart  # falls benötigt
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

# === Hauptaktion: Komplett-Button ===
if st.button("🚀 Komplett-Prognose & Log-Update starten"):
    try:
        # 1. API-Daten abrufen
        csv_path = fetch_bitget_spot_data_and_save()
        trainer = BTCModelTrainer(csv_path=csv_path)

        # 2. Daten laden & prüfen
        processed_df = trainer.preview_model_data(return_full=True)
        if processed_df.empty:
            st.error("❌ Keine gültigen Daten für Prognose.")
            st.stop()

        # 3. Prognose berechnen
        last_df = processed_df.tail(50).copy()
        forecast = trainer.predict_next_3h(last_df)
        current_price = last_df["close"].iloc[-1]
        last_time = pd.to_datetime(last_df["datetime"].iloc[-1])
        future_times = [last_time + pd.Timedelta(hours=i+1) for i in range(3)]
        forecast_time = future_times[-1]
        final_forecast = forecast[-1]
        delta_pct = ((final_forecast - current_price) / current_price) * 100
        delta_color = "green" if delta_pct > 0 else "red"
        delta_arrow = "🔺" if delta_pct > 0 else "🔻"

        # 4. Prognose speichern (nur forecast)
        result_row = {
            "forecast_timestamp": forecast_time.strftime("%Y-%m-%d %H:%M:%S"),
            "forecast_value": final_forecast,
            "actual_value": None,
            "difference": None
        }

        os.makedirs("data", exist_ok=True)
        output_file = "data/hourly_forecast_log.csv"
        file_exists = os.path.exists(output_file)
        pd.DataFrame([result_row]).to_csv(output_file, mode="a", index=False, header=not file_exists)

        # 5. Alte Forecasts mit IST-Werten ergänzen
        log_df = pd.read_csv(output_file)
        updated = False
        for idx, row in log_df.iterrows():
            if pd.isna(row["actual_value"]):
                forecast_ts = pd.to_datetime(row["forecast_timestamp"])
                actual_row = processed_df[processed_df["datetime"] == forecast_ts.strftime("%Y-%m-%d %H:%M:%S")]
                if not actual_row.empty:
                    actual_value = actual_row["close"].values[0]
                    diff = actual_value - row["forecast_value"]
                    log_df.at[idx, "actual_value"] = actual_value
                    log_df.at[idx, "difference"] = diff
                    updated = True
        if updated:
            log_df.to_csv(output_file, index=False)

        # 6. Plot anzeigen
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(last_df["datetime"], last_df["close"], label="Echt", color="gray")
        ax.plot(future_times, forecast, label="Prognose", linestyle="dashed", color="orange")
        ax.set_title("BTC-Kurs: Rückblick & Prognose (nächste 3h)")
        ax.set_ylabel("Preis (USDT)")
        ax.legend()
        st.pyplot(fig)

        # 7. Text anzeigen
        st.markdown(f"""
            <h4>📊 Prognose in 3 Stunden:</h4>
            <p style='font-size:20px;'>
            Aktueller Kurs: <b>{current_price:,.2f} USDT</b><br>
            Prognose: <b style='color:{delta_color};'>{delta_arrow} {final_forecast:,.2f} USDT</b><br>
            Änderung: <span style='color:{delta_color};'>{delta_pct:+.2f}%</span>
            </p>
        """, unsafe_allow_html=True)

        # 8. Log anzeigen
        st.subheader("📊 Prognose-Log (neueste 25 Einträge)")
        st.dataframe(log_df.tail(25))

    except Exception as e:
        st.error(f"❌ Fehler bei Ausführung: {e}")
