import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests
import datetime
import csv
from utils.ml_model import BTCModelTrainer

st.set_page_config(layout="wide", page_title="DaVinci 1.618 CryptoTrader")
st.title("🧠 DaVinci 1.618 CryptoTrader")

# === API-Daten abrufen ===
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
                float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])
            ])
    return path

# === Tabs-Struktur ===
tab1, tab2, tab3 = st.tabs(["📈 Live-Prognose", "📊 Prognose-Log", "⚙️ Modell & Daten"])

with tab1:
    st.header("🔮 BTC-Prognose in 3 Stunden")

    if st.button("🚀 Jetzt Vorhersage starten und Log aktualisieren"):
        try:
            csv_path = fetch_bitget_spot_data_and_save()
            trainer = BTCModelTrainer(csv_path=csv_path)
            processed_df = trainer.preview_model_data(return_full=True)

            if processed_df.empty:
                st.error("❌ Keine gültigen Daten für Prognose.")
                st.stop()

            last_df = processed_df.tail(50).copy()
            forecast = trainer.predict_next_3h(last_df)
            current_price = last_df["close"].iloc[-1]
            last_time = pd.to_datetime(last_df["datetime"].iloc[-1])
            forecast_time = last_time + pd.Timedelta(hours=3)
            final_forecast = forecast[-1]

            # Log-Dateipfad (Hauptverzeichnis, nicht data/)
            log_path = os.path.join(os.getcwd(), "hourly_forecast_log.csv")
            file_exists = os.path.isfile(log_path)

            # Log-Datei laden oder neue leere Tabelle
            if file_exists:
                df_log = pd.read_csv(log_path)
                df_log["forecast_timestamp"] = pd.to_datetime(df_log["forecast_timestamp"], errors="coerce")
            else:
                df_log = pd.DataFrame(columns=["forecast_timestamp", "forecast_value", "actual_value", "difference"])

            # Prognose eintragen (falls noch nicht vorhanden)
            if not (df_log["forecast_timestamp"] == forecast_time).any():
                new_row = {
                    "forecast_timestamp": forecast_time,
                    "forecast_value": final_forecast,
                    "actual_value": None,
                    "difference": None
                }
                df_log = pd.concat([df_log, pd.DataFrame([new_row])], ignore_index=True)
                df_log.to_csv(log_path, index=False)
                st.success(f"✅ Neue Prognose gespeichert für {forecast_time}")
            else:
                st.info(f"ℹ️ Prognose für {forecast_time} existiert bereits.")

            # Alte Prognosen nachtragen (wenn jetzt tatsächlicher Kurs bekannt ist)
            updated = False
            for idx, row in df_log.iterrows():
                if pd.isna(row["actual_value"]):
                    ts = row["forecast_timestamp"]
                    actual_row = processed_df[processed_df["datetime"] == ts.strftime("%Y-%m-%d %H:%M:%S")]
                    if not actual_row.empty:
                        actual_value = actual_row["close"].values[0]
                        diff = actual_value - row["forecast_value"]
                        df_log.at[idx, "actual_value"] = actual_value
                        df_log.at[idx, "difference"] = diff
                        updated = True

            if updated:
                df_log.to_csv(log_path, index=False)
                st.success("🔁 Alte Prognosen mit IST-Werten ergänzt.")

            # Visualisierung
            delta_pct = ((final_forecast - current_price) / current_price) * 100
            future_times = [last_time + pd.Timedelta(hours=i + 1) for i in range(3)]

            col1, col2, col3 = st.columns(3)
            col1.metric("Aktueller BTC-Kurs", f"{current_price:,.2f} USDT")
            col2.metric("Prognose in 3h", f"{final_forecast:,.2f} USDT", f"{delta_pct:+.2f} %")
            col3.metric("Δ absolut", f"{final_forecast - current_price:+.2f} USDT")

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(last_df["datetime"], last_df["close"], label="Echt", color="gray")
            ax.plot(future_times, forecast, label="Prognose", linestyle="dashed", color="orange")
            ax.set_title("BTC-Kurs: Rückblick & Prognose (nächste 3h)")
            ax.set_ylabel("Preis (USDT)")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Fehler: {e}")

with tab2:
    st.header("📊 Prognose-Log")
    log_path = os.path.join(os.getcwd(), "hourly_forecast_log.csv")
    if os.path.exists(log_path):
        df_log = pd.read_csv(log_path)
        if not df_log.empty:
            df_log["abs_diff"] = df_log["difference"].abs()
            df_clean = df_log.dropna().sort_values("forecast_timestamp", ascending=False)
            st.dataframe(df_clean[["forecast_timestamp", "forecast_value", "actual_value", "difference"]].head(25))
        else:
            st.info("⚠️ Logdatei vorhanden, aber leer.")
    else:
        st.info("📝 Noch keine Prognosen gespeichert.")

with tab3:
    st.header("⚙️ Daten & Modell")
    csv_path = "data/btc_bitget_7days.csv"
    if os.path.exists(csv_path):
        trainer = BTCModelTrainer(csv_path=csv_path)

        if st.button("📊 Vorschau auf Trainingsdaten"):
            df_preview = trainer.preview_model_data()
            if not df_preview.empty:
                st.dataframe(df_preview.head(50))
            else:
                st.warning("Keine Trainingsdaten gefunden.")

        if st.button("🧠 Modell trainieren"):
            model_path, info, fig = trainer.train_model()
            st.success(info)
            if fig:
                st.pyplot(fig)
    else:
        st.warning("Bitte zuerst eine Prognose starten.")
