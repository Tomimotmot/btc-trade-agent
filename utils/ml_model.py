import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

class BTCModelTrainer:
    def __init__(self, csv_path="data/btc_bitget_7days.csv", model_path="model/rf_btc_forecast.pkl"):
        self.csv_path = csv_path
        self.model_path = model_path
        self.latest_plot = None


    def preview_model_data(self, return_full=False):
        if not os.path.exists(self.csv_path):
            st.error(f"❌ CSV-Datei nicht gefunden: {self.csv_path}")
            return pd.DataFrame()
    
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            st.error(f"❌ Fehler beim Einlesen der CSV: {e}")
            return pd.DataFrame()

        # 📊 Zeitachse aus echten Daten übernehmen
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        elif "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        else:
            df["datetime"] = pd.date_range(start="2025-01-01 00:00", periods=len(df), freq="1H")
        
        time_axis = df["datetime"].iloc[split_idx:]

    
        # Feature Engineering
        df["ma_8"] = df["close"].rolling(window=8).mean()
        df["ma_14"] = df["close"].rolling(window=14).mean()
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["obv"] = np.where(df["close"].diff() > 0, df["volume"], -df["volume"]).cumsum()
        df["future_close"] = df["close"].shift(-3)
    
        # Spaltenauswahl für Vorschau
        columns_to_show = [
            "datetime", "open", "high", "low", "close",
            "ma_8", "ma_14", "rsi_14", "obv", "future_close"
        ]
    
        df = df.dropna(subset=columns_to_show)
    
        if return_full:
            return df  # komplette verarbeitete Tabelle zurückgeben
    
        preview = df[columns_to_show].head(20)
        return preview


    def train_model(self):
        if not os.path.exists(self.csv_path):
            return None, "❌ CSV-Datei nicht gefunden.", None

        df = pd.read_csv(self.csv_path)

        # === Feature Engineering ===
        df["ma_8"] = df["close"].rolling(window=8).mean()
        df["ma_14"] = df["close"].rolling(window=14).mean()
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["obv"] = np.where(df["close"].diff() > 0, df["volume"], -df["volume"]).cumsum()
        df["future_close"] = df["close"].shift(-3)  # Ziel: Preis in 3h

        df = df.dropna(subset=["ma_8", "ma_14", "rsi_14", "obv", "future_close"])

        features = ["close", "ma_8", "ma_14", "rsi_14", "obv"]
        X = df[features]
        y = df["future_close"]

        # Split in Training und Test
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Modell trainieren
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Vorhersage und Fehlerberechnung
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mae_pct = (mae / y_test.mean()) * 100

        # Modell speichern
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)

        # 📊 Plot: Vorhersage vs. Wahrheit mit Zeitachse (1h-Takt)
        num_points = len(y_test)
        start_time = pd.Timestamp("2025-01-01 00:00")  # ➤ Optional anpassen
        time_axis = pd.date_range(start=start_time, periods=num_points, freq="1H")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(time_axis, y_test.values, label="📈 Echt", color="black")
        ax.plot(time_axis, y_pred, label="🤖 Prognose", color="orange", linestyle="--")
        ax.set_title("BTC-Kurs in 3h: Echt vs. Vorhersage")
        ax.set_ylabel("BTC Preis (USDT)")
        ax.set_xlabel("Zeit (1h-Abstand)")
        ax.legend()
        plt.tight_layout()
        self.latest_plot = fig

        return (
            self.model_path,
            f"✅ Modell gespeichert: {self.model_path} — 📉 MAE: {mae:.2f} USDT ({mae_pct:.2f} %)",
            self.latest_plot
        )

    def predict_next_3h(self, df_recent):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Modell nicht gefunden. Bitte zuerst trainieren.")

        model = joblib.load(self.model_path)
        df = df_recent.copy()
        predictions = []

        for i in range(3):
            # Technische Indikatoren berechnen
            df["ma_8"] = df["close"].rolling(window=8).mean()
            df["ma_14"] = df["close"].rolling(window=14).mean()
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df["rsi_14"] = 100 - (100 / (1 + rs))
            df["obv"] = np.where(df["close"].diff() > 0, df["volume"], -df["volume"]).cumsum()

            # Drop alle unvollständigen Zeilen
            df = df.dropna(subset=["close", "ma_8", "ma_14", "rsi_14", "obv", "volume"])

            if len(df) == 0:
                raise ValueError("Nicht genug Daten für gültige Vorhersage.")

            last_features = df[["close", "ma_8", "ma_14", "rsi_14", "obv"]].iloc[-1:].copy()

            # Vorhersage
            try:
                y_pred = model.predict(last_features)[0]
            except Exception as e:
                raise ValueError(f"Fehler bei Modellvorhersage: {e}")

            predictions.append(y_pred)

            # Dummy-Volume annehmen für OBV-Fortschreibung
            df = pd.concat([
                df,
                pd.DataFrame([{
                    "close": y_pred,
                    "volume": df["volume"].iloc[-1]  # letzten Wert beibehalten
                }])
            ], ignore_index=True)

        return predictions
