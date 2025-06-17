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

    def _add_features(self, df):
        df = df.copy()
        df["ma_8"] = df["close"].rolling(window=8).mean()
        df["ma_14"] = df["close"].rolling(window=14).mean()
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["obv"] = np.where(df["close"].diff() > 0, df["volume"], -df["volume"]).cumsum()
        return df

    def _load_and_process_data(self):
        if not os.path.exists(self.csv_path):
            st.error(f"❌ CSV-Datei nicht gefunden: {self.csv_path}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            st.error(f"❌ Fehler beim Einlesen der CSV: {e}")
            return pd.DataFrame()

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        elif "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        else:
            df["datetime"] = pd.date_range(start="2025-01-01 00:00", periods=len(df), freq="1H")

        df = self._add_features(df)
        df["future_close"] = df["close"].shift(-3)

        df = df.dropna(subset=[
            "datetime", "open", "high", "low", "close", "volume",
            "ma_8", "ma_14", "rsi_14", "obv", "future_close"
        ])
        return df

    def preview_model_data(self, return_full=False):
        df = self._load_and_process_data()
        if df.empty:
            return pd.DataFrame()

        cols = [
            "datetime", "open", "high", "low", "close",
            "ma_8", "ma_14", "rsi_14", "obv", "future_close"
        ]
        return df if return_full else df[cols].head(20)

    def train_model(self):
        df = self._load_and_process_data()
        if df.empty:
            return None, "❌ Keine Daten vorhanden.", None

        features = ["close", "ma_8", "ma_14", "rsi_14", "obv"]
        X = df[features]
        y = df["future_close"]
        time_axis = df["datetime"]

        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        time_test = time_axis.iloc[split_idx:]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mae_pct = (mae / y_test.mean()) * 100

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(time_test, y_test.values, label="📈 Echt", color="black")
        ax.plot(time_test, y_pred, label="🤖 Prognose", color="orange", linestyle="--")
        ax.set_title("BTC-Kurs in 3h: Echt vs. Vorhersage")
        ax.set_ylabel("BTC Preis (USDT)")
        ax.set_xlabel("Zeit (echte Werte aus API)")
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

        if len(df) < 30:
            raise ValueError("Mindestens 30 Zeilen werden empfohlen für stabile Vorhersage.")

        predictions = []
        for _ in range(3):
            df = self._add_features(df)
            df = df.dropna(subset=["close", "ma_8", "ma_14", "rsi_14", "obv", "volume"])

            if df.empty:
                raise ValueError("Nicht genug gültige Daten für Vorhersage.")

            last_features = df[["close", "ma_8", "ma_14", "rsi_14", "obv"]].iloc[-1:]

            try:
                y_pred = model.predict(last_features)[0]
            except Exception as e:
                raise ValueError(f"Fehler bei Modellvorhersage: {e}")

            predictions.append(y_pred)

            new_row = {
                "close": y_pred,
                "volume": df["volume"].iloc[-1],
                "ma_8": np.nan,
                "ma_14": np.nan,
                "rsi_14": np.nan,
                "obv": np.nan
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        return predictions
