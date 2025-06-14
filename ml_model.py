# ml_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os

class BTCModelTrainer:
    def __init__(self, csv_path="data/btc_bitget_7days.csv", model_path="model/rf_btc_forecast.pkl"):
        self.csv_path = csv_path
        self.model_path = model_path

    def train_model(self):
        if not os.path.exists(self.csv_path):
            return None, "❌ CSV-Datei nicht gefunden."

        df = pd.read_csv(self.csv_path)

        # Features berechnen
        df["ma_8"] = df["close"].rolling(window=8).mean()
        df["ma_14"] = df["close"].rolling(window=14).mean()
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["obv"] = np.where(df["close"].diff() > 0, df["volume"], -df["volume"]).cumsum()
        df["future_close"] = df["close"].shift(-3)

        df = df.dropna(subset=["ma_8", "ma_14", "rsi_14", "obv", "future_close"])
        features = ["close", "ma_8", "ma_14", "rsi_14", "obv"]
        X = df[features]
        y = df["future_close"]

        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)

        return self.model_path, f"✅ Modell gespeichert: {self.model_path} — 📉 MAE: {mae:.2f} USDT"