# ml_model.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

class BTCModelTrainer:
    def __init__(self, csv_path="data/btc_bitget_7days.csv", model_path="model/rf_btc_forecast.pkl"):
        self.csv_path = csv_path
        self.model_path = model_path
        self.latest_plot = None

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

        # 📊 Plot
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(y_test.values, label="📈 Echt", color="black")
        ax.plot(y_pred, label="🤖 Prognose", color="orange", linestyle="--")
        ax.set_title("BTC-Kurs in 3h: Echt vs. Vorhersage")
        ax.set_ylabel("BTC Preis (USDT)")
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
            df["ma_8"] = df["close"].rolling(window=8).mean()
            df["ma_14"] = df["close"].rolling(window=14).mean()
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df["rsi_14"] = 100 - (100 / (1 + rs))
            df["obv"] = np.where(df["close"].diff() > 0, df["volume"], -df["volume"]).cumsum()
    
            df = df.dropna()
            X = df[["close", "ma_8", "ma_14", "rsi_14", "obv"]].iloc[-1:]
    
            y_pred = model.predict(X)[0]
            predictions.append(y_pred)
    
            # Neue Zeile anhängen (simulierte nächste Stunde)
            next_row = {"close": y_pred}
            df = pd.concat([df, pd.DataFrame([next_row])], ignore_index=True)

    return predictions