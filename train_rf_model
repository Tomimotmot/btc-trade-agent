import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os

# === Pfad zur CSV-Datei ===
csv_path = "data/btc_bitget_7days.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"🚫 Datei nicht gefunden: {csv_path}")

# === Daten einlesen ===
df = pd.read_csv(csv_path)

# === Technische Indikatoren berechnen ===
df["ma_8"] = df["close"].rolling(window=8).mean()
df["ma_14"] = df["close"].rolling(window=14).mean()

delta = df["close"].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
rs = gain / loss
df["rsi_14"] = 100 - (100 / (1 + rs))

df["obv"] = np.where(df["close"].diff() > 0, df["volume"], -df["volume"]).cumsum()

# === Zielwert: Preis in 3h ===
df["future_close"] = df["close"].shift(-3)

# === Nur vollständige Zeilen verwenden ===
df = df.dropna(subset=["ma_8", "ma_14", "rsi_14", "obv", "future_close"])

# === Features und Ziel definieren ===
features = ["close", "ma_8", "ma_14", "rsi_14", "obv"]
X = df[features]
y = df["future_close"]

# === Zeitbasierter Trainings-/Testsplit (80/20) ===
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# === Modell trainieren ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Performance evaluieren ===
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# === Modell speichern ===
os.makedirs("model", exist_ok=True)
model_path = "model/rf_btc_forecast.pkl"
joblib.dump(model, model_path)

# === Ausgabe ===
print(f"✅ Modell gespeichert unter: {model_path}")
print(f"📉 Mean Absolute Error: {mae:.2f} USDT")