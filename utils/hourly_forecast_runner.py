import os
import datetime
import pandas as pd
from utils.ml_model import BTCModelTrainer
from main_script import fetch_bitget_spot_data_and_save  # Passe Import ggf. an

# Pfade definieren
csv_path = fetch_bitget_spot_data_and_save()
trainer = BTCModelTrainer(csv_path=csv_path)

# Daten vorbereiten
processed_df = trainer.preview_model_data(return_full=True)
if processed_df.empty:
    print("❌ Keine gültigen Daten für Prognose.")
    exit()

last_df = processed_df.tail(50).copy()
forecast = trainer.predict_next_3h(last_df)

# Forecast-Wert in 3h (also letzter Wert der Prognose)
forecast_value = forecast[-1]
current_price = last_df["close"].iloc[-1]
last_time = pd.to_datetime(last_df["datetime"].iloc[-1])
forecast_time = last_time + pd.Timedelta(hours=3)
diff = forecast_value - current_price

# In Datei schreiben
result_row = {
    "forecast_timestamp": forecast_time.strftime("%Y-%m-%d %H:%M:%S"),
    "forecast_value": forecast_value,
    "actual_value": current_price,
    "difference": diff
}

output_file = "data/hourly_forecast_log.csv"
file_exists = os.path.exists(output_file)
df = pd.DataFrame([result_row])
df.to_csv(output_file, mode='a', index=False, header=not file_exists)
print(f"✅ Prognose geschrieben: {result_row}")
