# data/fetch_bitget.py

import requests
import time
import datetime
import csv
import os

def save_to_csv(candles, filename="btc_bitget_7days.csv"):
    import os
    import csv
    import datetime

    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", filename)

    with open(path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "datetime", "open", "high", "low", "close", "volume"])

        for c in candles:
            ts = int(c[0])
            dt = datetime.datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M:%S")
            open_ = float(c[1])
            high = float(c[2])
            low = float(c[3])
            close = float(c[4])
            volume = float(c[5])

            writer.writerow([ts, dt, open_, high, low, close, volume])

    print(f"✅ CSV gespeichert unter: {path}")