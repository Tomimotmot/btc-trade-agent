import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def draw_price_chart(df: pd.DataFrame, title: str = "BTC Close-Preis"):
    df = df.copy()

    # ✅ timestamp absichern
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"])

    # ✅ datetime erzeugen (aus timestamp in ms)
    df["datetime"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms")

    # ✅ nochmal absichern, dass alle Werte da sind
    df = df.dropna(subset=["datetime", "close"])

    # ✅ Preis in float konvertieren
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["datetime"], df["close"], label="Close", color="gray")

    ax.set_title(title)
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Preis (USDT)")
    ax.legend()

    # ✅ Zeitachse formatieren
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()
    fig.tight_layout()

    return fig