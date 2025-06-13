import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def draw_price_chart(df: pd.DataFrame, title: str = "BTC Close-Preis"):
    df = df.copy()

    # ✅ Zeitspalte erzeugen aus Timestamp
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Plot vorbereiten
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["datetime"], df["close"], label="Close", color="gray")

    ax.set_title(title)
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Preis (USDT)")
    ax.legend()

    # ✅ Zeitachse schön formatieren
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()
    fig.tight_layout()

    return fig