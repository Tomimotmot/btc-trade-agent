import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def draw_price_chart(df: pd.DataFrame, title: str = "BTC Close-Preis"):
    """
    Zeichnet den Close-Preis über Zeit auf Basis von Timestamps in Millisekunden.
    
    Anforderungen:
    - Spalten 'timestamp' (ms) und 'close'
    
    Rückgabe:
    - Matplotlib-Figure
    """
    df = df.copy()

    # ✅ Timestamp erzwingen (als Zahl)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    # ❌ Zeilen mit ungültigen Daten verwerfen
    df = df.dropna(subset=["timestamp", "close"])

    # ✅ Zeitspalte aus Unix-Timestamps (ms) erzeugen
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["datetime"], df["close"], label="Close", color="gray")

    ax.set_title(title)
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Preis (USDT)")
    ax.legend()

    # ✅ Zeitachse lesbar formatieren
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()
    fig.tight_layout()

    return fig