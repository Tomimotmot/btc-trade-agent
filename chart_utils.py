import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def draw_price_chart(df: pd.DataFrame, title: str = "BTC Close-Preis"):
    """
    Zeichnet den BTC-Preis über Zeit (Close).
    Erwartet eine Spalte 'datetime' (string/datetime) und 'close' (float).
    """
    df = df.copy()

    # 🧼 Sicherstellen, dass datetime korrekt formatiert ist
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # ❌ Ungültige Zeilen entfernen
    df = df.dropna(subset=["datetime", "close"])

    # ✅ Chart erzeugen
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["datetime"], df["close"], label="Close", color="gray")

    # ✨ Formatierung
    ax.set_title(title)
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Preis (USDT)")
    ax.legend()

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()
    fig.tight_layout()

    return fig