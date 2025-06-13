import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def draw_ma_chart(df: pd.DataFrame, title: str = "BTC Preis + MA 8/14"):
    df = df.copy()
    
    # 💡 Korrekte Zeitumwandlung
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Berechne MA nur, wenn nicht vorhanden
    if "ma_8" not in df.columns:
        df["ma_8"] = df["close"].rolling(window=8).mean()
    if "ma_14" not in df.columns:
        df["ma_14"] = df["close"].rolling(window=14).mean()

    # Nur gültige Zeilen
    df = df.dropna(subset=["close", "ma_8", "ma_14", "datetime"])

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["datetime"], df["close"], label="Close", color="gray")
    ax.plot(df["datetime"], df["ma_8"], label="MA 8", color="red")
    ax.plot(df["datetime"], df["ma_14"], label="MA 14", color="blue")

    ax.set_title(title)
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Preis (USDT)")
    ax.legend()

    # ✅ Zeitachse formatieren
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()
    fig.tight_layout()
    
    return fig