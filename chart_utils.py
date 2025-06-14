import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def draw_price_chart(df: pd.DataFrame, title="📈 BTC Close + MA8/MA14 (1H)"):

    # Absicherung
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    
    # Falls MA noch nicht berechnet wurde:
    if "ma_8" not in df.columns:
        df["ma_8"] = df["close"].rolling(window=8).mean()
    if "ma_14" not in df.columns:
        df["ma_14"] = df["close"].rolling(window=14).mean()

    df = df.dropna(subset=["datetime", "close", "ma_8", "ma_14"])

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df["datetime"], df["close"], label="Close", color="#444", linewidth=2)
    ax.plot(df["datetime"], df["ma_8"], label="MA 8", color="red", linestyle="--", linewidth=1.5)
    ax.plot(df["datetime"], df["ma_14"], label="MA 14", color="blue", linestyle="--", linewidth=1.5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Preis (USDT)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    return fig