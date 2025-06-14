import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def draw_price_chart(df: pd.DataFrame, title="📈 BTC Close-Preis (1H)"):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["datetime", "close"])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["datetime"], df["close"], color="#008B8B", linewidth=2, label="Close")

    ax.set_title(title, fontsize=14, fontweight="bold", loc="center")
    ax.set_xlabel("Zeit", fontsize=12)
    ax.set_ylabel("Preis (USDT)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    return fig