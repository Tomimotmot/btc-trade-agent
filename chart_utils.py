import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import pandas as pd

def draw_price_chart(df: pd.DataFrame, title="📈 BTC Close + MA (1H)"):

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    
    # MA berechnen falls nötig
    if "ma_8" not in df.columns:
        df["ma_8"] = df["close"].rolling(window=8).mean()
    if "ma_14" not in df.columns:
        df["ma_14"] = df["close"].rolling(window=14).mean()

    df = df.dropna(subset=["datetime", "close", "ma_8", "ma_14"])

    # Bereich ermitteln
    price_min = df["close"].min()
    price_max = df["close"].max()
    step = (price_max - price_min) / 3
    guide_levels = [price_min, price_min + step, price_min + 2 * step, price_max]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))

    # Linien plotten
    ax.plot(df["datetime"], df["close"], label="Close", color="#00cc99", linewidth=2)
    ax.plot(df["datetime"], df["ma_8"], label="MA 8", color="red", linestyle="--", linewidth=1.2)
    ax.plot(df["datetime"], df["ma_14"], label="MA 14", color="blue", linestyle="--", linewidth=1.2)

    # Horizontale Linien
    for level in guide_levels:
        ax.axhline(y=level, color="#cccccc", linestyle="--", linewidth=0.8)
        ax.text(df["datetime"].iloc[0], level, f"{level:,.2f}", color="#666666", va="bottom", fontsize=8)

    # Achsen, Layout
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Preis (USDT)")
    ax.grid(False)
    ax.legend()

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    return fig