import matplotlib.pyplot as plt
import pandas as pd

def draw_ma_chart(df: pd.DataFrame, title: str = "BTC Preis + MA 8/14"):
    """
    Erstellt ein Linien-Diagramm mit Close-Preis, MA 8 und MA 14.
    
    Argumente:
    - df: Pandas DataFrame mit Spalten 'datetime', 'close'
    - title: Titel des Charts

    Rückgabe:
    - Matplotlib-Figure-Objekt zur Verwendung in st.pyplot()
    """
    # Datumsformat absichern
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Moving Averages berechnen, falls nicht vorhanden
    if "ma_8" not in df.columns:
        df["ma_8"] = df["close"].rolling(window=8).mean()
    if "ma_14" not in df.columns:
        df["ma_14"] = df["close"].rolling(window=14).mean()

    # NaNs entfernen für saubere Darstellung
    df = df.dropna(subset=["ma_8", "ma_14", "close", "datetime"])

    # Plot erzeugen
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["datetime"], df["close"], label="Close", color="gray")
    ax.plot(df["datetime"], df["ma_8"], label="MA 8", color="red")
    ax.plot(df["datetime"], df["ma_14"], label="MA 14", color="blue")

    ax.set_title(title)
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Preis (USDT)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    
    return fig