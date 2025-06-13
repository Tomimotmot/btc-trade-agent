import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os

# === Einstellungen ===
st.set_page_config(page_title="daVinci1617 – BTC Signal Shop", layout="wide")

# === CSV laden ===
csv_path = "data/btc_bitget_7days.csv"
if not os.path.exists(csv_path):
    st.error("❌ Daten nicht gefunden. Bitte zuerst `main.py` ausführen.")
    st.stop()

df = pd.read_csv(csv_path)
df["datetime"] = pd.to_datetime(df["datetime"])
latest_price = df["close"].iloc[-1]
avg_price = df["close"].mean()

# === Titel & Intro ===
st.title("🧠 daVinci1617 – Bitcoin Signal Shop")
st.markdown("Willkommen im experimentellen BTC-Signaldashboard. Die Daten stammen aus der Bitget API (1h-Candles, 7 Tage).")

# === Chartbereich ===
st.subheader("📊 BTC/USDT Preisverlauf")
fig, ax = plt.subplots()
ax.plot(df["datetime"], df["close"], label="Close Price", color="blue")
ax.axhline(avg_price, color="gray", linestyle="--", label="⌀ Durchschnitt")
ax.set_xlabel("Zeit")
ax.set_ylabel("USD")
ax.legend()
st.pyplot(fig)

# === Aktuelles Signal (Platzhalter) ===
st.subheader("📌 Aktuelle Einschätzung")
signal = "🟢 BUY – unter Durchschnitt" if latest_price < avg_price else "🔴 HOLD – über Durchschnitt"
st.metric(label="Aktueller BTC/USDT-Kurs", value=f"${latest_price:.2f}")
st.success(f"Signal: {signal}")

# === Strategiewahl (simuliert) ===
st.subheader("🧪 Strategie-Auswahl")
strategies = {
    "Random Forest Regressor": "Basierend auf historischen Preis-Mustern und Volumen",
    "SMA-Kreuzung": "Signal bei SMA(20) über SMA(50)",
    "RSI-Basierend": "Kauft bei RSI < 30, verkauft bei RSI > 70"
}
selected = st.selectbox("🧠 Wähle deine Strategie", list(strategies.keys()))
st.markdown(f"📝 Beschreibung: {strategies[selected]}")

# === Shop-Button (Mock) ===
st.subheader("🚀 Aktion")
if st.button("📤 Exportiere diese Daten"):
    df.to_csv("data/exported_btc_data.csv", index=False)
    st.success("📁 CSV wurde exportiert nach `data/exported_btc_data.csv`")

st.markdown("---")
st.caption("© daVinci1617 | AI meets Trading | Bitget Data Feed")