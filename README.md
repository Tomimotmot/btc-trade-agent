# # 🤖 bitcandle-agent

Ein modularer Trading-Agent für Bitcoin, basierend auf stündlichen Candlestick-Daten von der Bitget API.

> Ziel: Mithilfe von **Random Forest Regression** optimale Einstiegs- und Ausstiegspunkte erkennen.

---

## 🎯 Ziel des Projekts

Das langfristige Ziel ist die Entwicklung eines datengetriebenen Handelssystems, das:

- 📈 historische Bitcoin-Kerzen analysiert
- 🧠 mit **Random Forest Regressor** den zukünftigen Preis vorhersagt
- ⏱ gezielt Einstiegs- und Ausstiegssignale erzeugt
- 💡 auf einfache technische Features wie gleitende Durchschnitte, Volatilität oder Candle-Patterns trainiert wird

---

## 🚀 Features (In Plannung)

- 🔄 Holt automatisch die letzten 7 Tage an BTC/USDT-Kerzen (1h) via Bitget API
- 💾 Speichert Daten als CSV (`btc_bitget_7days.csv`)
- 🧱 Struktur für Erweiterung mit ML-Modellen vorbereitet
- 📂 Modularer Aufbau (Daten, Modell, Backtest)