# main.py

from data.fetch_bitget import get_bitget_historical_data, save_to_csv
# from features.feature_engineering import add_technical_indicators
# from agent.model_forest import train_model, predict_signals

import pandas as pd

def main():
    print("📡 Starte Datenabruf von Bitget …")
    candles = get_bitget_historical_data()
    save_to_csv(candles)

    print("📂 Lade CSV-Datei …")
    df = pd.read_csv("btc_bitget_7days.csv")

    print("🔧 TODO: Technische Features berechnen …")
    # df = add_technical_indicators(df)

    print("🧠 TODO: Modell trainieren oder laden …")
    # model = train_model(df)

    print("📈 TODO: Einstieg/Ausstieg vorhersagen …")
    # signals = predict_signals(model, df)

    print("✅ Ablauf abgeschlossen.")

if __name__ == "__main__":
    main()