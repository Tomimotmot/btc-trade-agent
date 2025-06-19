import ccxt
import pandas as pd
import ta
import time
import schedule
import os
import csv
from datetime import datetime

# Konfiguration
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
LOOKBACK_CANDLES = 200
STOP_LOSS_PCT = 0.02
LEVERAGE = 2
TRADE_AMOUNT_USD = 100
LOGFILE = "trade_log.csv"

# Binance API initialisieren
exchange = ccxt.binance()

# 📝 Logging-Funktion
def log_event(action, price=None, rsi=None, bb_lower=None, bb_upper=None, comment=""):
    timestamp = datetime.now()
    header = ['timestamp', 'action', 'price', 'rsi', 'bb_lower', 'bb_upper', 'comment']
    row = [timestamp, action, price, rsi, bb_lower, bb_upper, comment]
    file_exists = os.path.isfile(LOGFILE)
    
    with open(LOGFILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)
    
    print(f"[{timestamp}] LOGGED: {action.upper()} - {comment}")

# 📥 Daten abrufen
def load_data():
    try:
        print(f"[{datetime.now()}] 📥 Lade Daten von Binance...")
        ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=LOOKBACK_CANDLES)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        log_event("error", comment=f"Datenabruf fehlgeschlagen: {e}")
        return pd.DataFrame()

# 📈 Indikatoren berechnen
def add_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    bb = ta.volatility.BollingerBands(close=df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    return df

# ✅ Regel: Buy-Signal
def check_buy_signal(row):
    return row['close'] < row['bb_lower'] and row['rsi'] < 30

# ✅ Regel: Sell-Signal
def check_sell_signal(row):
    return row['close'] > row['bb_upper'] and row['rsi'] > 70

# 🛒 Trade ausführen
def execute_trade(action, price, row):
    if action == "buy":
        stop_loss = price * (1 - STOP_LOSS_PCT)
        exposure = TRADE_AMOUNT_USD * LEVERAGE
        comment = f"Kauf mit Hebel {LEVERAGE}x, SL {stop_loss:.2f}, Volumen {exposure:.2f} USDT"
    elif action == "sell":
        comment = "Verkaufssignal erkannt – Full Exit (simuliert)"
    else:
        comment = "Unbekannte Aktion"
    
    log_event(
        action=action,
        price=price,
        rsi=row['rsi'],
        bb_lower=row['bb_lower'],
        bb_upper=row['bb_upper'],
        comment=comment
    )

# 🔄 Strategie ausführen
def strategy_loop():
    log_event("check", comment="Starte neuen Strategie-Check")
    df = load_data()
    if df.empty:
        return
    
    df = add_indicators(df)
    last_row = df.iloc[-1]

    if check_buy_signal(last_row):
        execute_trade("buy", last_row['close'], last_row)
    elif check_sell_signal(last_row):
        execute_trade("sell", last_row['close'], last_row)
    else:
        log_event("no_signal", price=last_row['close'], rsi=last_row['rsi'],
                  bb_lower=last_row['bb_lower'], bb_upper=last_row['bb_upper'],
                  comment="Kein Signal")

# 📆 Cronjob: alle 5 Minuten
schedule.every(5).minutes.do(strategy_loop)

# 🚀 Start
if __name__ == "__main__":
    print("🚀 Trading-Bot gestartet – prüfe alle 5 Minuten auf Signale...\n")
    log_event("start", comment="Bot gestartet")
    strategy_loop()  # Initial sofort ausführen
    while True:
        schedule.run_pending()
        time.sleep(1)