"""
Elite Algo Signal -> Telegram Alert (one-shot version for GitHub Actions)
--------------------------------------------------------------------------
Runs ONCE, checks the last fully closed 4h candle, sends a Telegram
message if a new buy/sell signal appeared, then exits.
Meant to be triggered on a schedule by GitHub Actions (see the
workflow file .github/workflows/elite_algo_alert.yml).

State (the timestamp of the last candle already alerted on) is stored
in last_signal_state.txt and committed back to the repo by the
workflow, so signals are never sent twice.

Secrets (BOT_TOKEN, CHAT_ID) are read from environment variables -
set them as GitHub repo secrets, never hardcode them in this file.
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
import ccxt

# =========================== CONFIG ===========================

BOT_TOKEN = os.environ.get("TG_BOT_TOKEN")
CHAT_ID = os.environ.get("TG_CHAT_ID")

EXCHANGE_ID = "kucoin"   # binance blocks US-based cloud IPs (incl. GitHub
                         # Actions runners) with a "restricted location"
                         # error; kucoin/okx/bybit do not have this issue
SYMBOL = os.environ.get("SYMBOL", "BTC/USDT")
TIMEFRAME = "4h"              # tuned per your request: fewer, stronger signals

# --- Supertrend settings ---
ATR_LEN = 10
SENSITIVITY = 3.5              # slightly higher on 4h to reduce noise
                                # (original indicator auto-adjusts 2.85-4
                                # based on historical volatility)

# --- Optional filters (mirrors the indicator's toggles) ---
USE_SMART_FILTER = True        # close vs EMA200 alignment -> stronger signal
USE_ADX_FILTER = True          # ADX(14) > 20 -> only trending conditions
USE_VOLUME_FILTER = False      # rising volume (EMA25 > EMA26 of volume)

STATE_FILE = os.path.join(os.path.dirname(__file__), "last_signal_state.txt")

# ================================================================


def send_telegram(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("[telegram error] TG_BOT_TOKEN / TG_CHAT_ID not set")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"[telegram error] {e}")


def fetch_ohlcv(exchange, symbol, timeframe, limit=500):
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df


def atr(df, length):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def supertrend(df, length, factor):
    hl2 = (df["high"] + df["low"]) / 2
    _atr = atr(df, length)
    upper = hl2 + factor * _atr
    lower = hl2 - factor * _atr

    final_upper = upper.copy()
    final_lower = lower.copy()
    trend = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        final_upper.iloc[i] = upper.iloc[i] if (
            upper.iloc[i] < final_upper.iloc[i - 1] or df["close"].iloc[i - 1] > final_upper.iloc[i - 1]
        ) else final_upper.iloc[i - 1]

        final_lower.iloc[i] = lower.iloc[i] if (
            lower.iloc[i] > final_lower.iloc[i - 1] or df["close"].iloc[i - 1] < final_lower.iloc[i - 1]
        ) else final_lower.iloc[i - 1]

        if trend.iloc[i - 1] == 1:
            trend.iloc[i] = -1 if df["close"].iloc[i] < final_lower.iloc[i] else 1
        else:
            trend.iloc[i] = 1 if df["close"].iloc[i] > final_upper.iloc[i] else -1

    st_line = np.where(trend == 1, final_lower, final_upper)
    return pd.Series(st_line, index=df.index), trend


def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def adx(df, length=14):
    high, low = df["high"], df["low"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) < 0] = 0
    minus_dm[(minus_dm - plus_dm) < 0] = 0

    _atr = atr(df, length)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / length, adjust=False).mean() / _atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / length, adjust=False).mean() / _atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / length, adjust=False).mean().fillna(0)


def compute_signals(df):
    df = df.copy()
    df["st"], df["trend"] = supertrend(df, ATR_LEN, SENSITIVITY)
    df["ema200"] = ema(df["close"], 200)
    df["adx"] = adx(df, 14)
    df["vol_ema25"] = ema(df["volume"], 25)
    df["vol_ema26"] = ema(df["volume"], 26)

    cross_up = (df["close"] > df["st"]) & (df["close"].shift(1) <= df["st"].shift(1))
    cross_down = (df["close"] < df["st"]) & (df["close"].shift(1) >= df["st"].shift(1))

    bull = cross_up.copy()
    bear = cross_down.copy()

    if USE_SMART_FILTER:
        bull &= df["close"] > df["ema200"]
        bear &= df["close"] < df["ema200"]
    if USE_ADX_FILTER:
        bull &= df["adx"] > 20
        bear &= df["adx"] > 20
    if USE_VOLUME_FILTER:
        vol_rising = (df["vol_ema25"] - df["vol_ema26"]) / df["vol_ema26"] > 0
        bull &= vol_rising
        bear &= vol_rising

    df["bull"] = bull
    df["bear"] = bear
    return df


def load_last_seen():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return f.read().strip()
    return ""


def save_last_seen(ts_str):
    with open(STATE_FILE, "w") as f:
        f.write(ts_str)


def main():
    exchange = getattr(ccxt, EXCHANGE_ID)()
    df = fetch_ohlcv(exchange, SYMBOL, TIMEFRAME)
    df = compute_signals(df)

    # Only the last FULLY CLOSED 4h candle - never the live/forming one
    closed = df.iloc[:-1]
    last_row = closed.iloc[-1]
    ts_str = str(last_row["time"])

    last_seen = load_last_seen()
    if ts_str == last_seen:
        print(f"No new closed candle since last run ({ts_str}). Nothing to do.")
        return

    if last_row["bull"]:
        msg = (f"BUY signal (strong: EMA200 + ADX filters passed)\n"
               f"{SYMBOL} [{TIMEFRAME}]\n"
               f"Close: {last_row['close']:.4f}\n"
               f"ADX: {last_row['adx']:.1f}\n"
               f"Candle time (UTC): {ts_str}")
        print(msg)
        send_telegram(msg)
    elif last_row["bear"]:
        msg = (f"SELL signal (strong: EMA200 + ADX filters passed)\n"
               f"{SYMBOL} [{TIMEFRAME}]\n"
               f"Close: {last_row['close']:.4f}\n"
               f"ADX: {last_row['adx']:.1f}\n"
               f"Candle time (UTC): {ts_str}")
        print(msg)
        send_telegram(msg)
    else:
        print(f"Closed candle {ts_str} checked, no signal.")

    # Always advance the state to the latest closed candle, even with no
    # signal, so we never re-scan the same candle twice.
    save_last_seen(ts_str)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[fatal error] {e}")
        sys.exit(1)
