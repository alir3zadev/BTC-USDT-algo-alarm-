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
TIMEFRAME = os.environ.get("TIMEFRAME", "4h")   # e.g. "15m","1h","4h","1d"
                                                 # set via GitHub Actions env,
                                                 # no need to edit this file

# --- Supertrend settings ---
ATR_LEN = 10
# Sensitivity is now computed automatically per-bar (real EWMA-based
# historical-volatility bands, ported directly from the Pine Script),
# matching the indicator's default "Auto Sensitivity: On" behavior.
# No fixed value needed anymore.

# --- Optional filters (mirrors the indicator's toggles) ---
# Defaults match the ORIGINAL indicator's own defaults exactly
# (smartSignalsOnly=false, consSignalsFilter=false) so signals match
# what you see on the chart with default settings. Set to "true" via
# GitHub Actions env if you want the extra (stricter) filtering back.
USE_SMART_FILTER = os.environ.get("USE_SMART_FILTER", "false").lower() == "true"
USE_ADX_FILTER = os.environ.get("USE_ADX_FILTER", "false").lower() == "true"
USE_VOLUME_FILTER = os.environ.get("USE_VOLUME_FILTER", "false").lower() == "true"

_state_suffix = f"{SYMBOL.replace('/', '-')}_{TIMEFRAME}"
STATE_FILE = os.path.join(os.path.dirname(__file__), f"last_signal_state_{_state_suffix}.txt")

# ================================================================


def send_telegram(text: str) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        print("[telegram error] TG_BOT_TOKEN / TG_CHAT_ID not set")
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=15)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"[telegram error] {e}")
        return False


def fetch_ohlcv(exchange, symbol, timeframe, limit=500):
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")

    # --- Data sanity check ---
    # A broken/blocked/stale exchange feed won't always raise an error -
    # it can silently return too few rows, NaNs, or a flat price series.
    # Pandas comparisons against NaN quietly evaluate to False, which
    # would make the script always report "no signal" without ever
    # crashing. Surface that loudly instead of failing silently.
    problems = []
    if len(df) < 210:  # need >=200 for EMA200 + a safety margin
        problems.append(f"only {len(df)} candles returned (expected ~500)")
    if df[["open", "high", "low", "close"]].isna().any().any():
        problems.append("NaN values found in OHLC data")
    if df["close"].tail(20).nunique() == 1:
        problems.append("last 20 closes are all identical (feed looks frozen/stale)")
    last_candle_age_min = (pd.Timestamp.utcnow().tz_localize(None) - df["time"].iloc[-1]).total_seconds() / 60
    if last_candle_age_min > 120:
        problems.append(f"most recent candle is {last_candle_age_min:.0f} minutes old (feed may be stale)")

    if problems:
        print(f"[data warning] Exchange feed looks suspicious: {'; '.join(problems)}. "
              f"Last close={df['close'].iloc[-1] if len(df) else 'N/A'}, "
              f"last candle time={df['time'].iloc[-1] if len(df) else 'N/A'}")
    else:
        print(f"[data check OK] {len(df)} candles fetched from {EXCHANGE_ID}, "
              f"last candle {df['time'].iloc[-1]}, last close={df['close'].iloc[-1]:.4f}")

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
    """factor can be a scalar OR a per-bar pandas Series (for auto-sensitivity).
    Source is ohlc4, exactly matching the original Pine Script's
    `supertrend(ohlc4, sensitivity, 10)` call - NOT hl2."""
    src = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    _atr = atr(df, length)
    if not isinstance(factor, pd.Series):
        factor = pd.Series(factor, index=df.index)

    upper = src + factor * _atr
    lower = src - factor * _atr

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


def ewma_volatility(logr, period=10, sqrt_annual=None):
    """Direct port of the Pine Script f_ewma() function used for
    Historical Volatility: lambda = (period-1)/(period+1), recursive."""
    lam = (period - 1) / (period + 1)
    squared = logr ** 2
    v = pd.Series(index=logr.index, dtype=float)
    prev = None
    for i in range(len(logr)):
        sq = squared.iloc[i]
        if pd.isna(sq):
            sq = 0.0
        if prev is None:
            prev = sq
        cur = lam * prev + (1 - lam) * sq
        v.iloc[i] = cur
        prev = cur
    return sqrt_annual * np.sqrt(v)


def compute_auto_sensitivity(df):
    """Direct port of the Pine Script auto-sensitivity block: computes
    Historical Volatility (EWMA model) and maps it to a Supertrend factor
    between 2.85 and 4, exactly matching the original indicator's bands."""
    period, annual, malen = 10, 365, 55
    sqrt_annual = np.sqrt(annual) * 100

    logr = np.log(df["close"] / df["close"].shift(1))
    hv = ewma_volatility(logr, period, sqrt_annual)
    avg_hv = hv.rolling(malen, min_periods=1).mean()

    maa = avg_hv / 100 * 140
    mab = avg_hv / 100 * 180
    mac = avg_hv / 100 * 240
    mad = avg_hv / 100 * 60
    mae = avg_hv / 100 * 20

    volatility = pd.Series(0.0, index=df.index)
    # Same branch order as the original Pine `if/else if` chain
    cond1 = (hv < maa) & (hv > avg_hv)
    cond2 = (hv < mab) & (hv > maa)
    cond3 = (hv < mac) & (hv > mab)
    cond4 = (hv > mac)
    cond5 = (hv < maa) & (hv > mad)
    cond6 = (hv < mad) & (hv > mae)
    cond7 = (hv < mae)

    volatility = np.select(
        [cond1, cond2, cond3, cond4, cond5, cond6, cond7],
        [3.15, 3.5, 3.6, 4.0, 3.0, 2.85, 3.0],
        default=3.0   # fallback if none match (shouldn't normally happen)
    )
    return pd.Series(volatility, index=df.index)


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
    df["sensitivity"] = compute_auto_sensitivity(df)
    df["st"], df["trend"] = supertrend(df, ATR_LEN, df["sensitivity"])
    df["ema200"] = ema(df["close"], 200)
    df["adx"] = adx(df, 14)
    df["vol_ema25"] = ema(df["volume"], 25)
    df["vol_ema26"] = ema(df["volume"], 26)

    cross_up = (df["close"] > df["st"]) & (df["close"].shift(1) <= df["st"].shift(1))
    cross_down = (df["close"] < df["st"]) & (df["close"].shift(1) >= df["st"].shift(1))

    # Keep the RAW crossover (matches the indicator's "Normal" strategy with
    # no extra filters) so we can tell apart "no crossover happened at all"
    # from "crossover happened but a filter blocked it".
    df["raw_bull"] = cross_up
    df["raw_bear"] = cross_down

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


def format_times(ts):
    """Return the candle time converted to Asia/Tehran local time only."""
    tehran_str = (ts + pd.Timedelta(hours=3, minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
    return f"{tehran_str} (Tehran time)"


def build_message(signal_type, row, ts):
    filters_note = "raw crossover (Normal strategy, no extra filters)"
    active = []
    if USE_SMART_FILTER:
        active.append("EMA200")
    if USE_ADX_FILTER:
        active.append("ADX")
    if USE_VOLUME_FILTER:
        active.append("Volume")
    if active:
        filters_note = "passed filters: " + " + ".join(active)
    return (f"{signal_type} signal ({filters_note})\n"
            f"{SYMBOL} [{TIMEFRAME}]\n"
            f"Close: {row['close']:.4f}\n"
            f"ADX: {row['adx']:.1f}\n"
            f"Candle time: {format_times(ts)}")


def main():
    if os.environ.get("FORCE_TEST_MESSAGE", "").lower() == "true":
        ok = send_telegram(
            f"Test message from GitHub Actions.\n"
            f"{SYMBOL} [{TIMEFRAME}] - if you see this, TG_BOT_TOKEN and "
            f"TG_CHAT_ID are correctly configured in this repo's Secrets."
        )
        print("Test message sent successfully." if ok else
              "Test message FAILED to send - check the [telegram error] above.")
        sys.exit(0 if ok else 1)

    exchange = getattr(ccxt, EXCHANGE_ID)()
    df = fetch_ohlcv(exchange, SYMBOL, TIMEFRAME)
    df = compute_signals(df)

    # All FULLY CLOSED candles - never the live/forming one
    closed = df.iloc[:-1].reset_index(drop=True)

    last_seen = load_last_seen()
    if last_seen:
        last_seen_ts = pd.to_datetime(last_seen)
        # Every closed candle strictly newer than the last one we processed
        new_rows = closed[closed["time"] > last_seen_ts]
    else:
        # First ever run: don't replay history, just start from the latest candle
        new_rows = closed.iloc[[-1]]

    if len(new_rows) == 0:
        print(f"No new closed candle since last run "
              f"({format_times(last_seen_ts)}). Nothing to do.")
        return

    if len(new_rows) > 1:
        print(f"[catch-up] {len(new_rows)} new closed candles found since last "
              f"run - checking all of them so no signal is skipped.")

    for _, row in new_rows.iterrows():
        ts = row["time"]
        ts_str = str(ts)

        if row["bull"]:
            msg = build_message("BUY", row, ts)
            print(msg)
            ok = send_telegram(msg)
            if not ok:
                print(f"[warning] Telegram send failed for candle {format_times(ts)}. "
                      f"State NOT advanced past this candle - it will be retried "
                      f"next run instead of being skipped or resent for candles "
                      f"already sent successfully.")
                sys.exit(1)
        elif row["bear"]:
            msg = build_message("SELL", row, ts)
            print(msg)
            ok = send_telegram(msg)
            if not ok:
                print(f"[warning] Telegram send failed for candle {format_times(ts)}. "
                      f"State NOT advanced past this candle - it will be retried "
                      f"next run instead of being skipped or resent for candles "
                      f"already sent successfully.")
                sys.exit(1)
        else:
            note = ""
            if row["raw_bull"] or row["raw_bear"]:
                blocked_type = "BUY" if row["raw_bull"] else "SELL"
                note = (f" [NOTE: a raw {blocked_type} crossover DID happen here, "
                        f"but was blocked by the EMA200/ADX filter - close="
                        f"{row['close']:.4f}, supertrend={row['st']:.4f}, "
                        f"adx={row['adx']:.1f}]")
            print(f"Closed candle {format_times(ts)} checked, no signal.{note}")

        # Save progress immediately after each candle (sent or signal-free).
        # This is what actually prevents duplicate notifications: once a
        # candle's alert is confirmed sent, it is marked done right away,
        # so even if a LATER candle in this same catch-up batch fails, the
        # earlier successful ones are never re-sent on the next run.
        save_last_seen(ts_str)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[fatal error] {e}")
        sys.exit(1)
