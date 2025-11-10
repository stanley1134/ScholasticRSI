"""
Renko Pro Scanner — v8 FULL CODE + P/L TOOLTIP (Entry | Current | $P/L)
- No Earnings within 7 days (Configurable)
- 1st/2nd Green Brick + K>D + K≤5 + X Red
- Persistent USA & India Watchlists
- Excel auto-add
- Fast batch rendering
- BUY/SELL Buttons with Live & Closed P/L
- ALL DATA (RESULTS + TRADES + SETTINGS) SAVED IN localStorage
- NO PAGE REFRESH ON BUY/SELL
- RESULTS GRID RESTORED ON PAGE REFRESH
- FIXED: No auto-refresh after scan
- FIXED: Excel upload typo (Branson → removed)
- NEW: P/L Cell Tooltip → Entry | Current/Exit | $P/L
"""

from flask import Flask, render_template_string, request, jsonify, Response
import pandas as pd
import numpy as np
import time
import datetime
import yfinance as yf
import uuid
import json
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from tradingview_ta import TA_Handler, Interval
import yahoo_fin.stock_info as si  # For earnings dates

app = Flask(__name__)

# ----------------------------------------------------------------------
# Cache (5 min) + Earnings Cache (1 hour)
# ----------------------------------------------------------------------
CACHE = {}
EARNINGS_CACHE = {}

def get_cached_data(symbol, period='120d'):
    key = f"{symbol}_{period}"
    now = datetime.datetime.now()
    if key in CACHE:
        data, ts = CACHE[key]
        if now - ts < datetime.timedelta(minutes=5):
            return data
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval='1d', actions=True)
        CACHE[key] = (df, now)
        return df
    except:
        return None

def get_earnings_date(symbol):
    key = symbol.upper()
    now = datetime.datetime.now()
    if key in EARNINGS_CACHE:
        date, ts = EARNINGS_CACHE[key]
        if now - ts < datetime.timedelta(hours=1):
            return date
    try:
        info = si.get_quote_table(symbol, dict_result=True)
        earnings_date = info.get('Earnings Date', None)
        if earnings_date:
            try:
                parsed = datetime.datetime.strptime(earnings_date.split(',')[0], '%b %d %Y').date()
            except:
                parsed = None
        else:
            parsed = None
        EARNINGS_CACHE[key] = (parsed, now)
        return parsed
    except:
        return None

# ----------------------------------------------------------------------
# Stoch RSI
# ----------------------------------------------------------------------
def calculate_rsi(prices, length=14):
    p = np.asarray(prices, dtype=float)
    if len(p) < length + 1:
        return np.full(len(p), np.nan)
    delta = np.diff(p)
    up = np.clip(delta, 0, None)
    down = np.abs(np.clip(delta, None, 0))
    avg_up = np.mean(up[:length])
    avg_down = np.mean(down[:length])
    rsi = np.full(len(p), np.nan)
    if avg_down > 0:
        rsi[length] = 100.0 - 100.0 / (1.0 + avg_up / avg_down)
    else:
        rsi[length] = 100.0
    for i in range(length + 1, len(p)):
        avg_up = (avg_up * (length - 1) + up[i-1]) / length
        avg_down = (avg_down * (length - 1) + down[i-1]) / length
        rs = avg_up / avg_down if avg_down > 0 else 1e10
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    rsi[:length] = rsi[length]
    return np.nan_to_num(rsi, nan=50.0)

def calculate_stoch_rsi(prices, rsi_length=14, stoch_length=14, k_sma=3, d_sma=3):
    rsi = calculate_rsi(prices, rsi_length)
    n = len(prices)
    if n < rsi_length + stoch_length:
        return np.full(n, np.nan), np.full(n, np.nan)
    raw_k = np.full(n, np.nan)
    for i in range(stoch_length - 1, n):
        rsi_window = rsi[i - stoch_length + 1 : i + 1]
        low_rsi = np.nanmin(rsi_window)
        high_rsi = np.nanmax(rsi_window)
        cur_rsi = rsi[i]
        if high_rsi > low_rsi:
            raw_k[i] = 100.0 * (cur_rsi - low_rsi) / (high_rsi - low_rsi)
        else:
            raw_k[i] = 50.0
    k_series = pd.Series(raw_k).rolling(window=k_sma, min_periods=1).mean()
    d_series = k_series.rolling(window=d_sma, min_periods=1).mean()
    return np.nan_to_num(k_series.to_numpy(), nan=50.0), np.nan_to_num(d_series.to_numpy(), nan=50.0)

# ----------------------------------------------------------------------
# Chart
# ----------------------------------------------------------------------
def generate_combined_chart(bricks, is_light_mode=False, width=800, height=500):
    if len(bricks) < 2:
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        bg = '#ffffff' if is_light_mode else '#0f172a'
        txt = '#000000' if is_light_mode else '#ffffff'
        plt.text(0.5, 0.5, 'Not enough data', ha='center', va='center', fontsize=12, color=txt)
        plt.axis('off')
        fig.patch.set_facecolor(bg)
    else:
        recent_bricks = bricks[-60:] if len(bricks) >= 60 else bricks
        renko_closes = [b['close'] for b in recent_bricks]
        opens = [renko_closes[0]] + renko_closes[:-1]
        closes = renko_closes
        highs = [max(o, c) for o, c in zip(opens, closes)]
        lows = [min(o, c) for o, c in zip(opens, closes)]
        green = '#059669' if is_light_mode else '#10b981'
        red = '#dc2626' if is_light_mode else '#ef4444'
        colors = [green if c >= o else red for o, c in zip(opens, closes)]

        stoch_k, stoch_d = calculate_stoch_rsi(renko_closes)

        bg = '#ffffff' if is_light_mode else '#0f172a'
        fg = '#000000' if is_light_mode else '#e2e8f0'
        grid = '#e5e7eb' if is_light_mode else '#334155'
        k_line = '#2563eb' if is_light_mode else '#3b82f6'
        d_line = '#ea580c' if is_light_mode else '#f97316'

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width/100, height/100), dpi=100,
                                       gridspec_kw={'height_ratios': [3, 2]}, facecolor=bg)
        fig.patch.set_facecolor(bg)

        for i in range(len(recent_bricks)):
            ax1.plot([i, i], [lows[i], highs[i]], color=colors[i], linewidth=6)
            ax1.plot([i-0.3, i+0.3], [opens[i], opens[i]], color=colors[i], linewidth=6)
            ax1.plot([i-0.3, i+0.3], [closes[i], closes[i]], color=colors[i], linewidth=6)
        ax1.set_ylabel('Price', color=fg)
        ax1.grid(True, alpha=0.3, color=grid)
        ax1.set_xticks([])
        ax1.set_facecolor(bg)
        ax1.tick_params(colors=fg)

        x = range(len(stoch_k))
        ax2.plot(x, stoch_k, color=k_line, linewidth=2.5, label='K')
        ax2.plot(x, stoch_d, color=d_line, linewidth=2.5, label='D')
        line_color = '#94a3b8' if is_light_mode else '#64748b'
        ax2.axhline(y=80, color=line_color, linestyle='--', alpha=0.7)
        ax2.axhline(y=40, color=line_color, linestyle='--', alpha=0.7)
        ax2.axhline(y=20, color=line_color, linestyle='--', alpha=0.7)
        ax2.axhline(y=5, color=line_color, linestyle='--', alpha=0.7)
        ax2.set_ylabel('StochRSI', color=fg)
        ax2.set_xlabel('Bricks ago', color=fg)
        ax2.set_ylim(0, 100)
        ax2.legend(fontsize=10, facecolor=bg, edgecolor=grid, labelcolor=fg)
        ax2.grid(True, alpha=0.3, color=grid)
        ax2.set_facecolor(bg)
        ax2.tick_params(colors=fg)

        plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor=bg)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def safe_float(x, default=0.0):
    try: return float(x) if x is not None else default
    except: return default

def validate_symbol(symbol, market):
    candidates = []
    if market == 'india':
        candidates.append(symbol.upper() + '.NS')
    else:
        candidates.append(symbol.upper())
    for s in candidates:
        df = get_cached_data(s)
        if df is not None and df.shape[0] >= 20:
            return s
    return None

def build_renko_percentage(df, percentage):
    closes = df['Close'].values
    if len(closes) < 2:
        return []
    bricks = []
    last_close = closes[0]
    box_size = last_close * (percentage / 100.0)
    if box_size <= 0:
        return []
    for price in closes[1:]:
        diff = price - last_close
        while abs(diff) >= box_size:
            direction = 1 if diff > 0 else -1
            last_close += box_size * direction
            bricks.append({'close': round(last_close, 4), 'direction': direction})
            diff = price - last_close
    return bricks

# ----------------------------------------------------------------------
# Core Analysis — v8: + No Earnings in 7 Days
# ----------------------------------------------------------------------
def analyze_ticker_local(symbol, brick_percent=1.0, touch_days=30, touch_threshold=5.0, 
                         is_light_mode=False, market='usa',
                         max_k=35, max_d=35, 
                         require_k_gt_d=True, require_touch=True,
                         min_red_bricks=5, green_brick_number=1,
                         require_no_earnings=True):
    valid_symbol = validate_symbol(symbol, market)
    if not valid_symbol:
        return {'success': False, 'ticker': symbol, 'error': 'Invalid symbol', 'tooltips': {'error': 'Not found'}}

    df = get_cached_data(valid_symbol)
    if df is None or df.shape[0] < 20:
        return {'success': False, 'ticker': symbol, 'error': 'No data', 'tooltips': {'error': 'Insufficient history'}}

    df = df.rename(columns={c: c.capitalize() for c in df.columns})

    # --- Earnings Check ---
    earnings_date = None
    earnings_in_7d = False
    if require_no_earnings:
        earnings_date = get_earnings_date(valid_symbol.split('.')[0])
        if earnings_date:
            days_to_earnings = (earnings_date - datetime.date.today()).days
            earnings_in_7d = 0 <= days_to_earnings <= 7

    # --- Dynamic Renko ---
    bricks = None
    final_percent = None
    percentages = [brick_percent, 0.5, 0.3, 0.1]
    for pct in percentages:
        bricks = build_renko_percentage(df, pct)
        if len(bricks) >= 30:
            final_percent = pct
            break

    if bricks and len(bricks) >= 30:
        green_streak = 0
        red_count_before_green = 0
        for i in range(len(bricks)-1, -1, -1):
            if bricks[i]['direction'] == 1:
                green_streak += 1
            else:
                for j in range(i, -1, -1):
                    if bricks[j]['direction'] == -1:
                        red_count_before_green += 1
                    else:
                        break
                break

        current_is_green = bricks[-1]['direction'] == 1
        renko_condition_met = (current_is_green and 
                              green_streak == green_brick_number and 
                              red_count_before_green >= min_red_bricks)

        renko_close_prices = [b['close'] for b in bricks]
        stoch_k, stoch_d = calculate_stoch_rsi(renko_close_prices)
        last_k, last_d = float(stoch_k[-1]), float(stoch_d[-1])

        k_bullish = last_k <= max_k and last_d <= max_d and (not require_k_gt_d or last_k > last_d)

        recent_k = stoch_k[-touch_days:] if len(stoch_k) >= touch_days else stoch_k
        touched_five_recent = require_touch and any(k <= touch_threshold for k in recent_k)

        no_earnings_ok = not require_no_earnings or not earnings_in_7d

        all_met = renko_condition_met and k_bullish and (not require_touch or touched_five_recent) and no_earnings_ok
        score = 100 if all_met else 50
        signal = 'BUY' if all_met else 'NEUTRAL'

        box_size = df['Close'].iloc[-1] * (final_percent / 100.0)
        chart_preview = generate_combined_chart(bricks, is_light_mode=is_light_mode)

        suffix = "st" if green_brick_number == 1 else "nd"
        earnings_text = f"Earnings: {earnings_date}" if earnings_date else "No earnings"

        return {
            'success': True,
            'ticker': symbol,
            'price': round(float(bricks[-1]['close']), 2),
            'brick_size': round(box_size, 4),
            'stoch_k': round(last_k, 2),
            'stoch_d': round(last_d, 2),
            'renko_condition_met': renko_condition_met,
            'k_bullish': k_bullish,
            'touched_five_recent': touched_five_recent,
            'no_earnings_ok': no_earnings_ok,
            'earnings_date': str(earnings_date) if earnings_date else None,
            'red_count_before_green': red_count_before_green,
            'green_streak': green_streak,
            'score': score,
            'signal': signal,
            'recommendation': signal,
            'chart_preview': chart_preview,
            'tooltips': {
                'renko': f"{green_brick_number}{suffix} green after {red_count_before_green}/{min_red_bricks} red | {final_percent}% LTP",
                'k_d': f"K={last_k:.1f}≤{max_k} | D={last_d:.1f}≤{max_d} | {'K>D' if last_k > last_d else 'K≤D'}",
                'touched': "K touched ≤5.0" if touched_five_recent else "K not ≤5.0",
                'earnings': earnings_text if not no_earnings_ok else "No earnings in 7d"
            }
        }

    # --- Fallback: TradingView ---
    try:
        time.sleep(0.2)
        exchange = 'NSE' if valid_symbol.endswith('.NS') else 'NASDAQ'
        screener = 'india' if valid_symbol.endswith('.NS') else 'america'
        handler = TA_Handler(symbol=valid_symbol, exchange=exchange, screener=screener,
                             interval=Interval.INTERVAL_1_DAY, timeout=10)
        analysis = handler.get_analysis()
        ind = analysis.indicators

        price = safe_float(ind.get('close'))
        k = safe_float(ind.get('STOCHRSIk')) or safe_float(ind.get('Stoch.K'))
        d = safe_float(ind.get('STOCHRSId')) or safe_float(ind.get('Stoch.D'))

        if pd.isna(k) or pd.isna(d):
            return {'success': False, 'ticker': symbol, 'error': 'No StochRSI', 'tooltips': {'error': 'Missing data'}}

        k_bullish = k <= max_k and d <= max_d and (not require_k_gt_d or k > d)
        touched_five_recent = require_touch and (k <= touch_threshold)
        no_earnings_ok = not require_no_earnings or not earnings_in_7d

        all_met = k_bullish and (not require_touch or touched_five_recent) and no_earnings_ok
        score = 100 if all_met else 50
        signal = 'BUY' if all_met else 'NEUTRAL'

        return {
            'success': True,
            'ticker': symbol,
            'price': round(price, 2) if price else 0,
            'brick_size': 0,
            'stoch_k': round(k, 2),
            'stoch_d': round(d, 2),
            'renko_condition_met': False,
            'k_bullish': k_bullish,
            'touched_five_recent': touched_five_recent,
            'no_earnings_ok': no_earnings_ok,
            'earnings_date': str(earnings_date) if earnings_date else None,
            'red_count_before_green': 0,
            'green_streak': 0,
            'score': score,
            'signal': signal,
            'recommendation': analysis.summary.get('RECOMMENDATION', 'NEUTRAL'),
            'chart_preview': '',
            'tooltips': {
                'renko': 'Low volatility → Renko skipped',
                'k_d': f"K={k:.1f}≤{max_k} | D={d:.1f}≤{max_d}",
                'touched': "K touched ≤5.0" if touched_five_recent else "K not ≤5.0",
                'earnings': earnings_text if not no_earnings_ok else "No earnings in 7d"
            }
        }
    except Exception as e2:
        err = 'Rate limited' if '429' in str(e2) else 'No data'
        return {'success': False, 'ticker': symbol, 'error': err, 'tooltips': {'error': str(e2)[:100]}}

# ----------------------------------------------------------------------
# HTML TEMPLATE — v8 + P/L TOOLTIP
# ----------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Renko Pro Scanner</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
* { box-sizing: border-box; margin:0; padding:0; }
:root { --transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); }
.light-mode { --bg: #f8fafc; --card: rgba(255, 255, 255, 0.8); --text: #1e293b; --text-light: #64748b; --primary: #3b82f6; --success: #10b981; --danger: #ef4444; --border: #e2e8f0; --glass: rgba(255, 255, 255, 0.7); }
.dark-mode { --bg: #0f172a; --card: rgba(30, 41, 59, 0.6); --text: #e2e8f0; --text-light: #94a3b8; --primary: #3b82f6; --success: #10b981; --danger: #ef4444; --border: #334155; --glass: rgba(255, 255, 255, 0.05); }
body { font-family: 'Inter', -apple-system, sans-serif; background: linear-gradient(135deg, var(--bg) 0%, color-mix(in srgb, var(--bg), #000 20%) 100%); color: var(--text); min-height: 100vh; padding: 16px; line-height: 1.6; transition: var(--transition); }
.container { max-width: 1400px; margin: 0 auto; }
.header { text-align: center; margin-bottom: 24px; animation: fadeIn 0.6s ease-out; }
.header h1 { font-size: 1.8rem; font-weight: 700; background: linear-gradient(90deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px; }
.header p { font-size: 0.9rem; color: var(--text-light); }
.glass-card { background: var(--card); backdrop-filter: blur(12px); border-radius: 16px; padding: 20px; margin-bottom: 16px; border: 1px solid var(--border); box-shadow: 0 8px 32px rgba(0,0,0,0.3); animation: fadeIn 0.5s ease-out; transition: var(--transition); }
.input-group { display: flex; flex-direction: column; gap: 8px; margin-bottom: 16px; }
.input-group label { font-size: 0.9rem; color: var(--text-light); font-weight: 500; }
.input-group input, .input-group textarea, .input-group select { padding: 12px; border-radius: 12px; border: 1px solid var(--border); background: var(--glass); color: var(--text); font-size: 1rem; transition: var(--transition); }
.input-group input:focus, .input-group textarea:focus, .input-group select:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3); }
.btn { padding: 12px 20px; border-radius: 12px; font-weight: 600; font-size: 1rem; cursor: pointer; transition: all 0.3s; border: none; display: inline-flex; align-items: center; justify-content: center; gap: 8px; }
.btn-primary { background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; box-shadow: 0 4px 14px rgba(59, 130, 246, 0.4); }
.btn-primary:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(59, 130, 246, 0.5); }
.btn-success { background: linear-gradient(135deg, #10b981, #34d399); color: white; }
.toggle-group { display: flex; gap: 8px; margin: 16px 0; }
.toggle-btn { flex: 1; padding: 10px; border-radius: 12px; font-weight: 600; font-size: 0.9rem; background: var(--glass); border: 1px solid var(--border); color: var(--text-light); transition: var(--transition); }
.toggle-btn.active { background: linear-gradient(135deg, var(--primary), #8b5cf6); color: white; border-color: transparent; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4); }
.results h3 { font-size: 1.1rem; margin-bottom: 12px; color: var(--text); }
.table-container { overflow-x: auto; border-radius: 12px; border: 1px solid var(--border); background: var(--card); backdrop-filter: blur(8px); }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th, td { padding: 12px 8px; text-align: left; border-bottom: 1px solid var(--border); }
th { background: rgba(51, 65, 85, 0.3); font-weight: 600; color: var(--text); }
tr:hover { background: rgba(255, 255, 255, 0.03); }
.buy-row { background: rgba(16, 185, 129, 0.15) !important; border-left: 4px solid var(--success); }
.error-row { background: rgba(239, 68, 68, 0.1); color: #fca5a5; }
.signal-badge { padding: 4px 10px; border-radius: 8px; font-weight: 700; font-size: 0.75rem; }
.signal-badge.BUY { background: var(--success); color: white; }
.signal-badge.NEUTRAL { background: #64748b; color: white; }
.preview-img { width: 100%; max-width: 140px; height: 70px; border-radius: 8px; cursor: zoom-in; border: 1px solid var(--border); transition: 0.3s; object-fit: cover; }
.preview-img:hover { transform: scale(1.05); box-shadow: 0 8px 20px rgba(0,0,0,0.3); }
.tooltip { position: relative; display: inline-block; cursor: help; }
.tooltip .tooltiptext { visibility: hidden; width: 220px; background: #1e293b; color: #fff; text-align: center; border-radius: 8px; padding: 8px; position: absolute; z-index: 10; bottom: 125%; left: 50%; margin-left: -110px; opacity: 0; transition: opacity 0.3s; font-size: 0.8rem; font-weight: 500; box-shadow: 0 4px 12px rgba(0,0,0,0.3); white-space: nowrap; }
.tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
.spinner { border: 4px solid var(--glass); border-top: 4px solid var(--primary); border-radius: 50%; width: 36px; height: 36px; animation: spin 1s linear infinite; margin: 20px auto; display: none; }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
.modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); justify-content: center; align-items: center; padding: 20px; }
.modal img { max-width: 100%; max-height: 90vh; border-radius: 12px; box-shadow: 0 20px 40px rgba(0,0,0,0.5); }
.modal .close { position: absolute; top: 20px; right: 30px; color: #fff; font-size: 40px; font-weight: bold; cursor: pointer; }
.error { background: rgba(239, 68, 68, 0.15); border: 1px solid #f87171; color: #fca5a5; padding: 12px; border-radius: 12px; margin-bottom: 16px; font-size: 0.9rem; }
.success { background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #6ee7b7; padding: 12px; border-radius: 12px; margin-bottom: 16px; font-size: 0.9rem; }
.theme-toggle { position: fixed; top: 16px; right: 16px; z-index: 1000; width: 48px; height: 48px; border-radius: 50%; background: var(--card); border: 1px solid var(--border); display: flex; align-items: center; justify-content: center; cursor: pointer; transition: var(--transition); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
.theme-toggle:hover { transform: scale(1.1); }
.theme-toggle svg { width: 24px; height: 24px; fill: var(--text); transition: var(--transition); }
.action-btn { padding: 4px 8px; border-radius: 6px; font-size: 0.7rem; font-weight: 600; cursor: pointer; border: none; margin-right: 4px; }
.buy-btn { background: #10b981; color: white; }
.sell-btn { background: #ef4444; color: white; }
.pl-positive { color: #10b981; font-weight: 600; }
.pl-negative { color: #ef4444; font-weight: 600; }
.pl-zero { color: #94a3b8; }
@media (max-width: 768px) { .header h1 { font-size: 1.5rem; } .glass-card { padding: 16px; } .btn { padding: 10px 16px; font-size: 0.9rem; } table { font-size: 0.75rem; } .preview-img { height: 60px; } }
</style>
</head>
<body class="dark-mode">
<div class="container">
  <div class="theme-toggle" id="themeToggle">
    <svg id="sunIcon" viewBox="0 0 24 24" style="display:none;"><path d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"/></svg>
    <svg id="moonIcon" viewBox="0 0 24 24"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
  </div>

  <div class="header">
    <h1>Renko Pro Scanner</h1>
    <p>BUY = Nth Green + K>D + K≤5 + X Red + No Earnings in 7d</p>
  </div>

  <div id="msgBox" class="hidden"></div>

  <div class="glass-card">
    <div class="input-group">
      <label>Upload Excel (.xlsx) → Auto-add to Watchlist</label>
      <input type="file" id="excelFile" accept=".xlsx"/>
      <button type="button" class="btn btn-primary" id="uploadBtn">Upload & Save</button>
      <span id="excelStatus" style="font-size:0.8rem;color:var(--text-light);"></span>
    </div>

    <div class="input-group">
      <label>Or Enter Tickers (comma-separated)</label>
      <textarea id="tickers" placeholder="AAPL, RELIANCE.NS..." style="height:80px;"></textarea>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
      <div class="input-group">
        <label>Box Size (% LTP)</label>
        <input id="brickPercent" value="1.0" type="number" step="0.1" min="0.1"/>
      </div>
      <div class="input-group">
        <label>Touch Bricks</label>
        <input id="touchDays" value="30" type="number"/>
      </div>
      <div class="input-group">
        <label>Threshold</label>
        <input id="touchThreshold" value="5.0" type="number" step="0.1"/>
      </div>
      <div class="input-group">
        <label>Market</label>
        <div class="toggle-group">
          <button type="button" id="toggleIndia" class="toggle-btn">India</button>
          <button type="button" id="toggleUSA" class="toggle-btn active">USA</button>
        </div>
      </div>
      <div class="input-group">
        <label>Max K/D</label>
        <input id="maxKd" value="35" type="number" step="1" min="1" max="100"/>
      </div>
      <div class="input-group">
        <label>K > D?</label>
        <div class="toggle-group">
          <button type="button" id="toggleKgtDYes" class="toggle-btn active">Yes</button>
          <button type="button" id="toggleKgtDNo" class="toggle-btn">No</button>
        </div>
      </div>
      <div class="input-group">
        <label>Require K≤5 Touch?</label>
        <div class="toggle-group">
          <button type="button" id="toggleTouchYes" class="toggle-btn active">Yes</button>
          <button type="button" id="toggleTouchNo" class="toggle-btn">No</button>
        </div>
      </div>
      <div class="input-group">
        <label>Green Brick</label>
        <div class="toggle-group">
          <button type="button" id="toggle1st" class="toggle-btn active">1st</button>
          <button type="button" id="toggle2nd" class="toggle-btn">2nd</button>
        </div>
      </div>
      <div class="input-group">
        <label>Min Red Bricks</label>
        <input id="minRedBricks" value="5" type="number" min="1" step="1"/>
      </div>
      <div class="input-group">
        <label>No Earnings in 7d?</label>
        <div class="toggle-group">
          <button type="button" id="toggleEarningsYes" class="toggle-btn active">Yes</button>
          <button type="button" id="toggleEarningsNo" class="toggle-btn">No</button>
        </div>
      </div>
    </div>

    <div style="display:flex;gap:12px;margin-top:12px;">
      <select id="watchlistSelect" style="flex:1;">
        <option value="">— Select Watchlist —</option>
        <option value="usa">USA Watchlist</option>
        <option value="india">India Watchlist</option>
      </select>
      <button type="button" class="btn btn-primary" id="loadWatchlistBtn">Load</button>
      <button type="button" class="btn btn-success" id="saveCurrentBtn">Save Current</button>
    </div>

    <button type="button" class="btn btn-success" id="scanBtn" style="width:100%;margin-top:12px;">
      Scan Stocks
    </button>
  </div>

  <div id="spinner" class="spinner hidden"></div>

  <div id="results" class="results hidden">
    <div class="glass-card">
      <h3>Results (<span id="resultCount">0</span>)</h3>
      <div class="table-container">
        <table id="resultsTable">
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Chart</th>
              <th>Price</th>
              <th>Brick</th>
              <th>K</th>
              <th>D</th>
              <th>Signal</th>
              <th>Score</th>
              <th>Renko</th>
              <th>K>D & ≤Y</th>
              <th>K≤5</th>
              <th>Earnings</th>
              <th>Red Bricks</th>
              <th>Action</th>
              <th>P/L</th>
            </tr>
          </thead>
          <tbody id="tableBody"></tbody>
        </table>
      </div>
      <button type="button" class="btn btn-primary" id="exportBtn" style="margin-top:12px;width:100%;">Export CSV</button>
      <button type="button" class="btn btn-primary" id="clearTradesBtn" style="margin-top:8px;width:100%;">Clear All Trades</button>
    </div>
  </div>
</div>

<div id="chartModal" class="modal">
  <span class="close">X</span>
  <img id="modalImage" src="" alt="Full Chart"/>
</div>

<script>
// === FULL PERSISTENCE: Results + Trades + Settings ===
let results = JSON.parse(localStorage.getItem('renko_results') || '[]');
let trades = JSON.parse(localStorage.getItem('renko_trades') || '{}');
let watchlists = {
  usa: JSON.parse(localStorage.getItem('watchlist_usa') || '[]'),
  india: JSON.parse(localStorage.getItem('watchlist_india') || '[]')
};

function saveResults() { localStorage.setItem('renko_results', JSON.stringify(results)); }
function saveTrades() { localStorage.setItem('renko_trades', JSON.stringify(trades)); }
function saveWatchlists() {
  localStorage.setItem('watchlist_usa', JSON.stringify(watchlists.usa));
  localStorage.setItem('watchlist_india', JSON.stringify(watchlists.india));
}

const BATCH_SIZE = 50;
let renderQueue = [];
let renderTimer = null;
let eventSource = null;
let currentMarket = 'usa';
let isLightMode = localStorage.getItem('theme') === 'light';
let requireKgtD = true;
let requireTouch = true;
let greenBrickNumber = 1;
let requireNoEarnings = true;
let plUpdateInterval = null;

const themeToggle = document.getElementById('themeToggle');
const sunIcon = document.getElementById('sunIcon');
const moonIcon = document.getElementById('moonIcon');

function applyTheme() {
  if (isLightMode) {
    document.body.classList.remove('dark-mode');
    document.body.classList.add('light-mode');
    sunIcon.style.display = 'block';
    moonIcon.style.display = 'none';
  } else {
    document.body.classList.remove('light-mode');
    document.body.classList.add('dark-mode');
    sunIcon.style.display = 'none';
    moonIcon.style.display = 'block';
  }
}
applyTheme();

themeToggle.onclick = () => {
  isLightMode = !isLightMode;
  localStorage.setItem('theme', isLightMode ? 'light' : 'dark');
  applyTheme();
  saveSettings();
};

function showMsg(msg, isSuccess = false) {
  const el = document.getElementById('msgBox');
  el.textContent = msg;
  el.className = isSuccess ? 'success' : 'error';
  el.classList.remove('hidden');
  setTimeout(() => el.classList.add('hidden'), 3000);
}

function showSpinner(){ document.getElementById('spinner').style.display = 'block'; }
function hideSpinner(){ document.getElementById('spinner').style.display = 'none'; }

// Prevent form submission
document.querySelectorAll('button').forEach(btn => {
  btn.addEventListener('click', e => e.preventDefault());
});

document.getElementById('toggleIndia').onclick = () => setMarket('india');
document.getElementById('toggleUSA').onclick = () => setMarket('usa');
document.getElementById('toggleKgtDYes').onclick = () => { requireKgtD = true; updateKgtDToggle(); saveSettings(); };
document.getElementById('toggleKgtDNo').onclick = () => { requireKgtD = false; updateKgtDToggle(); saveSettings(); };
document.getElementById('toggleTouchYes').onclick = () => { requireTouch = true; updateTouchToggle(); saveSettings(); };
document.getElementById('toggleTouchNo').onclick = () => { requireTouch = false; updateTouchToggle(); saveSettings(); };
document.getElementById('toggle1st').onclick = () => { greenBrickNumber = 1; updateGreenToggle(); saveSettings(); };
document.getElementById('toggle2nd').onclick = () => { greenBrickNumber = 2; updateGreenToggle(); saveSettings(); };
document.getElementById('toggleEarningsYes').onclick = () => { requireNoEarnings = true; updateEarningsToggle(); saveSettings(); };
document.getElementById('toggleEarningsNo').onclick = () => { requireNoEarnings = false; updateEarningsToggle(); saveSettings(); };
document.getElementById('uploadBtn').onclick = uploadExcel;
document.getElementById('scanBtn').onclick = scanStocks;
document.getElementById('exportBtn').onclick = exportCSV;
document.getElementById('loadWatchlistBtn').onclick = loadWatchlist;
document.getElementById('saveCurrentBtn').onclick = saveCurrentToWatchlist;
document.getElementById('clearTradesBtn').onclick = clearAllTrades;
document.querySelector('.modal').onclick = closeModal;
document.querySelector('.close').onclick = closeModal;

function setMarket(market){
  currentMarket = market;
  document.getElementById('toggleIndia').classList.toggle('active', market==='india');
  document.getElementById('toggleUSA').classList.toggle('active', market==='usa');
  saveSettings();
}

function updateKgtDToggle() {
  document.getElementById('toggleKgtDYes').classList.toggle('active', requireKgtD);
  document.getElementById('toggleKgtDNo').classList.toggle('active', !requireKgtD);
}
function updateTouchToggle() {
  document.getElementById('toggleTouchYes').classList.toggle('active', requireTouch);
  document.getElementById('toggleTouchNo').classList.toggle('active', !requireTouch);
}
function updateGreenToggle() {
  document.getElementById('toggle1st').classList.toggle('active', greenBrickNumber === 1);
  document.getElementById('toggle2nd').classList.toggle('active', greenBrickNumber === 2);
}
function updateEarningsToggle() {
  document.getElementById('toggleEarningsYes').classList.toggle('active', requireNoEarnings);
  document.getElementById('toggleEarningsNo').classList.toggle('active', !requireNoEarnings);
}

async function uploadExcel(){
  const f = document.getElementById('excelFile').files[0];
  if(!f){ showMsg('Choose .xlsx'); return; }
  const form = new FormData(); form.append('file', f);
  document.getElementById('excelStatus').textContent='Uploading...';
  try{
    const resp = await fetch('/api/upload_excel', {method:'POST', body:form});
    const data = await resp.json();
    if(!data.success){ showMsg(data.error); document.getElementById('excelStatus').textContent=''; return; }
    document.getElementById('tickers').value = data.tickers.join(',');
    document.getElementById('excelStatus').textContent = `Loaded ${data.tickers.length} tickers`;

    const key = currentMarket === 'india' ? 'india' : 'usa';
    const newTickers = data.tickers.filter(t => !watchlists[key].includes(t));
    if (newTickers.length > 0) {
      watchlists[key].push(...newTickers);
      saveWatchlists();
      showMsg(`Added ${newTickers.length} to ${key.toUpperCase()} Watchlist`, true);
    }
  }catch(e){ showMsg('Upload error: '+e.message); document.getElementById('excelStatus').textContent=''; }
}

function loadWatchlist() {
  const sel = document.getElementById('watchlistSelect').value;
  if (!sel) return showMsg('Select a watchlist');
  const list = watchlists[sel];
  if (list.length === 0) return showMsg(`${sel.toUpperCase()} watchlist is empty`);
  document.getElementById('tickers').value = list.join(',');
  setMarket(sel);
  showMsg(`Loaded ${list.length} from ${sel.toUpperCase()} watchlist`, true);
}

function saveCurrentToWatchlist() {
  const raw = document.getElementById('tickers').value.trim();
  const tickers = raw.split(',').map(s=>s.trim()).filter(s=>s);
  if (!tickers.length) return showMsg('No tickers to save');
  const key = currentMarket;
  watchlists[key] = [...new Set(tickers)];
  saveWatchlists();
  showMsg(`Saved ${tickers.length} to ${key.toUpperCase()} watchlist`, true);
}

function clearAllTrades() {
  if (confirm('Clear all trades?')) {
    trades = {};
    saveTrades();
    document.querySelectorAll('#tableBody tr').forEach(row => {
      const plCell = row.cells[row.cells.length - 1];
      plCell.innerHTML = '-';
      const buyBtn = row.querySelector('.buy-btn');
      const sellBtn = row.querySelector('.sell-btn');
      if (buyBtn) buyBtn.disabled = false;
      if (sellBtn) sellBtn.disabled = true;
    });
    showMsg('All trades cleared', true);
  }
}

function buyStock(ticker, price) {
  if (trades[ticker] && !trades[ticker].exit) {
    showMsg(`Already holding ${ticker}`, false);
    return;
  }
  trades[ticker] = { entry: price, exit: null, timestamp: Date.now() };
  saveTrades();
  updateRowPL(ticker);
  showMsg(`BUY ${ticker} @ $${price}`, true);
}

function sellStock(ticker, price) {
  if (!trades[ticker] || trades[ticker].exit) {
    showMsg(`No open position in ${ticker}`, false);
    return;
  }
  trades[ticker].exit = price;
  saveTrades();
  updateRowPL(ticker);
  const pl = ((price - trades[ticker].entry) / trades[ticker].entry * 100).toFixed(2);
  showMsg(`SELL ${ticker} @ $${price} | P/L: ${pl}%`, true);
}

// === UPDATED: P/L with Tooltip (Entry | Current/Exit | $P/L) ===
function updateRowPL(ticker) {
  const row = Array.from(document.querySelectorAll('#tableBody tr'))
    .find(r => r.cells[0].textContent.trim() === ticker);
  if (!row) return;

  const plCell = row.cells[row.cells.length - 1];
  const trade = trades[ticker];

  if (!trade || !trade.entry) {
    plCell.innerHTML = '-';
    return;
  }

  const currentPrice = parseFloat(row.cells[2].textContent.replace('$', '')) || 0;
  const entry = trade.entry;
  const exit = trade.exit;
  const isClosed = !!exit;
  const finalPrice = isClosed ? exit : currentPrice;

  const pct = ((finalPrice - entry) / entry * 100).toFixed(2);
  const dollarPL = (finalPrice - entry).toFixed(2);
  const signPct = pct > 0 ? '+' : '';
  const signDol = dollarPL > 0 ? '+' : '';

  const className = isClosed 
    ? (pct > 0 ? 'pl-positive' : pct < 0 ? 'pl-negative' : 'pl-zero')
    : (currentPrice > entry ? 'pl-positive' : currentPrice < entry ? 'pl-negative' : 'pl-zero');

  const tooltipText = `Entry: $${entry.toFixed(2)} | ${isClosed ? 'Exit' : 'Current'}: $${finalPrice.toFixed(2)} | $${signDol}${dollarPL}`;

  plCell.innerHTML = `
    <div class="tooltip">
      <span class="${className}">${signPct}${pct}%</span>
      <span class="tooltiptext">${tooltipText}</span>
    </div>
  `;
}

function startPLUpdates() {
  if (plUpdateInterval) return;
  plUpdateInterval = setInterval(() => {
    Object.keys(trades).forEach(ticker => {
      if (trades[ticker] && !trades[ticker].exit) {
        updateRowPL(ticker);
      }
    });
  }, 10000);
}

function stopPLUpdates() {
  if (plUpdateInterval) {
    clearInterval(plUpdateInterval);
    plUpdateInterval = null;
  }
}

function enhanceRowWithTradeButtons(tr, ticker) {
  const actionCell = tr.insertCell();
  const plCell = tr.insertCell();

  const isHeld = trades[ticker] && !trades[ticker].exit;
  const isClosed = trades[ticker] && trades[ticker].exit;

  actionCell.innerHTML = `
    <button type="button" class="action-btn buy-btn" onclick="buyStock('${ticker}', ${tr.cells[2].textContent.replace('$','') || 0})"
            ${isHeld || isClosed ? 'disabled' : ''}>BUY</button>
    <button type="button" class="action-btn sell-btn" onclick="sellStock('${ticker}', ${tr.cells[2].textContent.replace('$','') || 0})"
            ${!isHeld ? 'disabled' : ''}>SELL</button>
  `;

  updateRowPL(ticker); // Includes rich tooltip
}

async function scanStocks(){
  showSpinner();
  let raw = document.getElementById('tickers').value.trim();
  let tickers = raw.split(',').map(s=>s.trim()).filter(s=>s);
  if(!tickers.length){ hideSpinner(); showMsg('No tickers'); return; }

  const brickPercent = parseFloat(document.getElementById('brickPercent').value)||1.0;
  const touchDays = parseInt(document.getElementById('touchDays').value)||30;
  const touchThreshold = parseFloat(document.getElementById('touchThreshold').value)||5.0;
  const maxKd = parseFloat(document.getElementById('maxKd').value)||35;
  const minRedBricks = parseInt(document.getElementById('minRedBricks').value)||5;

  results = [];
  saveResults();
  document.getElementById('tableBody').innerHTML = '';
  renderQueue = [];
  document.getElementById('resultCount').textContent = '0';
  document.getElementById('results').classList.remove('hidden');

  if(eventSource) eventSource.close();

  const startResp = await fetch('/api/scan_start', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      tickers, market: currentMarket,
      params:{
        brick_percent:brickPercent, touch_days:touchDays, touch_threshold:touchThreshold,
        max_kd:maxKd, require_k_gt_d:requireKgtD, require_touch:requireTouch,
        min_red_bricks:minRedBricks, green_brick_number: greenBrickNumber,
        require_no_earnings: requireNoEarnings
      },
      is_light_mode: isLightMode
    })
  });
  const startData = await startResp.json();
  if(!startData.success){ hideSpinner(); showMsg(startData.error); return; }

  eventSource = new EventSource(`/api/scan_stream?token=${startData.token}`);

  eventSource.onmessage = function(e){
    if(e.data === '__END__'){
      eventSource.close();
      hideSpinner();
      saveResults();
      scheduleFinalFlush();
      startPLUpdates();
      return;
    }
    let res;
    try { res = JSON.parse(e.data); } catch(err) { return; }
    results.push(res);
    renderQueue.push(res);
    if (renderQueue.length >= BATCH_SIZE) flushRenderQueue();
    scheduleFinalFlush();
  };

  eventSource.onerror = function(){
    eventSource.close();
    hideSpinner();
    saveResults();
    if(results.length === 0){
      showMsg('No data. Check tickers or network.');
    } else {
      showMsg('Partial results. Some failed.');
    }
  };
}

function flushRenderQueue(){
  if (!renderQueue.length) return;
  const fragment = document.createDocumentFragment();
  const tbody = document.getElementById('tableBody');

  for (const r of renderQueue) {
    const tr = document.createElement('tr');
    if (r.signal === 'BUY') tr.classList.add('buy-row');
    if (!r.success) tr.classList.add('error-row');

    const img = r.chart_preview
      ? `<img src="${r.chart_preview}" class="preview-img" alt="Chart"
              onclick="openModal('${r.chart_preview.replace(/'/g, "\\'")}')">`
      : '';

    const errorCell = !r.success
      ? `<div class="tooltip">Error<span class="tooltiptext">${r.tooltips.error}</span></div>`
      : '';

    const renkoCell = r.renko_condition_met !== undefined
      ? `<div class="tooltip">${r.renko_condition_met ? 'Yes' : 'No'}<span class="tooltiptext">${r.tooltips.renko}</span></div>`
      : errorCell;

    const kdCell = r.k_bullish !== undefined
      ? `<div class="tooltip">${r.k_bullish ? 'Yes' : 'No'}<span class="tooltiptext">${r.tooltips.k_d}</span></div>` : '';

    const touchedCell = r.touched_five_recent !== undefined
      ? `<div class="tooltip">${r.touched_five_recent ? 'Yes' : 'No'}<span class="tooltiptext">${r.tooltips.touched}</span></div>` : '';

    const earningsCell = r.no_earnings_ok !== undefined
      ? `<div class="tooltip">${r.no_earnings_ok ? 'OK' : 'Soon'}<span class="tooltiptext">${r.tooltips.earnings}</span></div>` : '-';

    const redBricksCell = r.red_count_before_green !== undefined
      ? `<div class="tooltip">${r.red_count_before_green}/${minRedBricks}<span class="tooltiptext">${r.red_count_before_green} red bricks before green</span></div>` : '-';

    tr.innerHTML = `
      <td><strong>${r.ticker}</strong></td>
      <td>${img}</td>
      <td>${r.price !== undefined ? '$'+r.price : '-'}</td>
      <td>${r.brick_size !== undefined ? '$'+r.brick_size : '-'}</td>
      <td>${r.stoch_k !== undefined ? r.stoch_k : '-'}</td>
      <td>${r.stoch_d !== undefined ? r.stoch_d : '-'}</td>
      <td>${r.signal ? `<span class="signal-badge ${r.signal}">${r.signal}</span>` : '-'}</td>
      <td>${r.score !== undefined ? r.score : '-'}</td>
      <td>${renkoCell}</td>
      <td>${kdCell}</td>
      <td>${touchedCell}</td>
      <td>${earningsCell}</td>
      <td>${redBricksCell}</td>
    `;
    enhanceRowWithTradeButtons(tr, r.ticker);
    fragment.appendChild(tr);
  }

  tbody.appendChild(fragment);
  renderQueue = [];
  document.getElementById('resultCount').textContent = results.length;

  Object.keys(trades).forEach(ticker => {
    if (trades[ticker] && !trades[ticker].exit) {
      updateRowPL(ticker);
    }
  });
}

function scheduleFinalFlush(){
  clearTimeout(renderTimer);
  renderTimer = setTimeout(() => {
    flushRenderQueue();
    sortTable();
  }, 120);
}

function openModal(src) {
  const modal = document.getElementById('chartModal');
  const img = document.getElementById('modalImage');
  img.src = src;
  modal.style.display = 'flex';
  setTimeout(() => modal.style.opacity = 1, 10);
}

function closeModal() {
  const modal = document.getElementById('chartModal');
  modal.style.opacity = 0;
  setTimeout(() => modal.style.display = 'none', 300);
}

function sortTable(){
  const tbody = document.getElementById('tableBody');
  const rows  = Array.from(tbody.rows);
  rows.sort((a,b) => {
    const sa = a.cells[6].querySelector('.signal-badge')?.textContent || '';
    const sb = b.cells[6].querySelector('.signal-badge')?.textContent || '';
    const scoreA = parseInt(a.cells[7].textContent) || 0;
    const scoreB = parseInt(b.cells[7].textContent) || 0;
    if(sa === 'BUY' && sb !== 'BUY') return -1;
    if(sb === 'BUY' && sa !== 'BUY') return  1;
    return scoreB - scoreA;
  });
  tbody.innerHTML = '';
  rows.forEach(r => tbody.appendChild(r));
}

function exportCSV(){
  const headers = ['Ticker','Price','BrickSize','StochK','StochD','Signal','Score','RenkoCondition','KBullish','KTouch5','NoEarningsOK','EarningsDate','RedBricksBefore','GreenStreak','Entry','Exit','PL'];
  const rows = results.map(r => {
    const trade = trades[r.ticker] || {};
    const pl = trade.entry ? (trade.exit 
      ? ((trade.exit - trade.entry) / trade.entry * 100).toFixed(2)
      : ((r.price - trade.entry) / trade.entry * 100).toFixed(2)
    ) : '';
    return [
      r.ticker, r.price||'', r.brick_size||'', r.stoch_k||'', r.stoch_d||'', r.signal||'', r.score||'',
      r.renko_condition_met !== undefined ? (r.renko_condition_met ? 'YES' : 'NO') : '',
      r.k_bullish !== undefined ? (r.k_bullish ? 'YES' : 'NO') : '',
      r.touched_five_recent !== undefined ? (r.touched_five_recent ? 'YES' : 'NO') : '',
      r.no_earnings_ok !== undefined ? (r.no_earnings_ok ? 'YES' : 'NO') : '',
      r.earnings_date || '',
      r.red_count_before_green !== undefined ? r.red_count_before_green : '',
      r.green_streak !== undefined ? r.green_streak : '',
      trade.entry || '',
      trade.exit || '',
      pl
    ];
  });
  const csv = [headers, ...rows].map(r => r.join(',')).join('\n');
  const blob = new Blob([csv], {type: 'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = 'renko_scan_with_trades.csv'; a.click();
}

function saveSettings() {
  const settings = {
    brickPercent: document.getElementById('brickPercent').value,
    touchDays: document.getElementById('touchDays').value,
    touchThreshold: document.getElementById('touchThreshold').value,
    maxKd: document.getElementById('maxKd').value,
    minRedBricks: document.getElementById('minRedBricks').value,
    market: currentMarket,
    requireKgtD: requireKgtD,
    requireTouch: requireTouch,
    greenBrickNumber: greenBrickNumber,
    requireNoEarnings: requireNoEarnings,
    theme: isLightMode ? 'light' : 'dark'
  };
  localStorage.setItem('renkoSettings', JSON.stringify(settings));
}

function loadSettings() {
  const saved = localStorage.getItem('renkoSettings');
  if (!saved) return;
  const s = JSON.parse(saved);
  document.getElementById('brickPercent').value = s.brickPercent || 1.0;
  document.getElementById('touchDays').value = s.touchDays || 30;
  document.getElementById('touchThreshold').value = s.touchThreshold || 5.0;
  document.getElementById('maxKd').value = s.maxKd || 35;
  document.getElementById('minRedBricks').value = s.minRedBricks || 5;
  if (s.market) setMarket(s.market);
  requireKgtD = s.requireKgtD !== false;
  requireTouch = s.requireTouch !== false;
  greenBrickNumber = s.greenBrickNumber || 1;
  requireNoEarnings = s.requireNoEarnings !== false;
  updateKgtDToggle();
  updateTouchToggle();
  updateGreenToggle();
  updateEarningsToggle();
  if (s.theme === 'light') { isLightMode = true; }
  applyTheme();
}

function restoreResults() {
  if (results.length === 0) return;
  document.getElementById('results').classList.remove('hidden');
  document.getElementById('resultCount').textContent = results.length;
  const tbody = document.getElementById('tableBody');
  tbody.innerHTML = '';
  renderQueue = results.slice();
  flushRenderQueue();
  sortTable();
  startPLUpdates();
}

window.addEventListener('load', () => {
  hideSpinner();
  loadSettings();
  restoreResults();
});
</script>
</body>
</html>"""

# ----------------------------------------------------------------------
# Flask Routes
# ----------------------------------------------------------------------
STREAM_TOKENS = {}

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/upload_excel', methods=['POST'])
def upload_excel():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file'}), 400
        f = request.files['file']
        df = pd.read_excel(f, engine='openpyxl')
        if df.empty:
            return jsonify({'success': False, 'error': 'Empty file'}), 400
        cols = [c.lower() for c in df.columns]
        if 'ticker' in cols:
            tickers = df[df.columns[cols.index('ticker')]].dropna().astype(str).str.strip().tolist()
        else:
            tickers = df.iloc[:,0].dropna().astype(str).str.strip().tolist()
        tickers = [t.replace(' ', '').strip() for t in tickers if t]
        return jsonify({'success': True, 'tickers': tickers})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/scan_start', methods=['POST'])
def scan_start():
    data = request.get_json() or {}
    token = str(uuid.uuid4())
    params = data.get('params', {}) or {}
    STREAM_TOKENS[token] = {
        'tickers': data.get('tickers', []),
        'market': data.get('market', 'usa'),
        'params': {
            'brick_percent': float(params.get('brick_percent', 1.0)),
            'touch_days': int(params.get('touch_days', 30)),
            'touch_threshold': float(params.get('touch_threshold', 5.0)),
            'max_kd': float(params.get('max_kd', 35)),
            'require_k_gt_d': params.get('require_k_gt_d', True) in (True, 'true', 'True'),
            'require_touch': params.get('require_touch', True) in (True, 'true', 'True'),
            'min_red_bricks': int(params.get('min_red_bricks', 5)),
            'green_brick_number': int(params.get('green_brick_number', 1)),
            'require_no_earnings': params.get('require_no_earnings', True) in (True, 'true', 'True')
        },
        'is_light_mode': data.get('is_light_mode', False)
    }
    return jsonify({'success': True, 'token': token})

@app.route('/api/scan_stream')
def scan_stream():
    token = request.args.get('token')
    payload = STREAM_TOKENS.pop(token, None)
    if not payload:
        return "Invalid token", 400

    tickers = payload.get('tickers', [])
    params = payload.get('params', {}) or {}
    brick_percent = params.get('brick_percent', 1.0)
    touch_days = params.get('touch_days', 30)
    touch_threshold = params.get('touch_threshold', 5.0)
    max_kd = params.get('max_kd', 35)
    require_k_gt_d = params.get('require_k_gt_d', True)
    require_touch = params.get('require_touch', True)
    min_red_bricks = params.get('min_red_bricks', 5)
    green_brick_number = params.get('green_brick_number', 1)
    require_no_earnings = params.get('require_no_earnings', True)
    is_light_mode = payload.get('is_light_mode', False)
    market = payload.get('market', 'usa')

    def generate():
        for t in tickers:
            t = t.strip()
            if not t: continue
            res = analyze_ticker_local(
                t, 
                brick_percent=brick_percent,
                touch_days=touch_days,
                touch_threshold=touch_threshold,
                is_light_mode=is_light_mode,
                market=market,
                max_k=max_kd,
                max_d=max_kd,
                require_k_gt_d=require_k_gt_d,
                require_touch=require_touch,
                min_red_bricks=min_red_bricks,
                green_brick_number=green_brick_number,
                require_no_earnings=require_no_earnings
            )
            yield f"data: {json.dumps(res)}\n\n"
            time.sleep(0.08)
        yield "data: __END__\n\n"

    response = Response(generate(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response

if __name__ == '__main__':
    print("Renko Pro Scanner v8 — http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)