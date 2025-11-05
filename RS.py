"""
Stochastic RSI + Renko Scanner — FINAL WITH COMBINED CHART
Top: RENKO CANDLES (Green/Red)
Bottom: StochRSI (K/D)
BUY = ALL 3 CONDITIONS
"""

from flask import Flask, render_template_string, request, jsonify, Response
import pandas as pd
import numpy as np
import time
import traceback
import datetime
import yfinance as yf
import uuid
import json
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from tradingview_ta import TA_Handler, Interval

app = Flask(__name__)

# ----------------------------------------------------------------------
# Cache (5 min)
# ----------------------------------------------------------------------
CACHE = {}

def get_cached_data(symbol, period='120d'):
    key = f"{symbol}_{period}"
    now = datetime.datetime.now()
    if key in CACHE:
        data, ts = CACHE[key]
        if now - ts < datetime.timedelta(minutes=5):
            return data
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval='1d', actions=False)
    CACHE[key] = (df, now)
    return df

# ----------------------------------------------------------------------
# Helper: Extract Renko closes
# ----------------------------------------------------------------------
def renko_closes(bricks):
    if len(bricks) < 2:
        return np.array([], dtype=float)
    return np.array([b['close'] for b in bricks], dtype=float)

# ----------------------------------------------------------------------
# Stoch RSI (Pine v6 Match)
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
        low_rsi  = np.nanmin(rsi_window)
        high_rsi = np.nanmax(rsi_window)
        cur_rsi  = rsi[i]
        if high_rsi > low_rsi:
            raw_k[i] = 100.0 * (cur_rsi - low_rsi) / (high_rsi - low_rsi)
        else:
            raw_k[i] = 50.0

    k_series = pd.Series(raw_k).rolling(window=k_sma, min_periods=1).mean()
    d_series = k_series.rolling(window=d_sma, min_periods=1).mean()
    return np.nan_to_num(k_series.to_numpy(), nan=50.0), np.nan_to_num(d_series.to_numpy(), nan=50.0)

# ----------------------------------------------------------------------
# COMBINED CHART: Renko (Top) + StochRSI (Bottom)
# ----------------------------------------------------------------------
def generate_combined_chart(bricks, width=800, height=500):
    if len(bricks) < 2:
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        plt.text(0.5, 0.5, 'Not enough data', ha='center', va='center', fontsize=12)
        plt.axis('off')
    else:
        recent_bricks = bricks[-60:] if len(bricks) >= 60 else bricks
        renko_closes = [b['close'] for b in recent_bricks]
        opens = [renko_closes[0]] + renko_closes[:-1]
        closes = renko_closes
        highs = [max(o, c) for o, c in zip(opens, closes)]
        lows = [min(o, c) for o, c in zip(opens, closes)]
        colors = ['green' if c >= o else 'red' for o, c in zip(opens, closes)]

        # StochRSI
        stoch_k, stoch_d = calculate_stoch_rsi(renko_closes)

        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width/100, height/100), dpi=100, gridspec_kw={'height_ratios': [3, 2]})

        # --- RENKO ---
        for i in range(len(recent_bricks)):
            ax1.plot([i, i], [lows[i], highs[i]], color=colors[i], linewidth=6)
            ax1.plot([i-0.3, i+0.3], [opens[i], opens[i]], color=colors[i], linewidth=6)
            ax1.plot([i-0.3, i+0.3], [closes[i], closes[i]], color=colors[i], linewidth=6)
        ax1.set_title('Renko Bricks + StochRSI — Click to Enlarge', fontsize=14, pad=20)
        ax1.set_ylabel('Price', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks([])

        # --- StochRSI ---
        x = range(len(stoch_k))
        ax2.plot(x, stoch_k, color='blue', linewidth=2, label='K')
        ax2.plot(x, stoch_d, color='orange', linewidth=2, label='D')
        ax2.axhline(y=80, color='red', linestyle='--', alpha=0.6)
        ax2.axhline(y=40, color='purple', linestyle='--', alpha=0.6)
        ax2.axhline(y=20, color='green', linestyle='--', alpha=0.6)
        ax2.axhline(y=5, color='gold', linestyle='--', alpha=0.6)
        ax2.set_ylabel('StochRSI', fontsize=10)
        ax2.set_xlabel('Bricks ago', fontsize=10)
        ax2.set_ylim(0, 100)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
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

def atr_from_ohlc(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

def build_renko_from_ohlc(df, brick_size):
    if df.shape[0] < 2: return []
    bricks = []
    last_close = float(df['Close'].iloc[0])
    for price in df['Close'].iloc[1:]:
        price = float(price)
        diff = price - last_close
        while abs(diff) >= brick_size:
            direction = 1 if diff > 0 else -1
            last_close += brick_size * direction
            bricks.append({'close': last_close, 'direction': direction})
            diff = price - last_close
    return bricks

# ----------------------------------------------------------------------
# Core Analysis: ALL 3 + COMBINED CHART
# ----------------------------------------------------------------------
def analyze_ticker_local(symbol, brick_atr_mult=1.0, touch_days=30, touch_threshold=5.0):
    try:
        yf_sym = symbol.upper()
        if not yf_sym.endswith('.NS') and currentMarket == 'india':
            yf_sym += '.NS'
        df = get_cached_data(yf_sym)
        if df is None or df.shape[0] < 20:
            raise ValueError('Insufficient data')

        df = df.rename(columns={c: c.capitalize() for c in df.columns})

        # --- Build Renko ---
        atr = atr_from_ohlc(df, period=14)
        last_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.01
        brick_size = max(0.0001, last_atr * brick_atr_mult)
        bricks = build_renko_from_ohlc(df, brick_size=brick_size)
        if len(bricks) < 30:
            raise ValueError('Not enough Renko bricks')

        # --- Condition 1: Renko Reversal ---
        last_dir = bricks[-1]['direction']
        renko_first_green = last_dir == 1 and any(b['direction'] == -1 for b in bricks[:-1])

        # --- StochRSI ---
        renko_close_prices = renko_closes(bricks)
        stoch_k, stoch_d = calculate_stoch_rsi(renko_close_prices)
        last_k, last_d = float(stoch_k[-1]), float(stoch_d[-1])

        # --- Condition 2: K > D & Both < 40 ---
        k_bullish = last_k < 40 and last_d < 40 and last_k > last_d
        is_oversold = last_k < 40 and last_d < 40

        # --- Condition 3: Touched ≤5 ---
        recent_k = stoch_k[-touch_days:] if len(stoch_k) >= touch_days else stoch_k
        recent_d = stoch_d[-touch_days:] if len(stoch_d) >= touch_days else stoch_d
        touched_five_recent = any(k <= touch_threshold for k in recent_k) or any(d <= touch_threshold for d in recent_d)

        # --- RDI ---
        recent_bricks = bricks[-touch_days:] if len(bricks) >= touch_days else bricks
        green_count = sum(1 for b in recent_bricks if b['direction'] == 1)
        rdi = round((green_count / len(recent_bricks)) * 100, 1) if recent_bricks else 0

        # --- ALL 3 REQUIRED ---
        all_met = renko_first_green and k_bullish and touched_five_recent
        score = 100 if all_met else 50
        signal = 'BUY' if all_met else 'NEUTRAL'

        # --- Tooltips ---
        k_d_tooltip = f"K={last_k:.1f} | D={last_d:.1f}"
        touched_tooltip = "Touched ≤5.0" if touched_five_recent else "Not touched"
        renko_tooltip = f"First green after red | RDI={rdi}%"

        # --- COMBINED CHART ---
        chart_preview = generate_combined_chart(bricks)

        return {
            'success': True,
            'ticker': symbol,
            'price': round(float(bricks[-1]['close']), 2),
            'brick_size': round(brick_size, 4),
            'stoch_k': round(last_k, 2),
            'stoch_d': round(last_d, 2),
            'renko_first_green': renko_first_green,
            'k_bullish': k_bullish,
            'is_oversold': is_oversold,
            'touched_five_recent': touched_five_recent,
            'rdi': rdi,
            'score': score,
            'signal': signal,
            'recommendation': signal,
            'chart_preview': chart_preview,
            'tooltips': {
                'renko': renko_tooltip,
                'k_d': k_d_tooltip,
                'touched': touched_tooltip
            }
        }

    except Exception as e:
        try:
            exchange = 'NSE' if symbol.endswith('.NS') else 'NASDAQ'
            screener = 'india' if symbol.endswith('.NS') else 'america'
            handler = TA_Handler(symbol=symbol, exchange=exchange, screener=screener,
                                 interval=Interval.INTERVAL_1_DAY, timeout=10)
            analysis = handler.get_analysis()
            ind = analysis.indicators
            price = safe_float(ind.get('close') or ind.get('Close'))
            k = safe_float(ind.get('Stoch.K'))
            d = safe_float(ind.get('Stoch.D'))
            k_bullish = k < 40 and d < 40 and k > d
            touched_five_recent = k <= 5.0 or d <= 5.0
            all_met = k_bullish and touched_five_recent
            score = 100 if all_met else 50
            signal = 'BUY' if all_met else 'NEUTRAL'
            return {
                'success': True,
                'ticker': symbol,
                'price': round(price, 2),
                'brick_size': 0,
                'stoch_k': round(k, 2),
                'stoch_d': round(d, 2),
                'renko_first_green': False,
                'k_bullish': k_bullish,
                'is_oversold': k < 40 and d < 40,
                'touched_five_recent': touched_five_recent,
                'rdi': 0,
                'score': score,
                'signal': signal,
                'recommendation': analysis.summary.get('RECOMMENDATION', 'NEUTRAL'),
                'chart_preview': '',
                'tooltips': {
                    'renko': 'Renko not available',
                    'k_d': f"K={k:.1f} | D={d:.1f}",
                    'touched': "Touched ≤5.0" if touched_five_recent else "Not touched"
                }
            }
        except Exception as e2:
            tb = traceback.format_exc()
            return {'success': False, 'error': str(e2), 'traceback': tb, 'ticker': symbol}

# ----------------------------------------------------------------------
# HTML + JS
# ----------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Stochastic RSI + Renko Scanner</title>
<style>
* { box-sizing: border-box; margin:0; padding:0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial; background:#0f172a; color:#fff; padding:20px; }
.container { max-width:1800px; margin:0 auto; }
.header { text-align:center; margin-bottom:20px; }
.btn { background:linear-gradient(135deg,#3b82f6,#2563eb); color:white; padding:8px 14px; border-radius:6px; border:none; cursor:pointer; font-size:0.9rem; }
.toggle-btn { background:#1e293b; border:2px solid #3b82f6; padding:6px 12px; border-radius:8px; font-weight:600; }
.toggle-btn.active { background:#3b82f6; }
.hidden { display:none; }
.error { background:rgba(239,68,68,0.12); padding:12px; border-radius:6px; color:#fff; margin-bottom:12px; }
.results { background:rgba(255,255,255,0.03); padding:16px; border-radius:10px; margin-top:12px; overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-size:0.9rem; }
th, td { padding:10px 8px; text-align:left; border-bottom:1px solid rgba(255,255,255,0.1); }
th { background:rgba(255,255,255,0.05); font-weight:600; }
tr:hover { background:rgba(255,255,255,0.04); }
tr.buy-row { background:rgba(16,185,129,0.15); }
.signal-badge { padding:4px 8px; border-radius:6px; font-weight:700; font-size:0.8rem; }
.signal-badge.BUY { background:#10b981; color:#fff; }
.signal-badge.NEUTRAL { background:#6b7280; color:#fff; }
.preview-img { width:120px; height:60px; border-radius:4px; border:1px solid #333; cursor:zoom-in; transition:0.2s; }
.preview-img:hover { opacity:0.8; }
.spinner { border:4px solid #1e293b; border-top:4px solid #3b82f6; border-radius:50%; width:28px; height:28px; animation:spin 1s linear infinite; margin:20px auto; }
@keyframes spin { to { transform:rotate(360deg); } }
.modal { display:none; position:fixed; z-index:1000; left:0; top:0; width:100%; height:100%; background:rgba(0,0,0,0.9); justify-content:center; align-items:center; opacity:0; transition:opacity 0.3s; }
.modal img { max-width:90%; max-height:90%; border-radius:8px; box-shadow:0 0 30px rgba(0,0,0,0.6); }
.modal .close { position:absolute; top:20px; right:30px; color:#fff; font-size:40px; font-weight:bold; cursor:pointer; }
.tooltip { position:relative; display:inline-block; }
.tooltip .tooltiptext {
  visibility: hidden; width: 180px; background-color: #1e293b; color: #fff; text-align: center;
  border-radius: 6px; padding: 6px; position: absolute; z-index: 1; bottom: 125%; left: 50%;
  margin-left: -90px; opacity: 0; transition: opacity 0.3s; font-size: 0.8rem;
}
.tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>Stochastic RSI + Renko Scanner</h1>
    <p style="color:#aab">BUY = ALL 3: Green Brick + K>D & Both<40 + Touched≤5</p>
    <p style="color:#aab">Preview: Renko (Top) + StochRSI (Bottom)</p>
  </div>

  <div id="errorBox" class="error hidden"></div>

  <div style="margin-bottom:12px;">
    <label>Upload Excel (.xlsx) – column "ticker" or first column:</label><br/>
    <input type="file" id="excelFile" accept=".xlsx"/>
    <button class="btn" id="uploadBtn">Upload & Load</button>
    <span id="excelStatus" style="margin-left:12px;color:#9aa;"></span>
  </div>

  <div style="margin-bottom:12px;">
    <label>Or enter tickers (comma-separated):</label><br/>
    <textarea id="tickers" style="width:100%;height:80px">RELIANCE,HDFCBANK,TCS,HAVELLS</textarea>
  </div>

  <div style="display:flex;gap:12px;align-items:center;">
    <button class="btn" id="scanBtn">Scan Stocks</button>
    <label style="color:#aab">Renko brick ATR ×</label>
    <input id="brickAtrMult" value="1.0" style="width:70px;padding:6px;border-radius:6px;background:#0b1220;color:#fff;border:1px solid #233"/>
    <label style="color:#aab">Touch bricks</label>
    <input id="touchDays" value="30" style="width:60px;padding:6px;border-radius:6px;background:#0b1220;color:#fff;border:1px solid #233"/>
    <label style="color:#aab">Touch threshold</label>
    <input id="touchThreshold" value="5.0" style="width:60px;padding:6px;border-radius:6px;background:#0b1220;color:#fff;border:1px solid #233"/>
    
    <div style="margin-left:20px;display:flex;gap:6px;">
      <button id="toggleIndia" class="toggle-btn active">India</button>
      <button id="toggleUSA" class="toggle-btn">USA</button>
    </div>
  </div>

  <div id="spinner" class="spinner hidden"></div>

  <div id="results" class="results hidden">
    <h3>Results (<span id="resultCount">0</span>) — Market: <span id="marketLabel">India</span></h3>
    <table id="resultsTable">
      <thead>
        <tr>
          <th>Ticker</th>
          <th>Combined Chart</th>
          <th>Price</th>
          <th>Brick</th>
          <th>StochK</th>
          <th>StochD</th>
          <th>Signal</th>
          <th>Score</th>
          <th>Renko</th>
          <th>K>D & Both<40</th>
          <th>Touched 5?</th>
        </tr>
      </thead>
      <tbody id="tableBody"></tbody>
    </table>
    <button class="btn" id="exportBtn" style="margin-top:12px;">Export CSV</button>
  </div>
</div>

<div id="chartModal" class="modal">
  <span class="close">X</span>
  <img id="modalImage" src="" alt="Full Combined Chart"/>
</div>

<script>
let eventSource = null;
let results = [];
let currentMarket = 'india';

function showError(msg){ const el=document.getElementById('errorBox'); el.textContent=msg; el.classList.remove('hidden'); }
function hideError(){ document.getElementById('errorBox').classList.add('hidden'); }
function showSpinner(){ document.getElementById('spinner').classList.remove('hidden'); }
function hideSpinner(){ document.getElementById('spinner').classList.add('hidden'); }

document.getElementById('toggleIndia').onclick = () => setMarket('india');
document.getElementById('toggleUSA').onclick = () => setMarket('usa');
document.getElementById('uploadBtn').onclick = uploadExcel;
document.getElementById('scanBtn').onclick = scanStocks;
document.getElementById('exportBtn').onclick = exportCSV;
document.querySelector('.modal').onclick = closeModal;
document.querySelector('.close').onclick = closeModal;

function setMarket(market){
  currentMarket = market;
  document.getElementById('toggleIndia').classList.toggle('active', market==='india');
  document.getElementById('toggleUSA').classList.toggle('active', market==='usa');
  document.getElementById('marketLabel').textContent = market === 'india' ? 'India (.NS)' : 'USA';
}

async function uploadExcel(){
  hideError(); hideSpinner();
  const f = document.getElementById('excelFile').files[0];
  if(!f){ showError('Choose .xlsx'); return; }
  const form = new FormData(); form.append('file', f);
  document.getElementById('excelStatus').textContent='Uploading...';
  try{
    const resp = await fetch('/api/upload_excel', {method:'POST', body:form});
    const data = await resp.json();
    if(!data.success){ showError(data.error); document.getElementById('excelStatus').textContent=''; return; }
    document.getElementById('tickers').value = data.tickers.join(',');
    document.getElementById('excelStatus').textContent = `Loaded ${data.tickers.length} tickers`;
  }catch(e){ showError('Upload error: '+e.message); document.getElementById('excelStatus').textContent=''; }
}

async function scanStocks(){
  hideError(); hideSpinner();
  let raw = document.getElementById('tickers').value.trim();
  let tickers = raw.split(',').map(s=>s.trim()).filter(s=>s);
  if(!tickers.length){ showError('No tickers'); return; }

  const brickAtrMult = parseFloat(document.getElementById('brickAtrMult').value)||1.0;
  const touchDays = parseInt(document.getElementById('touchDays').value)||30;
  const touchThreshold = parseFloat(document.getElementById('touchThreshold').value)||5.0;

  if(currentMarket === 'india'){
    tickers = tickers.map(t => t.toUpperCase().endsWith('.NS') ? t : t + '.NS');
  }

  document.getElementById('tableBody').innerHTML = '';
  results = [];
  document.getElementById('resultCount').textContent = '0';
  document.getElementById('results').classList.remove('hidden');

  if(eventSource) eventSource.close();
  showSpinner();

  const startResp = await fetch('/api/scan_start', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({tickers, market: currentMarket, params:{brick_atr_mult:brickAtrMult, touch_days:touchDays, touch_threshold:touchThreshold}})
  });
  const startData = await startResp.json();
  if(!startData.success){ showError(startData.error); hideSpinner(); return; }

  eventSource = new EventSource(`/api/scan_stream?token=${startData.token}`);

  eventSource.onmessage = function(e){
    if(e.data === '__END__'){
      eventSource.close();
      hideSpinner();
      sortTable();
      return;
    }
    let res;
    try { res = JSON.parse(e.data); } catch(err) { return; }
    if(!res.success){ showError(res.error||'Error'); return; }
    results.push(res);
    document.getElementById('resultCount').textContent = results.length;
    appendRow(res);
  };

  eventSource.onerror = function(){
    eventSource.close();
    hideSpinner();
    showError('Stream error');
  };
}

function appendRow(r){
  const tbody = document.getElementById('tableBody');
  const tr = document.createElement('tr');
  if(r.signal === 'BUY') tr.classList.add('buy-row');

  const img = r.chart_preview ? `<img src="${r.chart_preview}" class="preview-img" alt="Combined Chart" onclick="openModal('${r.chart_preview.replace(/'/g, "\\'")}')">` : '';

  const renkoCell = `<div class="tooltip">${r.renko_first_green ? 'Yes' : 'No'}<span class="tooltiptext">${r.tooltips.renko}</span></div>`;
  const kdCell = `<div class="tooltip">${r.k_bullish ? 'Yes' : 'No'}<span class="tooltiptext">${r.tooltips.k_d}</span></div>`;
  const touchedCell = `<div class="tooltip">${r.touched_five_recent ? 'Yes' : 'No'}<span class="tooltiptext">${r.tooltips.touched}</span></div>`;

  tr.innerHTML = `
    <td><strong>${r.ticker}</strong></td>
    <td>${img}</td>
    <td>₹${r.price}</td>
    <td>₹${r.brick_size}</td>
    <td>${r.stoch_k}</td>
    <td>${r.stoch_d}</td>
    <td><span class="signal-badge ${r.signal}">${r.signal}</span></td>
    <td>${r.score}</td>
    <td>${renkoCell}</td>
    <td>${kdCell}</td>
    <td>${touchedCell}</td>
  `;
  tbody.appendChild(tr);
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
  const rows = Array.from(tbody.rows);
  rows.sort((a,b) => {
    const sa = a.cells[6].querySelector('.signal-badge').textContent;
    const sb = b.cells[6].querySelector('.signal-badge').textContent;
    const scoreA = parseInt(a.cells[7].textContent);
    const scoreB = parseInt(b.cells[7].textContent);
    if(sa === 'BUY' && sb !== 'BUY') return -1;
    if(sb === 'BUY' && sa !== 'BUY') return 1;
    return scoreB - scoreA;
  });
  tbody.innerHTML = '';
  rows.forEach(r => tbody.appendChild(r));
}

function exportCSV(){
  const headers = ['Ticker','Price','BrickSize','StochK','StochD','Signal','Score','RenkoFirstGreen','KBullish','Touched5','RDI'];
  const rows = results.map(r => [
    r.ticker, r.price, r.brick_size, r.stoch_k, r.stoch_d, r.signal, r.score,
    r.renko_first_green ? 'YES' : 'NO',
    r.k_bullish ? 'YES' : 'NO',
    r.touched_five_recent ? 'YES' : 'NO',
    r.rdi
  ]);
  const csv = [headers, ...rows].map(r => r.join(',')).join('\n');
  const blob = new Blob([csv], {type: 'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = 'renko_stoch_scan.csv'; a.click();
}
</script>
</body>
</html>"""

# ----------------------------------------------------------------------
# Flask Routes
# ----------------------------------------------------------------------
STREAM_TOKENS = {}
currentMarket = 'india'

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
    global currentMarket
    data = request.get_json() or {}
    currentMarket = data.get('market', 'india')
    token = str(uuid.uuid4())
    STREAM_TOKENS[token] = data
    return jsonify({'success': True, 'token': token})

@app.route('/api/scan_stream')
def scan_stream():
    token = request.args.get('token')
    payload = STREAM_TOKENS.pop(token, None)
    if not payload:
        return "Invalid token", 400

    tickers = payload.get('tickers', [])
    params = payload.get('params', {}) or {}
    brick_atr_mult = float(params.get('brick_atr_mult', 1.0))
    touch_days = int(params.get('touch_days', 30))
    touch_threshold = float(params.get('touch_threshold', 5.0))

    def generate():
        for t in tickers:
            t = t.strip()
            if not t: continue
            res = analyze_ticker_local(t, brick_atr_mult=brick_atr_mult, touch_days=touch_days, touch_threshold=touch_threshold)
            yield f"data: {json.dumps(res)}\n\n"
            time.sleep(0.22)
        yield "data: __END__\n\n"

    response = Response(generate(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response

if __name__ == '__main__':
    print("Starting Renko + StochRSI Scanner → http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)