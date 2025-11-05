"""
Stochastic RSI + Renko Stock Scanner (single file)

Install:
    pip install flask tradingview-ta pandas numpy yfinance openpyxl

Run:
    python scanner.py
Open:
    http://localhost:5000
"""
from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
import math
import time
import traceback
import yfinance as yf
from tradingview_ta import TA_Handler, Interval

app = Flask(__name__)

# ---- HTML template (keeps the UI you provided, plus Excel upload UI and JS) ----
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Stochastic RSI + Renko Scanner</title>
<style>
/* Paste your CSS here unchanged from original for exact look.
   For brevity in this message I keep it short — but in your file you can paste the full CSS you provided earlier. */
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial; background: #0f172a; color: #fff; padding: 20px; }
.container{max-width:1200px;margin:0 auto}
.header{ text-align:center;margin-bottom:20px;}
.btn{ background:linear-gradient(135deg,#3b82f6,#2563eb); color:white; padding:10px 16px; border-radius:6px; border:none; cursor:pointer }
.hidden{display:none}
.error{background:rgba(239,68,68,0.12);padding:12px;border-radius:6px;color:#fff;margin-bottom:12px}
.results{background:rgba(255,255,255,0.03);padding:16px;border-radius:10px;margin-top:12px}
.stock-card{padding:12px;border-radius:8px;margin-bottom:10px;background:rgba(255,255,255,0.02)}
.signal-badge{padding:6px 10px;border-radius:6px;font-weight:700}
.signal-badge.BUY{background:#10b981;color:#fff}
.signal-badge.SELL{background:#ef4444;color:#fff}
.signal-badge.NEUTRAL{background:#6b7280;color:#fff}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>📊 Stochastic RSI + Renko Scanner</h1>
    <p style="color:#aab">Renko-first-green + Stoch RSI K↑D entry only</p>
  </div>

  <div id="errorBox" class="error hidden"></div>

  <div>
    <label>Upload Excel (.xlsx) with tickers (column named "ticker" or first column):</label><br/>
    <input type="file" id="excelFile" accept=".xlsx"/>
    <button id="loadExcelBtn" class="btn" onclick="uploadExcel()">Upload & Load</button>
    <span id="excelStatus" style="margin-left:12px;color:#9aa;"></span>
  </div>

  <div style="margin-top:12px">
    <label>Or enter tickers (comma-separated):</label><br/>
    <textarea id="tickers" style="width:100%;height:80px">AAPL,MSFT,TSLA</textarea>
  </div>

  <div style="display:flex;gap:12px;margin-top:12px;">
    <button class="btn" onclick="scanStocks()">🔍 Scan Stocks</button>
    <div style="display:flex;gap:8px;align-items:center;">
      <label style="color:#aab">Renko brick ATR multiplier</label>
      <input id="brickAtrMult" value="1.0" style="width:70px;padding:6px;border-radius:6px;background:#0b1220;color:#fff;border:1px solid #233" />
    </div>
  </div>

  <div id="results" class="results hidden">
    <h3>Results (<span id="resultCount">0</span>)</h3>
    <div id="stocksList"></div>
  </div>
</div>

<script>
function showError(msg){ const el=document.getElementById('errorBox'); el.textContent=msg; el.classList.remove('hidden'); }
function hideError(){ document.getElementById('errorBox').classList.add('hidden'); }

async function uploadExcel(){
  hideError();
  const f = document.getElementById('excelFile').files[0];
  if(!f){ showError('Please choose a .xlsx file'); return; }
  const form = new FormData(); form.append('file', f);
  document.getElementById('excelStatus').textContent = 'Uploading...';
  try{
    const resp = await fetch('/api/upload_excel', { method:'POST', body: form });
    const data = await resp.json();
    if(!data.success){ showError(data.error || 'Upload failed'); document.getElementById('excelStatus').textContent=''; return; }
    const tickers = data.tickers.join(',');
    document.getElementById('tickers').value = tickers;
    document.getElementById('excelStatus').textContent = `Loaded ${data.tickers.length} tickers`;
  }catch(e){
    showError('Upload failed: '+e.message);
    document.getElementById('excelStatus').textContent='';
  }
}

async function scanStocks(){
  hideError();
  const raw = document.getElementById('tickers').value;
  const tickers = raw.split(',').map(s=>s.trim()).filter(s=>s);
  if(tickers.length===0){ showError('No tickers provided'); return; }
  const btn = event?.target;
  const brickAtrMult = parseFloat(document.getElementById('brickAtrMult').value) || 1.0;
  try{
    const resp = await fetch('/api/scan', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ tickers: tickers, params: { brick_atr_mult: brickAtrMult } })
    });
    const data = await resp.json();
    if(!data.success){ showError(data.error || 'Scan failed'); return; }
    displayResults(data.results);
  }catch(e){
    showError('Scan error: '+e.message);
  }
}

function displayResults(results){
  document.getElementById('results').classList.remove('hidden');
  document.getElementById('resultCount').textContent = results.length;
  const container = document.getElementById('stocksList');
  container.innerHTML = results.map(r => {
    const tags = [];
    if(r.renko_first_green) tags.push('<span style="background:#10b981;padding:4px;border-radius:4px;margin-right:6px">Renko-first-green</span>');
    if(r.stoch_k_cross) tags.push('<span style="background:#60a5fa;padding:4px;border-radius:4px">K↑D</span>');
    if(r.signal) tags.push('<span class="signal-badge '+r.signal+'">'+r.signal+'</span>');
    return `<div class="stock-card">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div><strong>${r.ticker}</strong> — ₹${r.price ?? 'N/A'}</div>
        <div>${tags.join(' ')}</div>
      </div>
      <div style="margin-top:8px;color:#aab">Renko bricks: last_dir=${r.renko_last_dir}, renko_first_green=${r.renko_first_green}</div>
      <div style="margin-top:4px;color:#aab">StochK=${r.stoch_k} StochD=${r.stoch_d} (Kprev=${r.stoch_k_prev} Dprev=${r.stoch_d_prev})</div>
      <div style="margin-top:6px;color:#aab">Score: ${r.score} &nbsp; Recommendation: ${r.recommendation}</div>
    </div>`;
  }).join('');
}
</script>
</body>
</html>
"""

# ------------------------
# Utility / Indicator Code
# ------------------------
def safe_float(x, default=0.0):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def atr_from_ohlc(df, period=14):
    """Compute ATR (true range average)."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return atr

def build_renko_from_ohlc(df, brick_size=None, atr_period=14, atr_mult=1.0):
    """
    Build simple Renko bricks from daily OHLC dataframe (pandas DataFrame with columns Open, High, Low, Close).
    Returns list of bricks: each brick is dict {'close':..., 'direction': 1 or -1}
    Direction 1 = up (green), -1 = down (red)
    brick_size: if None, use ATR(atr_period)*atr_mult as brick size
    """
    if df.shape[0] < 2:
        return []

    if brick_size is None:
        atr = atr_from_ohlc(df, period=atr_period)
        # use last ATR value as brick size (safeguard if NaN)
        last_atr = float(atr.iloc[-1]) if not math.isnan(float(atr.iloc[-1])) else 0.0
        brick_size = max(0.0001, last_atr * atr_mult) if last_atr > 0 else 0.01

    bricks = []
    last_brick_close = float(df['Close'].iloc[0])
    # We iterate through closes and create bricks when price moves enough
    for price in df['Close'].iloc[1:]:
        price = float(price)
        diff = price - last_brick_close
        # build as many bricks as fit
        while abs(diff) >= brick_size:
            direction = 1 if diff > 0 else -1
            last_brick_close = last_brick_close + brick_size * direction
            bricks.append({'close': last_brick_close, 'direction': direction})
            diff = price - last_brick_close
    return bricks

def calculate_rsi_series(prices, period=14):
    """Wilder's RSI producing full series (numpy array)."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < period + 1:
        return np.full(len(prices), 50.0)
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = np.sum(seed[seed > 0]) / period
    down = -np.sum(seed[seed < 0]) / period
    rs = up / down if down != 0 else math.inf if up > 0 else 0.0
    rsi = np.empty(len(prices))
    rsi[:period] = 100.0 - 100.0 / (1.0 + (rs if rs != 0 else 1.0))
    up_ema = up
    down_ema = down
    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        up_val = delta if delta > 0 else 0.0
        down_val = -delta if delta < 0 else 0.0
        up_ema = (up_ema * (period - 1) + up_val) / period
        down_ema = (down_ema * (period - 1) + down_val) / period
        rs = up_ema / down_ema if down_ema != 0 else math.inf if up_ema > 0 else 0.0
        rsi[i] = 100.0 - 100.0 / (1.0 + (rs if rs != 0 else 1.0))
    return rsi

def calculate_stoch_rsi(prices, rsi_period=14, stoch_length=14, k_period=3, d_period=3):
    """
    Calculate Stochastic RSI from price series (Close prices).
    Returns arrays: stoch_k, stoch_d where each is same length as input (first values may be NaN).
    stoch = (RSI - lowest_RSI) / (highest_RSI - lowest_RSI) * 100
    K is SMA of stoch values over k_period, D is SMA of K over d_period.
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < max(rsi_period, stoch_length) + 1:
        # not enough data, return constant mid-values
        l = len(prices)
        return (np.full(l, 50.0), np.full(l, 50.0))
    rsi = calculate_rsi_series(prices, period=rsi_period)
    stoch = np.full_like(rsi, np.nan)
    for i in range(stoch_length - 1, len(rsi)):
        window = rsi[i - stoch_length + 1: i + 1]
        low = np.nanmin(window)
        high = np.nanmax(window)
        if high - low == 0:
            stoch[i] = 50.0
        else:
            stoch[i] = 100.0 * (rsi[i] - low) / (high - low)
    # compute %K as SMA of stoch
    stoch_k = pd.Series(stoch).rolling(k_period, min_periods=1).mean().to_numpy()
    stoch_d = pd.Series(stoch_k).rolling(d_period, min_periods=1).mean().to_numpy()
    # fill any NaN with 50
    stoch_k = np.nan_to_num(stoch_k, 50.0)
    stoch_d = np.nan_to_num(stoch_d, 50.0)
    return stoch_k, stoch_d

# -------------------------
# Analysis function (uses yfinance + computed Renko/Stoch RSI)
# -------------------------
def analyze_ticker_local(symbol, exchange=''):
    """
    Attempt to fetch OHLC with yfinance and compute:
     - renko bricks
     - stoch RSI (K & D)
     - determine 'renko_first_green' and 'k cross over d' entry logic
    Returns dict with expected fields for frontend.
    """
    try:
        # Try multiple yfinance symbol forms: if user passed NSE:RELIANCE convert to RELIANCE.NS
        yf_symbol = symbol
        if ':' in symbol:
            exch, sym = symbol.split(':', 1)
            # Common mapping for NSE to yfinance: add .NS
            if exch.upper() in ('NSE', 'BSE') and not sym.upper().endswith('.NS'):
                yf_symbol = sym + '.NS'
            else:
                yf_symbol = sym
        # fetch up to 120 days to be safe
        df = yf.Ticker(yf_symbol).history(period='120d', interval='1d', actions=False)
        if df is None or df.shape[0] < 20:
            raise ValueError('Insufficient OHLC data from yfinance')

        # ensure proper columns
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
        # compute ATR & renko
        atr = atr_from_ohlc(df, period=14)
        brick_size = float(atr.iloc[-1]) if not math.isnan(float(atr.iloc[-1])) else 0.01
        # allow external multiplier passed via params; but default 1.0
        # We'll accept params via global var if needed; but for now brick_size derived here
        bricks = build_renko_from_ohlc(df, brick_size=brick_size, atr_period=14, atr_mult=1.0)

        last_dir = bricks[-1]['direction'] if bricks else 0
        # Determine if the last brick is the FIRST green after red sequence:
        renko_first_green = False
        if len(bricks) >= 1 and last_dir == 1:
            # if previous brick exists and was down OR if there are multiple down bricks before
            if len(bricks) == 1:
                # only one brick -> treat as first up
                renko_first_green = True
            else:
                # check previous brick direction; require immediate prior brick be down (-1)
                if bricks[-2]['direction'] == -1:
                    renko_first_green = True

        # compute stoch rsi
        closes = df['Close'].to_numpy(dtype=float)
        stoch_k, stoch_d = calculate_stoch_rsi(closes, rsi_period=14, stoch_length=14, k_period=3, d_period=3)
        # use last and previous values to detect a CROSS (prev K <= prev D, last K > last D)
        last_k = float(stoch_k[-1])
        last_d = float(stoch_d[-1])
        prev_k = float(stoch_k[-2]) if len(stoch_k) >= 2 else last_k
        prev_d = float(stoch_d[-2]) if len(stoch_d) >= 2 else last_d
        stoch_cross = (prev_k <= prev_d) and (last_k > last_d)

        # price (last close)
        price = float(closes[-1])

        # score / signal logic: require both conditions to be BUY for strong buy
        is_oversold = (last_k < 20 and last_d < 20)
        is_overbought = (last_k > 80 and last_d > 80)

        score = 50
        if renko_first_green and stoch_cross and is_oversold:
            score = 100
        elif renko_first_green and stoch_cross:
            score = 95
        elif is_oversold and stoch_cross:
            score = 85
        elif stoch_cross:
            score = 70
        elif is_overbought and (prev_k >= prev_d and last_k < last_d):
            score = 20

        signal = 'BUY' if score >= 70 else 'SELL' if score <= 30 else 'NEUTRAL'

        # build response object
        return {
            'success': True,
            'ticker': symbol,
            'price': round(price, 2),
            'stoch_k': round(last_k, 2),
            'stoch_d': round(last_d, 2),
            'stoch_k_prev': round(prev_k, 2),
            'stoch_d_prev': round(prev_d, 2),
            'renko_last_dir': int(last_dir),
            'renko_first_green': bool(renko_first_green),
            'stoch_k_cross': bool(stoch_cross),
            'is_oversold': bool(is_oversold),
            'is_overbought': bool(is_overbought),
            'score': int(score),
            'signal': signal,
            'recommendation': 'BUY' if score >= 70 else 'SELL' if score <= 30 else 'NEUTRAL'
        }

    except Exception as e:
        # fallback: try tradingview_ta analysis (keeps previous behavior)
        try:
            handler = TA_Handler(symbol=symbol, exchange='NSE', screener='india', interval=Interval.INTERVAL_1_DAY, timeout=10)
            analysis = handler.get_analysis()
            indicators = getattr(analysis, 'indicators', {}) or {}
            price = safe_float(indicators.get('close', indicators.get('Close', 0)))
            stoch_k = safe_float(indicators.get('Stoch.K', indicators.get('STOCHK', 50)))
            stoch_d = safe_float(indicators.get('Stoch.D', indicators.get('STOCHD', 50)))
            # approximate the cross by comparing current k/d (no historic)
            stoch_cross = stoch_k > stoch_d
            is_oversold = stoch_k < 20 and stoch_d < 20
            score = 95 if is_oversold and stoch_cross else 75 if is_oversold else 65 if stoch_cross else 50
            signal = 'BUY' if score >= 65 else 'NEUTRAL'
            return {
                'success': True,
                'ticker': symbol,
                'price': round(price, 2),
                'stoch_k': round(stoch_k, 2),
                'stoch_d': round(stoch_d, 2),
                'stoch_k_prev': None,
                'stoch_d_prev': None,
                'renko_last_dir': 0,
                'renko_first_green': False,
                'stoch_k_cross': bool(stoch_cross),
                'is_oversold': bool(is_oversold),
                'is_overbought': bool(stoch_k > 80 and stoch_d > 80),
                'score': int(score),
                'signal': signal,
                'recommendation': getattr(analysis, 'summary', {}).get('RECOMMENDATION', 'NEUTRAL')
            }
        except Exception as e2:
            tb = traceback.format_exc()
            return { 'success': False, 'error': str(e2), 'traceback': tb, 'ticker': symbol }

# -------------------------
# Flask routes
# -------------------------
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/upload_excel', methods=['POST'])
def upload_excel():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        f = request.files['file']
        # read excel into pandas
        df = pd.read_excel(f, engine='openpyxl')
        if df.empty:
            return jsonify({'success': False, 'error': 'Excel file is empty'}), 400
        # If there's column named 'ticker' (case-insensitive) use it
        cols = [c.lower() for c in df.columns]
        if 'ticker' in cols:
            tick_col = df.columns[cols.index('ticker')]
            tickers = df[tick_col].dropna().astype(str).str.strip().tolist()
        else:
            # fallback: use first column
            tickers = df.iloc[:,0].dropna().astype(str).str.strip().tolist()
        # normalize tickers: replace spaces & common separators
        tickers = [t.replace(' ', '').replace('\\n','').strip() for t in tickers if t and str(t).strip() != '']
        return jsonify({'success': True, 'tickers': tickers})
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({'success': False, 'error': str(e), 'traceback': tb}), 500

@app.route('/api/scan', methods=['POST'])
def api_scan():
    try:
        data = request.json or {}
        tickers = data.get('tickers', [])
        params = data.get('params', {}) or {}
        results = []
        # brick atr multiplier optional param
        brick_atr_mult = float(params.get('brick_atr_mult', 1.0))
        for t in tickers:
            t = t.strip()
            if not t:
                continue
            # analyze with local computation
            res = analyze_ticker_local(t)
            # If we want to apply external brick_atr_mult, recompute bricks with multiplier (simple approach: re-run compute)
            try:
                # best effort: if data available we recompute bricks with multiplier using yfinance
                yf_sym = t
                if ':' in t:
                    exch, sym = t.split(':',1)
                    if exch.upper() in ('NSE','BSE') and not sym.upper().endswith('.NS'):
                        yf_sym = sym + '.NS'
                    else:
                        yf_sym = sym
                df = yf.Ticker(yf_sym).history(period='120d', interval='1d', actions=False)
                if df is not None and df.shape[0] > 10:
                    atr = atr_from_ohlc(df, period=14)
                    base_brick = float(atr.iloc[-1]) if not math.isnan(float(atr.iloc[-1])) else 0.01
                    bricks = build_renko_from_ohlc(df, brick_size=base_brick * brick_atr_mult, atr_period=14, atr_mult=1.0)
                    last_dir = bricks[-1]['direction'] if bricks else 0
                    renko_first_green = False
                    if len(bricks) >= 1 and last_dir == 1:
                        if len(bricks) == 1:
                            renko_first_green = True
                        else:
                            if bricks[-2]['direction'] == -1:
                                renko_first_green = True
                    res['renko_last_dir'] = int(last_dir)
                    res['renko_first_green'] = bool(renko_first_green)
            except Exception:
                pass

            results.append(res)
            # polite delay
            time.sleep(0.2)

        # optionally sort BUYs first
        results.sort(key=lambda r: (0 if r.get('signal')=='BUY' else 1, r.get('score', 0)), reverse=False)
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({'success': False, 'error': str(e), 'traceback': tb}), 500

if __name__ == '__main__':
    print("Starting scanner on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
