import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime
import smtplib
import os
import time
import warnings
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
# Use Environment Variables or replace strings below
GMAIL_USER         = os.environ.get("GMAIL_USER", "your_email@gmail.com")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "your_16_digit_app_password")
EMAIL_RECIPIENT    = os.environ.get("EMAIL_RECIPIENT", "your_destination_email@gmail.com")

RISK_FREE_RATE     = 0.05
HAIRCUT_MULTIPLIER = 0.60  # 40% reduction from MID price for ultra-realistic fills
MIN_CW_RATIO       = 0.20  # Min Credit/Width ratio after the 40% haircut

def get_sp500_tickers():
    """Pull live S&P 500 tickers from Wikipedia."""
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        tickers = tables[0]['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    except Exception as e:
        print(f"Wikipedia fetch failed: {e}. Using fallback ticker list.")
        return ['NVDA','AMD','QCOM','AAPL','MSFT','GOOGL','AMZN','ABNB','CRM','OXY','TSLA','META','NFLX','GOOG']

# --- OPTION MATH (BLACK-SCHOLES) ---

def bs_put_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_put_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return None
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1) - 1
    except: return None

def implied_vol(market_price, S, K, T, r):
    if T <= 0 or market_price <= 0: return None
    try:
        def f(sigma): return bs_put_price(S, K, T, r, sigma) - market_price
        return brentq(f, 0.01, 5.0, xtol=1e-5)
    except: return None

# --- VOLATILITY & EARNINGS ---

def get_hv_and_ivr(ticker_obj, window=30):
    """Calculates 30-day Historical Volatility and IV Rank."""
    try:
        hist = ticker_obj.history(period='1y')
        if len(hist) < window + 10: return None, None
        returns = hist['Close'].pct_change().dropna()
        hv_current = returns.tail(window).std() * np.sqrt(252)
        rolling_hv = returns.rolling(window).std().dropna() * np.sqrt(252)
        ivr = (hv_current - rolling_hv.min()) / (rolling_hv.max() - rolling_hv.min()) * 100
        return hv_current, round(ivr, 1)
    except: return None, None

def get_earnings_info(ticker_obj):
    """Robust fetch for next earnings date with yfinance handling."""
    try:
        cal = ticker_obj.calendar
        if cal is None or (isinstance(cal, pd.DataFrame) and cal.empty):
            return None, None
        
        # Determine format of calendar output
        if isinstance(cal, dict):
            earn_date = cal.get('Earnings Date', [None])[0]
        else:
            earn_date = cal.columns[0] if hasattr(cal, 'columns') else cal.iloc[0,0]
        
        if hasattr(earn_date, 'date'):
            earn_date = earn_date.date()
        elif isinstance(earn_date, str):
            earn_date = datetime.strptime(earn_date, '%Y-%m-%d').date()
            
        today = datetime.today().date()
        days_to = (earn_date - today).days
        
        if days_to < 0:
            return earn_date, f"JUST REPORTED ({abs(days_to)}d ago)"
        return earn_date, f"EARNINGS IN {days_to}d"
    except:
        return None, None

def get_atm_iv(puts, S, T, r):
    try:
        p = puts.copy()
        p['dist'] = abs(p['strike'] - S)
        atm = p.nsmallest(1, 'dist').iloc[0]
        mid = (atm['bid'] + atm['ask']) / 2
        return implied_vol(mid, S, atm['strike'], T, r)
    except: return None

# --- SPREAD FINDER ---

def find_best_spread(S, iv_for_delta, T, puts):
    """Scans for best strike pairs meeting Delta and C/W requirements."""
    for _, put in puts.iterrows():
        K_short = put['strike']
        if put['bid'] <= 0 or put['ask'] <= 0: continue
        short_mid = (put['bid'] + put['ask']) / 2
        
        delta = bs_put_delta(S, K_short, T, RISK_FREE_RATE, iv_for_delta)
        if delta is None or not (0.22 <= abs(delta) <= 0.40): continue

        for width in [2, 3, 5, 10]:
            K_long = K_short - width
            long_rows = puts[puts['strike'] == K_long]
            if long_rows.empty or long_rows.iloc[0]['bid'] <= 0: continue
            
            long_mid = (long_rows.iloc[0]['bid'] + long_rows.iloc[0]['ask']) / 2
            credit_raw = short_mid - long_mid
            
            # --- APPLY 40% HAIRCUT ---
            credit = round(credit_raw * HAIRCUT_MULTIPLIER, 2)
            credit_ratio = credit / width
            breakeven = K_short - credit
            buffer_pct = (S - breakeven) / S * 100

            if credit_ratio >= MIN_CW_RATIO and buffer_pct >= 5:
                return {
                    'short_strike': K_short, 'long_strike': K_long,
                    'width': width, 'delta': round(abs(delta), 2),
                    'credit': credit, 'credit_ratio': round(credit_ratio * 100, 1),
                    'breakeven': round(breakeven, 2), 'buffer': round(buffer_pct, 1),
                    'max_profit': int(credit * 100), 'max_loss': int((width - credit) * 100)
                }
    return None

# --- MAIN SCREENER LOGIC ---

def run_screen():
    today = datetime.today()
    results = []
    tickers = get_sp500_tickers()
    
    print(f"Starting Screen: {len(tickers)} tickers.")
    print(f"Params: 40% Haircut | Min 20% C/W | Delta 0.22-0.40 | EARNINGS FILTER: ON")

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period='5d')
            if hist.empty: continue
            S = hist['Close'].iloc[-1]

            hv, ivr = get_hv_and_ivr(t)
            if ivr is None or ivr < 50: continue

            exps = t.options
            target_exp = None
            for exp in exps:
                dte = (datetime.strptime(exp, '%Y-%m-%d') - today).days
                if 28 <= dte <= 50:
                    target_exp, target_dte = exp, dte
                    break
            if not target_exp: continue

            # --- EARNINGS HARD FILTER ---
            earn_date, earn_flag = get_earnings_info(t)
            if earn_date:
                expiry_dt = datetime.strptime(target_exp, '%Y-%m-%d').date()
                # SKIP trade if earnings fall between now and expiration
                if today.date() <= earn_date <= expiry_dt:
                    print(f"  [{ticker}] SKIP: Earnings on {earn_date} is before Expiry {target_exp}")
                    continue

            T = target_dte / 365
            chain = t.option_chain(target_exp)
            atm_iv = get_atm_iv(chain.puts, S, T, RISK_FREE_RATE)
            iv_hv_ratio = round(atm_iv / hv, 2) if (atm_iv and hv) else 1.0
            
            # Only selling where IV is at least 20% higher than HV
            if iv_hv_ratio < 1.2: continue

            spread = find_best_spread(S, atm_iv or hv, T, chain.puts)
            if spread:
                results.append({
                    'ticker': ticker, 'price': round(S, 2), 'ivr': ivr,
                    'iv_hv_ratio': iv_hv_ratio, 'dte': target_dte,
                    'expiry': target_exp, 'earnings_date': str(earn_date),
                    'earnings_flag': earn_flag, **spread
                })
                print(f"  [{ticker}] SUCCESS: Spread found at ${spread['short_strike']}")
            
            time.sleep(0.3) # Avoid hitting API limits
        except Exception as e:
            continue

    results.sort(key=lambda x: -x['iv_hv_ratio'])
    return results

# --- EMAIL FORMATTING & DELIVERY ---

def build_email_html(results, date_str):
    th = 'style="padding:8px;font-size:10px;color:#7a7870;text-transform:uppercase;border-bottom:1px solid #1e232b;text-align:right;"'
    th_l = th.replace('right','left')
    
    rows = ""
    for i, r in enumerate(results[:15], 1):
        bg = "#111418" if i % 2 == 1 else "#0f1317"
        td = f'style="padding:10px;font-family:monospace;font-size:12px;color:#e8e6e0;text-align:right;background:{bg};"'
        td_l = td.replace('right','left')
        
        rows += f"""<tr>
            <td {td_l}><b>{r['ticker']}</b></td>
            <td {td}>${r['price']}</td>
            <td {td}>{r['ivr']}%</td>
            <td {td} style="color:#f0a500;">{r['iv_hv_ratio']}x</td>
            <td {td} style="text-align:center;">${r['short_strike']}/${r['long_strike']}</td>
            <td {td} style="color:#3dba6e;font-weight:bold;">${r['credit']}</td>
            <td {td}>{r['credit_ratio']}%</td>
            <td {td}>{r['buffer']}%</td>
            <td {td} style="color:#3dba6e;">${r['max_profit']}</td>
            <td {td} style="color:#e05252;">-${r['max_loss']}</td>
            <td {td} style="text-align:center;">{r['dte']}d</td>
            <td {td} style="font-size:10px;text-align:center;color:#5b9cf6;">{r['earnings_flag'] or '—'}</td>
        </tr>"""

    return f"""
    <html>
    <body style="background:#0a0c0f;color:#e8e6e0;font-family:sans-serif;padding:20px;">
        <h2 style="color:#f0a500;margin-bottom:4px;">BULL PUT SPREAD OPPORTUNITIES</h2>
        <p style="font-size:12px;color:#7a7870;">{date_str} | Haircut: 40% | Earnings Hard Filter: ON</p>
        <table style="width:100%;border-collapse:collapse;margin-top:20px;">
            <thead>
                <tr>
                    <th {th_l}>Ticker</th><th {th}>Price</th><th {th}>IVR</th><th {th}>Edge</th>
                    <th {th} style="text-align:center;">Strikes</th><th {th}>Credit*</th><th {th}>C/W%</th>
                    <th {th}>Buffer</th><th {th}>Profit*</th><th {th}>Risk</th><th {th}>DTE</th><th {th}>Earnings</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        <div style="margin-top:20px;font-size:11px;color:#4a4840;line-height:1.6;">
            * <b>Credits and Profits</b> are calculated using the MID price minus 40% to ensure realistic fills on Robinhood.<br>
            * <b>Earnings Filter</b> is active. Any ticker reporting before the option expiry has been automatically excluded.
        </div>
    </body>
    </html>
    """

def send_email(results):
    if not results:
        print("No opportunities found today. Skipping email.")
        return
    
    date_str = datetime.today().strftime("%A, %B %d, %Y")
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Bull Put Ops: {len(results)} Opportunities ({datetime.today().strftime('%Y-%m-%d')})"
    msg["From"] = GMAIL_USER
    msg["To"] = EMAIL_RECIPIENT
    
    html_content = build_email_html(results, date_str)
    msg.attach(MIMEText(html_content, "html"))
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.send_message(msg)
        print(f"Email sent successfully to {EMAIL_RECIPIENT}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# --- EXECUTION ---

if __name__ == "__main__":
    final_results = run_screen()
    send_email(final_results)
