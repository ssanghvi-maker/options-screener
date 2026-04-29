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
GMAIL_USER         = os.environ.get("GMAIL_USER", "your_email@gmail.com")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "your_app_password")
EMAIL_RECIPIENT    = os.environ.get("EMAIL_RECIPIENT", "your_email@gmail.com")

HAIRCUT_MULTIPLIER = 0.60  
MIN_CW_RATIO       = 0.20  
RISK_FREE_RATE     = 0.05

def get_sp500_tickers():
    """Manual S&P 500 Ticker List to bypass LXML/Wikipedia errors."""
    # This is a robust subset of the most liquid S&P 500 names. 
    # You can add all 500 here, but the top 100-200 provide the best option liquidity.
    return [
        'AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AVGO','AMD','NFLX',
        'ADBE','CRM','ORCL','INTU','QCOM','TXN','AMAT','LRCX','KLAC','ADI',
        'JPM','GS','MS','BAC','C','WFC','SCHW','BLK','AXP','SPGI',
        'ICE','CME','COF','USB','PNC','XOM','CVX','OXY','SLB','EOG','COP','PSX','VLO','MPC','HAL',
        'LLY','UNH','JNJ','MRK','ABBV','PFE','TMO','DHR','BMY','AMGN',
        'ISRG','SYK','MDT','ZTS','VRTX',

        'CAT','DE','GE','HON','ETN','PH','ITW','MMM','GD','LMT',
        'BA','NOC','EMR','ROK','OTIS',

        'COST','WMT','HD','LOW','MCD','SBUX','NKE','TJX','TGT',
        'BKNG','ABNB','RCL','MAR','HLT',

        'PLTR','SNOW','CRWD','PANW','DDOG','NET','ZS','MDB','OKTA','TEAM',
        'SHOP','UBER','LYFT','DASH','RIVN','LCID',

        'MRVL','NXPI','ON','INTC','MPWR',

        'PLD','AMT','CCI','EQIX','O','PSA','SPG','DLR',

        'SPY','QQQ','IWM','XLF','XLE','XLK','XLV','SMH','ARKK','DIA'
        # Add more tickers here as needed
    ]

# --- OPTION MATH ---
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
    try:
        cal = ticker_obj.calendar
        if cal is None or (isinstance(cal, pd.DataFrame) and cal.empty): return None, None
        if isinstance(cal, dict): earn_date = cal.get('Earnings Date', [None])[0]
        else: earn_date = cal.columns[0] if hasattr(cal, 'columns') else cal.iloc[0,0]
        if hasattr(earn_date, 'date'): earn_date = earn_date.date()
        today = datetime.today().date()
        days_to = (earn_date - today).days
        if days_to < 0: return earn_date, f"JUST REPORTED ({abs(days_to)}d ago)"
        return earn_date, f"EARNINGS IN {days_to}d"
    except: return None, None

def find_best_spread(S, iv_for_delta, T, puts):
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
            credit = round((short_mid - long_mid) * HAIRCUT_MULTIPLIER, 2)
            if (credit / width) >= MIN_CW_RATIO and (S - (K_short - credit)) / S * 100 >= 5:
                return {
                    'short_strike': K_short, 'long_strike': K_long, 'width': width,
                    'delta': round(abs(delta), 2), 'credit': credit, 'credit_ratio': round((credit/width)*100, 1),
                    'buffer': round((S - (K_short - credit)) / S * 100, 1), 'max_profit': int(credit * 100),
                    'max_loss': int((width - credit) * 100), 'breakeven': round(K_short - credit, 2)
                }
    return None

# --- RUN SCREENER ---
def run_screen():
    today = datetime.today()
    results = []
    tickers = get_sp500_tickers()
    print(f"Scanning {len(tickers)} tickers for Bull Put Spreads...")

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

            # EARNINGS HARD FILTER
            earn_date, earn_flag = get_earnings_info(t)
            if earn_date:
                expiry_dt = datetime.strptime(target_exp, '%Y-%m-%d').date()
                if today.date() <= earn_date <= expiry_dt:
                    print(f"  [{ticker}] SKIP: Earnings inside window ({earn_date})")
                    continue

            chain = t.option_chain(target_exp)
            p = chain.puts.copy()
            p['dist'] = abs(p['strike'] - S)
            atm_put = p.nsmallest(1, 'dist').iloc[0]
            atm_iv = implied_vol((atm_put['bid']+atm_put['ask'])/2, S, atm_put['strike'], target_dte/365, RISK_FREE_RATE)
            
            if not atm_iv or (atm_iv / hv) < 1.2: continue

            spread = find_best_spread(S, atm_iv, target_dte/365, chain.puts)
            if spread:
                results.append({'ticker': ticker, 'price': round(S, 2), 'ivr': ivr, 'iv_hv': round(atm_iv/hv, 2), 
                                'dte': target_dte, 'expiry': target_exp, 'earnings': earn_flag, **spread})
                print(f"  [{ticker}] SUCCESS found.")
            time.sleep(0.1)
        except: continue
    
    results.sort(key=lambda x: -x['iv_hv'])
    return results

# --- EMAIL ---
def send_email(results):
    if not results: return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Bull Put Ops - {len(results)} Found - {datetime.today().strftime('%Y-%m-%d')}"
    msg["From"], msg["To"] = GMAIL_USER, EMAIL_RECIPIENT
    
    html = f"<html><body style='background:#0a0c0f;color:#eee;padding:20px;font-family:sans-serif;'><h2>Bull Put Screener</h2>"
    html += "<table border='1' style='border-collapse:collapse;width:100%;'><tr><th>Ticker</th><th>Price</th><th>IVR</th><th>Strikes</th><th>Credit*</th><th>C/W%</th><th>Buffer</th><th>Earnings</th></tr>"
    for r in results:
        html += f"<tr><td>{r['ticker']}</td><td>{r['price']}</td><td>{r['ivr']}%</td><td>{r['short_strike']}/{r['long_strike']}</td><td>${r['credit']}</td><td>{r['credit_ratio']}%</td><td>{r['buffer']}%</td><td>{r['earnings']}</td></tr>"
    html += "</table><p style='font-size:10px;'>*40% Haircut applied. Earnings excluded.</p></body></html>"
    
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.send_message(msg)

if __name__ == "__main__":
    data = run_screen()
    send_email(data)
