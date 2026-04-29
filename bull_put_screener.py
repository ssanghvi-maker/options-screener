import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime
import requests
import smtplib
import os
import warnings
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Suppress noise
warnings.filterwarnings('ignore')

# --- CONFIGURATION (STRICT RULES) ---
HAIRCUT_MULTIPLIER = 0.60  # 40% Slashing
MIN_CW_RATIO       = 0.20  # 20% C/W Floor AFTER haircut
IVR_THRESHOLD      = 50    
IV_HV_RATIO        = 1.2   
RISK_FREE_RATE     = 0.05

# EMAIL CONFIG
GMAIL_USER = "your_email@gmail.com"
GMAIL_APP_PASS = "your_app_password" 
RECIPIENT = "your_email@gmail.com"

# --- MATH ENGINE ---
def bs_put_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_put_delta(S, K, T, r, sigma):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1) - 1
    except: return -0.5

def implied_vol(market_price, S, K, T, r):
    try:
        def f(sigma): return bs_put_price(S, K, T, r, sigma) - market_price
        return brentq(f, 0.001, 5.0, xtol=1e-4)
    except: return None

# --- DATA FETCHING ---
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        df = pd.read_html(response.text)[0]
        tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
        etfs = ['SPY', 'QQQ', 'IWM', 'DIA', 'SMH', 'XLF', 'XLE', 'XLK']
        return sorted(list(set(tickers + etfs)))
    except: return ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'NVDA']

def get_vol_data(ticker_obj):
    try:
        hist = ticker_obj.history(period='1y')
        if len(hist) < 252: return None, None
        returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        hv = returns.tail(30).std() * np.sqrt(252)
        rolling_hv = returns.rolling(30).std().dropna() * np.sqrt(252)
        ivr = ((hv - rolling_hv.min()) / (rolling_hv.max() - rolling_hv.min())) * 100
        return hv, round(ivr, 1)
    except: return None, None

# --- SPREAD FINDER ---
def find_spread(S, iv_est, T, puts):
    for _, put in puts.iterrows():
        K_short = put['strike']
        if put['bid'] <= 0 or put['ask'] <= 0: continue
        delta = bs_put_delta(S, K_short, T, RISK_FREE_RATE, iv_est)
        if not (-0.40 <= delta <= -0.15): continue

        for width in [2, 3, 5, 10]:
            K_long = K_short - width
            long_row = puts[puts['strike'] == K_long]
            if long_row.empty or long_row.iloc[0]['bid'] <= 0: continue
            
            raw_credit = ((put['bid'] + put['ask'])/2) - ((long_row.iloc[0]['bid'] + long_row.iloc[0]['ask'])/2)
            final_credit = round(raw_credit * HAIRCUT_MULTIPLIER, 2)
            
            if (final_credit / width) >= MIN_CW_RATIO:
                return {'Short': K_short, 'Long': K_long, 'Width': width, 'Credit': final_credit, 
                        'CW_Pct': round((final_credit/width)*100,1), 'Delta': round(abs(delta),2)}
    return None

# --- MAIN SCANNER ---
def run_screen():
    tickers = get_sp500_tickers()
    today = datetime.today().date()
    final_picks = []
    print(f"--- SCANNING {len(tickers)} TICKERS ---")

    for i, ticker in enumerate(tickers, 1):
        try:
            # Triple-Lock Earnings Shield
            if ticker in ['MSFT', 'AMZN', 'GOOGL', 'META', 'AAPL']: continue
            t = yf.Ticker(ticker)
            cal = t.calendar
            if cal is not None and not cal.empty:
                e_date = cal.iloc[0,0].date()
                if -2 <= (e_date - today).days <= 35: continue

            hv, ivr = get_vol_data(t)
            if ivr is None or ivr < IVR_THRESHOLD: continue

            exps = t.options
            target_exp = next((e for e in exps if 28 <= (datetime.strptime(e, '%Y-%m-%d').date() - today).days <= 50), None)
            if not target_exp: continue

            chain = t.option_chain(target_exp)
            S = t.history(period='1d')['Close'].iloc[-1]
            atm_put = chain.puts.iloc[(chain.puts['strike'] - S).abs().argsort()[:1]]
            iv_est = implied_vol((atm_put['bid'].iloc[0]+atm_put['ask'].iloc[0])/2, S, atm_put['strike'].iloc[0], 40/365, RISK_FREE_RATE)
            
            if not iv_est or (iv_est / hv) < IV_HV_RATIO: continue

            pick = find_spread(S, iv_est, 40/365, chain.puts)
            if pick:
                print(f"  [FOUND] {ticker}")
                final_picks.append({'Ticker': ticker, 'Price': round(S,2), 'IVR': ivr, 'Expiry': target_exp, **pick})
        except: continue
        if i % 50 == 0: print(f"Progress: {i}/{len(tickers)}")
    return final_picks

# --- FORMATTED EMAIL ---
def send_html_email(data):
    if not data: return
    df = pd.DataFrame(data)
    
    html_table = df.to_html(index=False, classes='mystyle')
    html_body = f"""
    <html>
    <head>
    <style>
        .mystyle {{font-family: Arial; border-collapse: collapse; width: 100%;}}
        .mystyle td, .mystyle th {{border: 1px solid #ddd; padding: 8px; text-align: center;}}
        .mystyle tr:nth-child(even){{background-color: #f2f2f2;}}
        .mystyle th {{padding-top: 12px; background-color: #004d99; color: white;}}
    </style>
    </head>
    <body>
    <h2>High Conviction Trade Report</h2>
    <p>Criteria: 40% Haircut applied, Min 20% C/W Ratio, IVR > 50.</p>
    {html_table}
    </body>
    </html>
    """
    
    msg = MIMEMultipart()
    msg['Subject'] = f"Trade Scan: {len(data)} Candidates Found"
    msg.attach(MIMEText(html_body, 'html'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_APP_PASS)
        server.sendmail(GMAIL_USER, RECIPIENT, msg.as_string())
        server.quit()
        print("Email sent successfully.")
    except Exception as e: print(f"Email failed: {e}")

if __name__ == "__main__":
    trades = run_screen()
    send_html_email(trades)
