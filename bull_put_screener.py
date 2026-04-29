import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, date
import requests
import smtplib
import warnings
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Suppress technical noise and 404 logs
warnings.filterwarnings('ignore')

# --- CONFIGURATION (YOUR STRICT RULES) ---
HAIRCUT_MULTIPLIER = 0.60  # 40% Slashing
MIN_CW_RATIO       = 0.20  # 20% C/W floor AFTER haircut
IVR_THRESHOLD      = 50    
IV_HV_RATIO        = 1.2   
RISK_FREE_RATE     = 0.05

# Email Settings (Update these!)
GMAIL_USER = os.environ.get("GMAIL_USER", "your_email@gmail.com")
GMAIL_PASS = os.environ.get("GMAIL_APP_PASSWORD", "your_app_password")
RECIPIENT  = os.environ.get("EMAIL_RECIPIENT", "your_email@gmail.com")

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

# --- RESILIENT DATA FETCHING ---
def get_sp500_tickers():
    """Fetches full S&P 500 list with headers to bypass blocks."""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        # We specify 'lxml' to ensure it parses the full table
        df = pd.read_html(response.text, flavor='lxml')[0]
        tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
        etfs = ['SPY', 'QQQ', 'IWM', 'DIA', 'SMH', 'XLF', 'XLE', 'XLK']
        full_list = sorted(list(set(tickers + etfs)))
        print(f"--- SUCCESS: Scanning {len(full_list)} Tickers ---")
        return full_list
    except Exception as e:
        print(f"Wikipedia Fetch Failed: {e}. Using fallback.")
        return ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META']

def get_earnings_safe(ticker_obj):
    """Fails open on 404s to prevent script crashes."""
    try:
        cal = ticker_obj.calendar
        if cal is not None and not cal.empty:
            e_date = cal.iloc[0, 0] if isinstance(cal, pd.DataFrame) else cal.get('Earnings Date', [None])[0]
            if hasattr(e_date, 'date'): return e_date.date()
    except: pass
    return None

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

# --- CORE LOGIC ---
def run_screen():
    tickers = get_sp500_tickers()
    today = datetime.today().date()
    final_results = []

    for i, ticker in enumerate(tickers, 1):
        try:
            # 1. TRIPLE-LOCK SHIELD
            if ticker in ['MSFT', 'AMZN', 'AAPL', 'GOOGL', 'META']: continue # Manual Blacklist
            
            t = yf.Ticker(ticker)
            earn_date = get_earnings_safe(t)
            if earn_date and -1 <= (earn_date - today).days <= 35: continue

            # 2. VOLATILITY GATES
            hv, ivr = get_vol_data(t)
            if ivr is None or ivr < IVR_THRESHOLD: continue

            # 3. OPTIONS ANALYSIS (Targeting 30-50 DTE)
            exps = t.options
            target_exp = next((e for e in exps if 28 <= (datetime.strptime(e, '%Y-%m-%d').date() - today).days <= 50), None)
            if not target_exp: continue
            
            chain = t.option_chain(target_exp)
            S = t.history(period='1d')['Close'].iloc[-1]
            
            # IV Sniffer (Checks if IV is bloated)
            puts = chain.puts
            atm_put = puts.iloc[(puts['strike'] - S).abs().argsort()[:1]]
            mid_atm = (atm_put['bid'].iloc[0] + atm_put['ask'].iloc[0]) / 2
            iv_est = implied_vol(mid_atm, S, atm_put['strike'].iloc[0], 40/365, RISK_FREE_RATE)
            
            if not iv_est or (iv_est / hv) < IV_HV_RATIO: continue

            # 4. SPREAD MATH
            for _, put in puts.iterrows():
                K_short = put['strike']
                if put['bid'] <= 0: continue
                delta = bs_put_delta(S, K_short, 40/365, RISK_FREE_RATE, iv_est)
                if not (-0.40 <= delta <= -0.15): continue

                for width in [2, 5, 10]:
                    K_long = K_short - width
                    l_row = puts[puts['strike'] == K_long]
                    if l_row.empty: continue
                    
                    raw_credit = ((put['bid']+put['ask'])/2) - ((l_row.iloc[0]['bid']+l_row.iloc[0]['ask'])/2)
                    final_credit = round(raw_credit * HAIRCUT_MULTIPLIER, 2)
                    
                    if (final_credit / width) >= MIN_CW_RATIO:
                        print(f"  [SUCCESS] {ticker} Found")
                        final_results.append({
                            'Ticker': ticker, 'Price': round(S,2), 'IVR': ivr, 'Expiry': target_exp,
                            'Short': K_short, 'Long': K_long, 'Width': width, 
                            'Credit': final_credit, 'CW_Pct': round((final_credit/width)*100,1), 'Delta': round(abs(delta),2)
                        })
                        break # Found one for this ticker, move on
        except: continue
        if i % 50 == 0: print(f"Progress: {i}/{len(tickers)} scanned...")

    return final_results

def send_html_report(results):
    if not results: return
    df = pd.DataFrame(results)
    html_table = df.to_html(index=False, classes='mystyle')
    
    html_body = f"""
    <html><head><style>
        .mystyle {{font-family: Arial; border-collapse: collapse; width: 100%; font-size: 12px;}}
        .mystyle td, .mystyle th {{border: 1px solid #ddd; padding: 8px; text-align: center;}}
        .mystyle th {{background-color: #004d99; color: white;}}
        .mystyle tr:nth-child(even){{background-color: #f2f2f2;}}
    </style></head>
    <body><h2>Trade Scan Results</h2>{html_table}</body></html>
    """
    msg = MIMEMultipart(); msg['Subject'] = f"S&P 500 Scan: {len(results)} Found"; msg.attach(MIMEText(html_body, 'html'))
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls(); server.login(GMAIL_USER, GMAIL_PASS); server.send_message(msg)
            print("Report Sent.")
    except Exception as e: print(f"Email Failed: {e}")

if __name__ == "__main__":
    results = run_screen()
    send_html_report(results)
