import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime
import smtplib
import os
import warnings
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Suppress technical noise
warnings.filterwarnings('ignore')

# --- CONFIGURATION (YOUR RULES) ---
HAIRCUT_MULTIPLIER = 0.60  # 40% Slashing
MIN_CW_RATIO       = 0.20  # 20% C/W floor AFTER haircut
IVR_THRESHOLD      = 50    
IV_HV_RATIO        = 1.2   
RISK_FREE_RATE     = 0.05

# GitHub Secrets / Env Vars
GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_PASS = os.environ.get("GMAIL_APP_PASSWORD")
RECIPIENT  = os.environ.get("EMAIL_RECIPIENT")

# --- THE MANUALLY ADDED LIST (Expanded Fallback) ---
def get_hardcoded_tickers():
    return [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
        'XOM', 'V', 'PG', 'MA', 'AVGO', 'HD', 'CVX', 'ABBV', 'LLY', 'MRK',
        'COST', 'PEP', 'TMO', 'WMT', 'KO', 'DIS', 'CSCO', 'ACN', 'ADBE', 'ORCL',
        'AMD', 'NFLX', 'CRM', 'ABT', 'CMCSA', 'TXN', 'DHR', 'INTC', 'HON', 'QCOM',
        'VZ', 'PM', 'NKE', 'LOW', 'RTX', 'UPS', 'T', 'COP', 'SPGI', 'IBM',
        'CAT', 'AXP', 'LMT', 'AMAT', 'GE', 'BA', 'INTU', 'GS', 'PLD', 'C',
        'ELV', 'DE', 'BKNG', 'MDLZ', 'SYK', 'ADI', 'GILD', 'ISRG', 'TJX', 'REGN',
        'VRTX', 'LRCX', 'ZTS', 'MMC', 'SCHW', 'MU', 'PANW', 'SNPS', 'CDNS', 'ETN',
        'SLB', 'CVS', 'CI', 'BSX', 'WM', 'BDX', 'TGT', 'KLAC', 'PGR', 'MCD',
        'SPY', 'QQQ', 'IWM', 'SMH', 'XLF', 'XLK', 'XLE', 'XLV', 'XLY', 'XLI'
    ]

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

# --- SCANNING ENGINE ---
def run_screen():
    tickers = get_hardcoded_tickers()
    today = datetime.today().date()
    final_results = []

    print(f"--- STARTING HARDCODED SCAN: {len(tickers)} TICKERS ---")

    for ticker in tickers:
        try:
            # Shield: MSFT/AMZN/GOOGL have earnings today (April 29, 2026)
            if ticker in ['MSFT', 'AMZN', 'GOOGL', 'META']: continue

            t = yf.Ticker(ticker)
            
            # Avoid the 404 by skipping t.calendar and t.info. 
            # We calculate HV from historical prices only.
            hist = t.history(period='1y')
            if len(hist) < 252: continue
            
            returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
            hv = returns.tail(30).std() * np.sqrt(252)
            rolling_hv = returns.rolling(30).std().dropna() * np.sqrt(252)
            ivr = ((hv - rolling_hv.min()) / (rolling_hv.max() - rolling_hv.min())) * 100
            
            if ivr < IVR_THRESHOLD: continue

            # Get Options Expiry
            exps = t.options
            target_exp = next((e for e in exps if 28 <= (datetime.strptime(e, '%Y-%m-%d').date() - today).days <= 50), None)
            if not target_exp: continue

            chain = t.option_chain(target_exp)
            S = hist['Close'].iloc[-1]
            puts = chain.puts
            
            # Estimate IV from ATM Put
            atm_put = puts.iloc[(puts['strike'] - S).abs().argsort()[:1]]
            mid_atm = (atm_put['bid'].iloc[0] + atm_put['ask'].iloc[0]) / 2
            iv_est = implied_vol(mid_atm, S, atm_put['strike'].iloc[0], 40/365, RISK_FREE_RATE)
            
            if not iv_est or (iv_est / hv) < IV_HV_RATIO: continue

            # Search Spreads
            for _, put in puts.iterrows():
                K_short = put['strike']
                if put['bid'] <= 0 or K_short > S: continue
                
                delta = bs_put_delta(S, K_short, 40/365, RISK_FREE_RATE, iv_est)
                if not (-0.40 <= delta <= -0.15): continue

                for width in [2, 5, 10]:
                    K_long = K_short - width
                    l_row = puts[puts['strike'] == K_long]
                    if l_row.empty: continue
                    
                    raw_credit = ((put['bid']+put['ask'])/2) - ((l_row.iloc[0]['bid']+l_row.iloc[0]['ask'])/2)
                    final_credit = round(raw_credit * HAIRCUT_MULTIPLIER, 2)
                    
                    if (final_credit / width) >= MIN_CW_RATIO:
                        print(f"  [FOUND] {ticker}")
                        final_results.append({
                            'Ticker': ticker, 'Price': round(S,2), 'IVR': round(ivr,1),
                            'Short': K_short, 'Long': K_long, 'Credit': final_credit, 
                            'CW_Pct': round((final_credit/width)*100,1), 'Delta': abs(round(delta,2))
                        })
                        break 
        except: continue
    return final_results

def send_email(data):
    if not data or not GMAIL_USER: return
    df = pd.DataFrame(data)
    html = f"<html><body>{df.to_html(index=False)}</body></html>"
    msg = MIMEMultipart(); msg['Subject'] = "Manual Trade Report"
    msg.attach(MIMEText(html, 'html'))
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls(); s.login(GMAIL_USER, GMAIL_PASS); s.send_message(msg)
            print("Email Sent.")
    except Exception as e: print(f"Email Failed: {e}")

if __name__ == "__main__":
    found = run_screen()
    send_email(found
