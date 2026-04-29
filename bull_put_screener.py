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

# Suppress pandas/yfinance warnings for cleaner logs
warnings.filterwarnings('ignore')

# --- CONFIGURATION (YOUR STRICT DISCIPLINE) ---
HAIRCUT_MULTIPLIER = 0.60  # Your 40% reduction from Mid-price
MIN_CW_RATIO       = 0.20  # Your 20% Credit/Width Floor after haircut
IVR_THRESHOLD      = 50    # High Volatility Requirement
IV_HV_RATIO        = 1.2   # Edge Requirement
RISK_FREE_RATE     = 0.05

# Email Settings (Optional)
GMAIL_USER         = os.environ.get("GMAIL_USER", "your_email@gmail.com")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "your_app_password")
EMAIL_RECIPIENT    = os.environ.get("EMAIL_RECIPIENT", "your_email@gmail.com")

# --- MATH ENGINE ---
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
        return brentq(f, 0.001, 5.0, xtol=1e-4)
    except: return None

# --- DATA FETCHING ---
def get_sp500_tickers():
    """Fetches full S&P 500 from Wikipedia."""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        df = pd.read_html(url)[0]
        tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
        etfs = ['SPY', 'QQQ', 'IWM', 'DIA', 'SMH', 'XLF', 'XLE', 'XLK']
        return sorted(list(set(tickers + etfs)))
    except:
        return ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'NVDA']

def get_vol_data(ticker_obj):
    """Calculates Current HV and IV Rank."""
    try:
        hist = ticker_obj.history(period='1y')
        if len(hist) < 252: return None, None
        returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        hv_current = returns.tail(30).std() * np.sqrt(252)
        rolling_hv = returns.rolling(30).std().dropna() * np.sqrt(252)
        min_v, max_v = rolling_hv.min(), rolling_hv.max()
        ivr = ((hv_current - min_v) / (max_v - min_v)) * 100
        return hv_current, round(ivr, 1)
    except: return None, None

def get_earnings_date(ticker_obj):
    """Avoids stocks with earnings in the next 35 days."""
    try:
        if ticker_obj.info.get('quoteType') == 'ETF': return None
        cal = ticker_obj.calendar
        if cal is None: return None
        # Handle different calendar formats in yfinance
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            e_date = cal.iloc[0, 0]
        elif isinstance(cal, dict):
            e_date = cal.get('Earnings Date', [None])[0]
        else: return None
        
        if hasattr(e_date, 'date'): return e_date.date()
        return None
    except: return None

# --- SCANNING LOGIC ---
def find_best_spread(S, iv_for_delta, T, puts):
    """Applies your 40% haircut and 20% C/W floor."""
    for _, put in puts.iterrows():
        K_short = put['strike']
        if put['bid'] <= 0 or put['ask'] <= 0: continue
        
        # Check Delta (targeting 25-35 range)
        delta = bs_put_delta(S, K_short, T, RISK_FREE_RATE, iv_for_delta)
        if delta is None or not (-0.40 <= delta <= -0.20): continue

        for width in [2, 3, 5, 10]:
            K_long = K_short - width
            long_row = puts[puts['strike'] == K_long]
            if long_row.empty or long_row.iloc[0]['bid'] <= 0: continue
            
            mid_short = (put['bid'] + put['ask']) / 2
            mid_long = (long_row.iloc[0]['bid'] + long_row.iloc[0]['ask']) / 2
            raw_credit = mid_short - mid_long
            
            # --- THE 40% HAIRCUT ---
            final_credit = round(raw_credit * HAIRCUT_MULTIPLIER, 2)
            cw_ratio = final_credit / width
            
            if cw_ratio >= MIN_CW_RATIO:
                return {
                    'short_strike': K_short, 'long_strike': K_long, 'width': width,
                    'credit': final_credit, 'cw_pct': round(cw_ratio*100, 1),
                    'delta': round(abs(delta), 2), 'be': round(K_short - final_credit, 2)
                }
    return None

def run_screen():
    tickers = get_sp500_tickers()
    results = []
    today = datetime.today().date()
    print(f"--- STARTING SCAN: {len(tickers)} TICKERS ---")

    for i, ticker in enumerate(tickers, 1):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period='5d')
            if hist.empty: continue
            S = hist['Close'].iloc[-1]

            # 1. EARNINGS FILTER
            earn_date = get_earnings_date(t)
            if earn_date:
                days_to = (earn_date - today).days
                if 0 <= days_to <= 35:
                    print(f"  [{ticker}] SKIP: Earnings in {days_to} days.")
                    continue

            # 2. VOLATILITY GATES
            hv, ivr = get_vol_data(t)
            if ivr is None or ivr < IVR_THRESHOLD:
                if i % 10 == 0: print(f"  [{ticker}] REJECT: IV Rank ({ivr}%) below {IVR_THRESHOLD}%.")
                continue
            
            print(f"  [{ticker}] PASSED Volatility (IVR: {ivr}%). Searching for spreads...")

            # 3. EXPIRATION SELECTION (28-50 Days)
            exps = t.options
            target_exp = None
            for exp in exps:
                dte = (datetime.strptime(exp, '%Y-%m-%d').date() - today).days
                if 28 <= dte <= 50:
                    target_exp, target_dte = exp, dte
                    break
            
            if not target_exp: continue

            # 4. SPREAD CALCULATION
            chain = t.option_chain(target_exp)
            puts = chain.puts
            # Estimate ATM IV for Delta calculation
            atm_put = puts.iloc[(puts['strike'] - S).abs().argsort()[:1]]
            mid_atm = (atm_put['bid'].iloc[0] + atm_put['ask'].iloc[0]) / 2
            iv_est = implied_vol(mid_atm, S, atm_put['strike'].iloc[0], target_dte/365, RISK_FREE_RATE)
            
            if not iv_est or (iv_est / hv) < IV_HV_RATIO:
                print(f"  [{ticker}] REJECT: IV/HV ratio ({round(iv_est/hv,2) if iv_est else 0}) below {IV_HV_RATIO}.")
                continue

            spread = find_best_spread(S, iv_est, target_dte/365, puts)
            if spread:
                print(f"  [{ticker}] *** SUCCESS! Spread Found ***")
                results.append({'ticker': ticker, 'price': round(S,2), 'ivr': ivr, 'expiry': target_exp, **spread})
            else:
                print(f"  [{ticker}] REJECT: No spread met {int(MIN_CW_RATIO*100)}% C/W after 40% haircut.")

        except Exception as e:
            continue
        
        if i % 50 == 0: print(f"Progress: {i}/{len(tickers)} tickers complete...")

    return results

# --- EMAIL FUNCTION ---
def send_report(results):
    if not results:
        print("\nNo high-conviction trades found today.")
        return

    msg = MIMEMultipart(); msg["Subject"] = f"Trade Alerts: {len(results)} Found"; msg["From"] = GMAIL_USER; msg["To"] = EMAIL_RECIPIENT
    body = f"High Conviction Bull Put Spreads (Rules: 40% Haircut, 20% C/W floor, IVR > 50)\n\n"
    df_res = pd.DataFrame(results)
    body += df_res.to_string(index=False)
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587); server.starttls(); server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.send_message(msg); server.quit()
        print("\nReport Emailed Successfully.")
    except:
        print("\nEmail failed. Check your GMAIL_APP_PASSWORD.")

if __name__ == "__main__":
    found_trades = run_screen()
    send_report(found_trades)
