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

# --- CONFIGURATION ---
HAIRCUT_MULTIPLIER = 0.60
MIN_CW_RATIO       = 0.20
IVR_THRESHOLD      = 50    
IV_HV_RATIO        = 1.2   
RISK_FREE_RATE     = 0.05

GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_PASS = os.environ.get("GMAIL_APP_PASSWORD")
RECIPIENT  = os.environ.get("EMAIL_RECIPIENT")

def get_manual_tickers():
    """Audited Ticker List: Verified quotes and brackets."""
    return [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
        'XOM', 'V', 'PG', 'MA', 'AVGO', 'HD', 'CVX', 'ABBV', 'LLY', 'MRK', 'COST', 'PEP',
        'TMO', 'WMT', 'KO', 'DIS', 'CSCO', 'ACN', 'ADBE', 'ORCL', 'AMD', 'NFLX', 'CRM',
        'ABT', 'CMCSA', 'TXN', 'DHR', 'INTC', 'HON', 'QCOM', 'VZ', 'PM', 'NKE', 'LOW',
        'RTX', 'UPS', 'T', 'COP', 'SPGI', 'IBM', 'CAT', 'AXP', 'LMT', 'AMAT', 'GE',
        'BA', 'INTU', 'GS', 'PLD', 'C', 'ELV', 'DE', 'BKNG', 'MDLZ', 'SYK', 'ADI',
        'GILD', 'ISRG', 'TJX', 'REGN', 'VRTX', 'LRCX', 'ZTS', 'MMC', 'SCHW', 'MU',
        'PANW', 'SNPS', 'CDNS', 'ETN', 'SLB', 'CVS', 'CI', 'BSX', 'WM', 'BDX', 'TGT',
        'KLAC', 'PGR', 'MCD', 'EOG', 'MCK', 'EQIX', 'ORLY', 'APH', 'MAR', 'CMG', 'AIG',
        'MO', 'MET', 'DASH', 'CRWD', 'MSTR', 'COIN', 'COF', 'O', 'MS', 'USB', 'PFE',
        'F', 'GM', 'UBER', 'ABNB', 'PYPL', 'ADSK', 'A', 'ADM', 'AES', 'AFL', 'AKAM',
        'ALB', 'ALGN', 'ALLE', 'AMCR', 'AME', 'AMGN', 'AMP', 'AMT', 'ANET', 'ANSS',
        'AON', 'AOS', 'APA', 'APD', 'APTV', 'ARE', 'ATO', 'AVB', 'AVY', 'AWK', 'AXON',
        'AZO', 'BALL', 'BAX', 'BBWI', 'BBY', 'BEN', 'BF-B', 'BIIB', 'BIO', 'BK',
        'BKR', 'BLDR', 'BLK', 'BMY', 'BR', 'BRO', 'BWA', 'BXP', 'CAG', 'CAH', 'CARR',
        'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDW', 'CE', 'CEG', 'CF', 'CFG', 'CHD', 
        'CHRW', 'CHTR', 'CINF', 'CL', 'CLX', 'CMA', 'CME', 'CMI', 'CMS', 'CNC', 'CNP', 
        'COO', 'CPB', 'CPRT', 'CPT', 'CSGP', 'CSX', 'CTAS', 'CTRA', 'CTSH', 'CTVA', 
        'D', 'DAL', 'DD', 'DFS', 'DG', 'DGX', 'DHI', 'DISH', 'DLTR', 'DLR', 'DOV', 
        'DOW', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 
        'EFX', 'EIX', 'EL', 'EMN', 'EMR', 'ENPH', 'EPAM', 'EQR', 'EQT', 'ES', 'ESS', 
        'ETR', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'FDS', 'FDX', 'FE', 'FFIV', 
        'FIS', 'FISV', 'FITB', 'FLT', 'FMC', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 
        'GD', 'GEHC', 'GEN', 'GIS', 'GL', 'GLW', 'GNRC', 'GOOG', 'GPC', 'GPN', 'GRMN', 
        'GWRE', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HES', 'HIG', 'HII', 'HLT', 
        'HOLX', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUM', 'HWM', 'IBM', 
        'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INVH', 'IP', 'IPG', 'IQV', 'IR', 
        'IRM', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JCI', 'JKHY', 'JNPR', 'JPM', 'K', 
        'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KMB', 'KMI', 'KMX', 'KR', 'L', 'LDOS', 
        'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LNC', 'LNT', 'LUV', 'LVS', 'LW', 'LYB', 
        'LYV', 'MAA', 'MAS', 'MCHP', 'MCO', 'MDT', 'MGM', 'MHK', 'MKC', 'MKTX', 
        'MLM', 'MMM', 'MNST', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRNA', 'MSCI', 'MSI', 
        'MTB', 'MTCH', 'MTD', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NI', 'NLOK', 
        'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVR', 'NWL', 'NWS', 
        'NWSA', 'ODFL', 'OKE', 'OMC', 'ON', 'OTIS', 'OXY', 'PARA', 'PAYC', 'PAYX', 
        'PCAR', 'PCG', 'PEAK', 'PEG', 'PFG', 'PH', 'PHM', 'PKG', 'PKI', 'PNC', 'PNR', 
        'PNW', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC', 'PVH', 'PWR', 'PXD', 
        'QRVO', 'RCL', 'RE', 'REG', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 
        'ROP', 'ROST', 'RSG', 'RVTY', 'SBAC', 'SBNY', 'SBUX', 'SCHW', 'SEDG', 'SEE', 
        'SHW', 'SIVB', 'SJK', 'SNA', 'SO', 'SPG', 'SRE', 'STE', 'STT', 'STX', 'STZ', 
        'SWK', 'SWKS', 'SYF', 'SYY', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 
        'TFX', 'TMUS', 'TPR', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 
        'TXT', 'TYL', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNP', 'URI', 'VFC', 'VLO', 
        'VMC', 'VNO', 'VRSK', 'VRSN', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD', 
        'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WMB', 'WRB', 'WST', 'WTW', 'WY', 
        'WYNN', 'XEL', 'XLY', 'XOM', 'XRAY', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZION', 
        'SPY', 'QQQ', 'IWM', 'DIA', 'SMH', 'XLF'
    ]

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

def run_screen():
    tickers = get_manual_tickers()
    today = datetime.today().date()
    final_picks = []

    print(f"--- STARTING SCAN ON {len(tickers)} TICKERS ---")

    for i, ticker in enumerate(tickers):
        try:
            if ticker in ['MSFT', 'AMZN', 'GOOGL', 'META', 'AAPL']: continue

            t = yf.Ticker(ticker)
            hist = t.history(period='1y')
            if len(hist) < 252: continue

            returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
            hv = returns.tail(30).std() * np.sqrt(252)
            rolling_hv = returns.rolling(30).std().dropna() * np.sqrt(252)
            ivr = ((hv - rolling_hv.min()) / (rolling_hv.max() - rolling_hv.min())) * 100
            
            if ivr < IVR_THRESHOLD: continue

            exps = t.options
            target_exp = next((e for e in exps if 28 <= (datetime.strptime(e, '%Y-%m-%d').date() - today).days <= 50), None)
            if not target_exp: continue

            chain = t.option_chain(target_exp)
            S = hist['Close'].iloc[-1]
            puts = chain.puts
            
            atm_put = puts.iloc[(puts['strike'] - S).abs().argsort()[:1]]
            mid_atm = (atm_put['bid'].iloc[0] + atm_put['ask'].iloc[0]) / 2
            iv_est = implied_vol(mid_atm, S, atm_put['strike'].iloc[0], 40/365, RISK_FREE_RATE)
            
            if not iv_est or (iv_est / hv) < IV_HV_RATIO: continue

            for _, put in puts.iterrows():
                K_short = put['strike']
                if put['bid'] <= 0 or K_short >= S: continue
                
                delta = bs_put_delta(S, K_short, 40/365, RISK_FREE_RATE, iv_est)
                if not (-0.40 <= delta <= -0.15): continue

                for width in [2, 5, 10]:
                    K_long = K_short - width
                    l_row = puts[puts['strike'] == K_long]
                    if l_row.empty: continue
                    
                    mid_s = (put['bid'] + put['ask']) / 2
                    mid_l = (l_row.iloc[0]['bid'] + l_row.iloc[0]['ask']) / 2
                    raw_credit = mid_s - mid_l
                    final_credit = round(raw_credit * HAIRCUT_MULTIPLIER, 2)
                    
                    if (final_credit / width) >= MIN_CW_RATIO:
                        print(f"  [FOUND] {ticker} - Credit: {final_credit}")
                        final_picks.append({
                            'Ticker': ticker, 'Price': round(S, 2), 'IVR': round(ivr, 1),
                            'Short': K_short, 'Long': K_long, 'Width': width,
                            'Credit': final_credit, 'CW_Pct': round((final_credit/width)*100, 1), 
                            'Delta': abs(round(delta, 2))
                        })
                        break 
        except: continue
        if (i + 1) % 100 == 0: print(f"Progress: {i + 1}/{len(tickers)} scanned...")
        
    return final_picks

def send_email(data):
    if not data or not GMAIL_USER:
        print("Scan finished. No candidates found or email not configured.")
        return
    
    df = pd.DataFrame(data)
    html_content = f"<html><body><h2>Option Scan Results</h2>{df.to_html(index=False)}</body></html>"
    msg = MIMEMultipart()
    msg['Subject'] = f"Option Report: {len(data)} Candidates"
    msg.attach(MIMEText(html_content, 'html'))
    
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(GMAIL_USER, GMAIL_PASS)
            
            # send_message returns a dict of failures. 
            # If it's empty ({}), the email was 100% successful.
            errors = server.send_message(msg)
            
            if not errors:
                print("✅ Email sent successfully!")
            else:
                print(f"⚠️ Email sent, but these addresses failed: {errors}")
                
    except Exception as e: 
        # This will now only catch REAL errors (like wrong passwords)
        print(f"❌ SMTP Connection Error: {e}")

if __name__ == "__main__":
    results = run_screen()
    send_email(results)
