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

# --- CONFIGURATION (STRICT RULES) ---
HAIRCUT_MULTIPLIER = 0.60  # 40% Haircut
MIN_CW_RATIO       = 0.20  # 20% C/W Floor
IVR_THRESHOLD      = 50    
IV_HV_RATIO        = 1.2   
RISK_FREE_RATE     = 0.05

# Environment Variables
GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_PASS = os.environ.get("GMAIL_APP_PASSWORD")
RECIPIENT  = os.environ.get("EMAIL_RECIPIENT")

def get_manual_500():
    """Manually hardcoded list of 500+ liquid tickers to bypass scraping issues."""
    # This includes the bulk of the S&P 500 + Major ETFs
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
        'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDW', 'CE', 'CEG', 'CF', 'CFG',
        'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMI',
        'CMS', 'CNC', 'CNP', 'COF', 'COO', 'CPB', 'CPRT', 'CPT', 'CSGP', 'CSX', 'CTAS',
        'CTRA', 'CTSH', 'CTVA', 'CVS', 'CVX', 'D', 'DAL', 'DD', 'DE', 'DFS', 'DG',
        'DGX', 'DHI', 'DHR', 'DISH', 'DLTR', 'DLR', 'DOV', 'DOW', 'DRI', 'DTE', 'DUK',
        'DVA', 'DVN', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMN',
        'EMR', 'ENPH', 'EOG', 'EPAM', 'EQR', 'EQT', 'ES', 'ESS', 'ETR', 'EVRG', 'EW',
        'EXC', 'EXPD', 'EXPE', 'EXR', 'FDS', 'FDX', 'FE', 'FFIV', 'FIS', 'FISV', 'FITB',
        'FLT', 'FMC', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GE', 'GEHC', 'GEN',
        'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GPC', 'GPN', 'GRMN', 'GS',
        'GWRE', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HES', 'HIG', 'HII', 'HLT',
        'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUM', 'HWM', 'IBM',
        'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IPG',
        'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JCI', 'JKHY',
        'JNJ', 'JNPR', 'JPM', 'K', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KMB', 'KMI',
        'KMX', 'KO', 'KR', 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT',
        'LNC', 'LNT', 'LOW', 'LRCX', 'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA',
        'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM',
        'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPC',
        'MPWR', 'MRK', 'MRNA', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU',
        'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NLOK', 'NOC', 'NOW',
        'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWL', 'NWS', 'NWSA', 'O',
        'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PARA', 'PAYC',
        'PAYX', 'PCAR', 'PCG', 'PEAK', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH',
        'PHM', 'PKG', 'PKI', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'POOL', 'PPG', 'PPL',
        'PRU', 'PSA', 'PSX', 'PTC', 'PVH', 'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL',
        'RE', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP',
        'ROST', 'RSG', 'RTX', 'RVTY', 'SBAC', 'SBNY', 'SBUX', 'SCHW', 'SEDG', 'SEE',
        'SHW', 'SIVB', 'SJK', 'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE',
        'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG',
        'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TMUS', 'TPR',
        'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL',
        'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VFC',
        'VLO', 'VMC', 'VNO', 'VRSK', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT',
        'WBA', 'WBD', 'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'WRB',
        'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XLY', 'XOM', 'XRAY', 'XYL', 'YUM', 'ZBH',
        'ZBRA', 'ZION', 'ZTS', 'SPY', 'QQQ', 'IWM', 'DIA'
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

# --- SCANNER LOGIC ---
def run_screen():
    tickers = get_manual_500()
    today = datetime.today().date()
    final_picks = []

    print(f"--- STARTING SCAN ON {len(tickers)} TICKERS ---")

    for i, ticker in enumerate(tickers):
        try:
            # Manual Earnings Skip for Today/Tomorrow (April 29-30, 2026)
            if ticker in ['MSFT', 'AMZN', 'GOOGL', 'META', 'AAPL']: continue

            t = yf.Ticker(ticker)
            hist = t.history(period='1y')
            if len(hist) < 252: continue

            # Volatility Logic
            returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
            hv = returns.tail(30).std() * np.sqrt(252)
            rolling_hv = returns.rolling(30).std().dropna() * np.sqrt(252)
            ivr = ((hv - rolling_hv.min()) / (rolling_hv.max() - rolling_hv.min())) * 100
            
            if ivr < IVR_THRESHOLD: continue

            # Options Selection
            exps = t.options
            target_exp = next((e for e in exps if 28 <= (datetime.strptime(e, '%Y-%m-%d').date() - today).days <= 50), None)
            if not target_exp: continue

            chain = t.option_chain(target_exp)
            S = hist['Close'].iloc[-1]
            puts = chain.puts
            
            # ATM IV Check
            atm_put = puts.iloc[(puts['strike'] - S).abs().argsort()[:1]]
            mid_atm = (atm_put['bid'].iloc[0] + atm_put['ask'].iloc[0]) / 2
            iv_est = implied_vol(mid_atm, S, atm_put['strike'].iloc[0], 40/365, RISK_FREE_RATE)
            
            if not iv_est or (iv_est / hv) < IV_HV_RATIO: continue

            # Spread Math
            for _, put in puts.iterrows():
                K_short = put['strike']
                if put['bid'] <= 0 or K_short >= S: continue
                
                delta = bs_put_delta(S, K_short, 40/365, RISK_FREE_RATE, iv_est)
                if not (-0.40 <= delta <= -0.15): continue

                for width in [2, 5, 10]:
                    K_long = K_short - width
                    l_row = puts[puts['strike'] == K_long]
                    if l_row.empty: continue
                    
                    raw_credit = ((put['bid']+put['ask'])/2) - ((l_row.iloc[0]['bid']+l_row.iloc[0]['ask'])/2)
                    final_credit = round(raw_credit * HAIRCUT_MULTIPLIER, 2)
                    
                    if (final_credit / width) >= MIN_CW_RATIO:
                        print(f"  [FOUND] {ticker} | IVR: {round(ivr,1)}")
                        final_picks.append({
                            'Ticker': ticker, 'Price': round(S,2), 'IVR': round(ivr,1),
                            'Short': K_short, 'Long': K_long, 'Credit': final_credit, 
                            'CW_Pct': round((final_credit/width)*100,1), 'Delta': abs(round(delta,2))
                        })
                        break 
        except: continue
        if i % 100 == 0: print(f"Progress: {i}/{len(tickers)}")
        
    return final_picks

def send_email(data):
    if not data or not GMAIL_USER:
