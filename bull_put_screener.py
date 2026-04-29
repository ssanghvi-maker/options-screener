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
    """Verified 500+ Ticker List with correct syntax."""
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
        'TFX', 'TMUS', 'TPR', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSN', 'TT', 'TTWO', 
        'TXT', 'TYL', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNP', 'URI', 'VFC', 'VLO', 
        'VMC', 'VNO', 'VRSK', 'VRSN', 'VTR
