"""
bull_put_screener.py

Approach:
- Uses MID price for credits (bid+ask)/2 — what brokers display
- Adds IV/HV ratio check — only trades where IV > 1.2x realized vol (real edge)
- Note in email: actual fills ~10-15% below mid on liquid names
- Table format email
- Sorted by IV/HV ratio descending
"""

import yfinance as yf
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

GMAIL_USER         = os.environ.get("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
EMAIL_RECIPIENT    = os.environ.get("EMAIL_RECIPIENT", "")
RISK_FREE_RATE     = 0.05

TICKERS = [
   'A','AAL','AAP','AAPL','ABBV','ABC','ABMD','ABT','ACGL','ACN',
'ADBE','ADI','ADM','ADP','ADSK','AEE','AEP','AES','AFL','AIG',
'AIZ','AJG','AKAM','ALB','ALGN','ALK','ALL','ALLE','AMAT','AMCR',
'AMD','AME','AMGN','AMP','AMT','AMZN','ANET','ANSS','AON','AOS',
'APA','APD','APH','APTV','ARE','ATO','ATVI','AVB','AVGO','AVY',
'AWK','AXP','AZO','BA','BAC','BALL','BAX','BBWI','BBY','BDX',
'BEN','BF.B','BIIB','BIO','BK','BKNG','BKR','BLK','BLL','BMY',
'BR','BRK.B','BRO','BSX','BWA','BXP','C','CAG','CAH','CARR',
'CAT','CB','CBOE','CBRE','CCI','CCL','CDNS','CDW','CE','CEG',
'CF','CFG','CHD','CHRW','CHTR','CI','CINF','CL','CLX','CMA',
'CMCSA','CME','CMG','CMI','CMS','CNC','CNP','COF','COO','COP',
'COST','CPB','CPRT','CPT','CRL','CRM','CSCO','CSGP','CSX','CTAS',
'CTLT','CTRA','CTSH','CTVA','CVS','CVX','CZR','D','DAL','DD',
'DE','DFS','DG','DGX','DHI','DHR','DIS','DLR','DLTR','DOV',
'DOW','DPZ','DRI','DTE','DUK','DVA','DVN','DXC','EA','EBAY',
'ECL','ED','EFX','EIX','EL','EMN','EMR','ENPH','EOG','EPAM',
'EQIX','EQR','EQT','ES','ESS','ETN','ETR','ETSY','EVRG','EW',
'EXC','EXPD','EXPE','EXR','F','FANG','FAST','FCX','FDS','FDX',
'FE','FFIV','FIS','FISV','FITB','FLT','FMC','FOX','FOXA','FRT',
'FTNT','FTV','GD','GE','GEN','GILD','GIS','GL','GLW','GM',
'GNRC','GOOG','GOOGL','GPC','GPN','GRMN','GS','GWW','HAL','HAS',
'HBAN','HCA','HD','HES','HIG','HII','HLT','HOLX','HON','HPE',
'HPQ','HRL','HSIC','HST','HSY','HUM','HWM','IBM','ICE','IDXX',
'IEX','IFF','ILMN','INCY','INTC','INTU','INVH','IP','IPG','IQV',
'IR','IRM','ISRG','IT','ITW','IVZ','J','JBHT','JCI','JKHY',
'JNJ','JNPR','JPM','K','KDP','KEY','KEYS','KHC','KIM','KLAC',
'KMB','KMI','KMX','KO','KR','L','LDOS','LEN','LH','LHX',
'LIN','LKQ','LLY','LMT','LNC','LNT','LOW','LRCX','LUV','LVS',
'LW','LYB','LYV','MA','MAA','MAR','MAS','MCD','MCHP','MCK',
'MCO','MDLZ','MDT','MET','META','MGM','MHK','MKC','MKTX','MLM',
'MMC','MMM','MNST','MO','MOS','MPC','MPWR','MRK','MRNA','MRO',
'MS','MSCI','MSFT','MSI','MTB','MTCH','MTD','MU','NCLH','NDAQ',
'NDSN','NEE','NEM','NFLX','NI','NKE','NOC','NOW','NRG','NSC',
'NTAP','NTRS','NUE','NVDA','NVR','NWL','NWS','NWSA','NXPI','O',
'ODFL','OKE','OMC','ON','ORCL','ORLY','OTIS','OXY','PAYC','PAYX',
'PCAR','PCG','PEAK','PEG','PEP','PFE','PFG','PG','PGR','PH',
'PHM','PKG','PLD','PM','PNC','PNR','PNW','PODD','POOL','PPG',
'PPL','PRU','PSA','PSX','PTC','PWR','PXD','PYPL','QCOM','QRVO',
'RCL','REG','REGN','RF','RHI','RJF','RL','RMD','ROK','ROL',
'ROP','ROST','RSG','RTX','SBAC','SBUX','SCHW','SEDG','SEE','SHW',
'SIVB','SJM','SLB','SNA','SNPS','SO','SPG','SPGI','SRE','STE',
'STT','STX','STZ','SWK','SWKS','SYF','SYK','SYY','T','TAP',
'TDG','TDY','TECH','TEL','TER','TFC','TFX','TGT','TJX','TMO',
'TMUS','TPR','TRGP','TRMB','TROW','TRV','TSCO','TSLA','TSN','TT',
'TTWO','TXN','TXT','TYL','UAL','UDR','UHS','ULTA','UNH','UNP',
'UPS','URI','USB','V','VFC','VICI','VLO','VMC','VNO','VRSK',
'VRSN','VRTX','VTR','VTRS','VZ','WAB','WAT','WBA','WBD','WDC',
'WEC','WELL','WFC','WHR','WM','WMB','WMT','WRB','WRK','WST',
'WTW','WY','WYNN','XEL','XOM','XRAY','XYL','YUM','ZBH','ZBRA','ZTS'
]


# ── BLACK-SCHOLES ─────────────────────────────────────────────────

def bs_put_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_put_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1) - 1
    except:
        return None


def implied_vol(market_price, S, K, T, r):
    if T <= 0 or market_price <= 0:
        return None
    try:
        if market_price <= max(K - S, 0) + 0.01:
            return None
        def f(sigma):
            return bs_put_price(S, K, T, r, sigma) - market_price
        return brentq(f, 0.01, 5.0, xtol=1e-5, maxiter=100)
    except:
        return None


# ── VOLATILITY ────────────────────────────────────────────────────

def get_hv_and_ivr(ticker_obj, window=30):
    try:
        hist = ticker_obj.history(period='1y')
        if len(hist) < window + 10:
            return None, None
        returns = hist['Close'].pct_change().dropna()
        hv_current = returns.tail(window).std() * np.sqrt(252)
        rolling_hv = returns.rolling(window).std().dropna() * np.sqrt(252)
        hv_max = rolling_hv.max()
        hv_min = rolling_hv.min()
        if hv_max == hv_min:
            return hv_current, None
        ivr = (hv_current - hv_min) / (hv_max - hv_min) * 100
        return hv_current, round(ivr, 1)
    except:
        return None, None


def get_atm_iv(puts, S, T, r):
    try:
        p = puts.copy()
        p['dist'] = abs(p['strike'] - S)
        atm = p.nsmallest(1, 'dist').iloc[0]
        bid = atm.get('bid', 0)
        ask = atm.get('ask', 0)
        if bid <= 0 or ask <= 0:
            return None
        mid = (bid + ask) / 2
        return implied_vol(mid, S, atm['strike'], T, r)
    except:
        return None


# ── SPREAD FINDER ─────────────────────────────────────────────────

def find_best_spread(S, iv_for_delta, T, puts):
    """Find best spread using MID price for credits."""
    for _, put in puts.iterrows():
        K_short  = put['strike']
        s_bid    = put.get('bid', 0)
        s_ask    = put.get('ask', 0)
        if s_bid <= 0 or s_ask <= 0:
            continue
        short_mid = (s_bid + s_ask) / 2

        delta = bs_put_delta(S, K_short, T, RISK_FREE_RATE, iv_for_delta)
        if delta is None:
            continue
        abs_delta = abs(delta)

        if not (0.22 <= abs_delta <= 0.40):
            continue

        for width in [2, 3, 5, 10]:
            K_long    = K_short - width
            long_rows = puts[puts['strike'] == K_long]
            if long_rows.empty:
                continue

            l_bid = long_rows.iloc[0].get('bid', 0)
            l_ask = long_rows.iloc[0].get('ask', 0)
            if l_bid <= 0 or l_ask <= 0:
                continue
            long_mid = (l_bid + l_ask) / 2

            credit       = short_mid - long_mid
            if credit < 0.10:
                continue

            credit_ratio = credit / width
            breakeven    = K_short - credit
            buffer_pct   = (S - breakeven) / S * 100

            if credit_ratio >= 0.33 and buffer_pct >= 5:
                return {
                    'short_strike': K_short,
                    'long_strike':  K_long,
                    'width':        width,
                    'delta':        round(abs_delta, 2),
                    'credit':       round(credit, 2),
                    'credit_ratio': round(credit_ratio * 100, 1),
                    'breakeven':    round(breakeven, 2),
                    'buffer':       round(buffer_pct, 1),
                    'max_profit':   int(credit * 100),
                    'max_loss':     int((width - credit) * 100),
                }
    return None


# ── MAIN SCREEN ───────────────────────────────────────────────────

def run_screen():
    today   = datetime.today()
    results = []

    print(f"Bull Put Spread Screener — {today.strftime('%Y-%m-%d')}")
    print(f"Credits: MID price | IV/HV min 1.2x | IVR min 50% | C/W min 33%\n")

    for ticker in TICKERS:
        try:
            t = yf.Ticker(ticker)

            hist = t.history(period='5d')
            if hist.empty:
                continue
            S = hist['Close'].iloc[-1]
            if S <= 0:
                continue

            hv, ivr = get_hv_and_ivr(t)
            if hv is None or ivr is None or ivr < 50:
                continue

            exps = t.options
            if not exps:
                continue
            target_exp = target_dte = None
            for exp in exps:
                dte = (datetime.strptime(exp, '%Y-%m-%d') - today).days
                if 28 <= dte <= 50:
                    target_exp = exp
                    target_dte = dte
                    break
            if not target_exp:
                continue

            T     = target_dte / 365
            chain = t.option_chain(target_exp)
            puts  = chain.puts
            if puts.empty:
                continue

            atm_iv = get_atm_iv(puts, S, T, RISK_FREE_RATE)

            if atm_iv and atm_iv > 0:
                iv_for_delta = atm_iv
                iv_pct       = round(atm_iv * 100, 1)
                iv_hv_ratio  = round(atm_iv / hv, 2)
            else:
                iv_for_delta = hv
                iv_pct       = round(hv * 100, 1)
                iv_hv_ratio  = round(1.0, 2)

            # IV/HV filter
            if iv_hv_ratio < 1.2:
                print(f"  [{ticker}] Skip — IV/HV {iv_hv_ratio}x")
                continue

            spread = find_best_spread(S, iv_for_delta, T, puts)

            if spread:
                results.append({
                    'ticker':      ticker,
                    'price':       round(S, 2),
                    'ivr':         ivr,
                    'iv_pct':      iv_pct,
                    'hv_pct':      round(hv * 100, 1),
                    'iv_hv_ratio': iv_hv_ratio,
                    'dte':         target_dte,
                    'expiry':      target_exp,
                    **spread,
                })
                print(f"  [{ticker}] ✓ ${spread['short_strike']}/${spread['long_strike']} "
                      f"Credit:${spread['credit']} C/W:{spread['credit_ratio']}% "
                      f"IV/HV:{iv_hv_ratio}x Buffer:{spread['buffer']}%")
            else:
                print(f"  [{ticker}] No qualifying spread")

            time.sleep(0.3)

        except Exception as e:
            print(f"  [{ticker}] Error: {e}")
            continue

    results.sort(key=lambda x: (-x['iv_hv_ratio'], -x['ivr']))
    print(f"\nDone. {len(results)} opportunities found.")
    return results


# ── EMAIL ─────────────────────────────────────────────────────────

def build_email(results, date_str):
    th   = 'style="padding:8px 12px;font-size:9px;color:#7a7870;text-transform:uppercase;letter-spacing:.08em;font-weight:400;white-space:nowrap;border-bottom:1px solid #1e232b;text-align:right;"'
    th_l = 'style="padding:8px 12px;font-size:9px;color:#7a7870;text-transform:uppercase;letter-spacing:.08em;font-weight:400;white-space:nowrap;border-bottom:1px solid #1e232b;text-align:left;"'
    th_c = 'style="padding:8px 12px;font-size:9px;color:#7a7870;text-transform:uppercase;letter-spacing:.08em;font-weight:400;white-space:nowrap;border-bottom:1px solid #1e232b;text-align:center;"'

    rows = ""
    for i, r in enumerate(results[:15], 1):
        bg          = "#111418" if i % 2 == 1 else "#0f1317"
        ratio       = r['iv_hv_ratio']
        ratio_color = "#f0a500" if ratio >= 1.5 else "#3dba6e" if ratio >= 1.3 else "#7a7870"
        ivr_color   = "#f0a500" if r['ivr'] >= 80 else "#3dba6e" if r['ivr'] >= 65 else "#5b9cf6"
        cw_color    = "#3dba6e" if r['credit_ratio'] >= 45 else "#e8e6e0"

        td   = f'style="padding:9px 12px;font-family:monospace;font-size:12px;color:#e8e6e0;text-align:right;white-space:nowrap;background:{bg};"'
        td_l = f'style="padding:9px 12px;font-family:monospace;font-size:12px;font-weight:600;color:#e8e6e0;text-align:left;white-space:nowrap;background:{bg};"'
        td_c = f'style="padding:9px 12px;font-family:monospace;font-size:12px;color:#e8e6e0;text-align:center;white-space:nowrap;background:{bg};"'

        rows += f"""<tr>
          <td {td_l}>{i}. {r['ticker']}</td>
          <td {td}>${r['price']}</td>
          <td style="padding:9px 12px;font-family:monospace;font-size:12px;color:{ivr_color};text-align:right;background:{bg};">{r['ivr']}%</td>
          <td {td}>{r['iv_pct']}%</td>
          <td {td}>{r['hv_pct']}%</td>
          <td style="padding:9px 12px;font-family:monospace;font-size:12px;color:{ratio_color};text-align:right;font-weight:600;background:{bg};">{ratio}x</td>
          <td {td_c}>${r['short_strike']}/${r['long_strike']}</td>
          <td {td_c}>{r['delta']}</td>
          <td style="padding:9px 12px;font-family:monospace;font-size:12px;color:#3dba6e;text-align:right;font-weight:600;background:{bg};">${r['credit']}</td>
          <td style="padding:9px 12px;font-family:monospace;font-size:12px;color:{cw_color};text-align:right;background:{bg};">{r['credit_ratio']}%</td>
          <td {td}>${r['breakeven']}</td>
          <td {td}>{r['buffer']}%</td>
          <td style="padding:9px 12px;font-family:monospace;font-size:12px;color:#3dba6e;text-align:right;background:{bg};">${r['max_profit']}</td>
          <td style="padding:9px 12px;font-family:monospace;font-size:12px;color:#e05252;text-align:right;background:{bg};">-${r['max_loss']}</td>
          <td {td_c}>{r['dte']}d · {r['expiry']}</td>
        </tr>"""

    empty = "" if results else '<tr><td colspan="15" style="padding:40px;text-align:center;color:#7a7870;font-size:13px;">No opportunities met all criteria today.</td></tr>'

    return f"""<!DOCTYPE html><html>
<body style="background:#0a0c0f;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:20px;margin:0;">
<div style="max-width:1000px;margin:0 auto;">

  <div style="border-bottom:1px solid #1e232b;padding-bottom:12px;margin-bottom:14px;">
    <div style="font-family:monospace;font-size:15px;font-weight:500;color:#f0a500;letter-spacing:.06em;">BULL PUT SPREAD SCREENER</div>
    <div style="font-size:10px;color:#7a7870;letter-spacing:.1em;text-transform:uppercase;margin-top:2px;">
      {date_str} &nbsp;·&nbsp; {len(results)} opportunit{'ies' if len(results)!=1 else 'y'} &nbsp;·&nbsp; Sorted by IV/HV edge
    </div>
  </div>

  <div style="background:#111418;border:1px solid #1e232b;border-radius:3px;padding:10px 14px;margin-bottom:16px;font-size:11px;color:#7a7870;font-family:monospace;line-height:1.8;">
    Criteria: IVR &gt;50% &nbsp;|&nbsp; IV/HV &gt;1.2x &nbsp;|&nbsp; Delta 0.22–0.40 &nbsp;|&nbsp; C/W &gt;33% &nbsp;|&nbsp; Buffer &gt;5% &nbsp;|&nbsp; DTE 30–45<br>
    Credits shown at MID price — expect actual fills 10–15% below mid on liquid names
  </div>

  <div style="overflow-x:auto;">
  <table style="width:100%;border-collapse:collapse;font-size:12px;">
    <thead style="background:#0a0c0f;">
      <tr>
        <th {th_l}>Ticker</th>
        <th {th}>Price</th>
        <th {th}>IVR</th>
        <th {th}>IV</th>
        <th {th}>HV30</th>
        <th {th}>IV/HV ↓</th>
        <th {th_c}>Strikes</th>
        <th {th_c}>Delta</th>
        <th {th}>Credit*</th>
        <th {th}>C/W%</th>
        <th {th}>Breakeven</th>
        <th {th}>Buffer%</th>
        <th {th}>Max Profit*</th>
        <th {th}>Max Loss</th>
        <th {th_c}>Expiry</th>
      </tr>
    </thead>
    <tbody>{rows}{empty}</tbody>
  </table>
  </div>

  <div style="margin-top:14px;font-size:10px;color:#4a4840;line-height:1.8;">
    * Credit and Max Profit shown at mid price. Actual fills on Robinhood/broker typically 10–15% lower.<br>
    <strong style="color:#7a7870;">IV/HV</strong> = implied vol ÷ 30-day realized vol. Higher = options more overpriced vs actual movement = stronger selling edge.<br>
    <span style="color:#f0a500;">Orange</span> = strong edge (&gt;1.5x) &nbsp;·&nbsp; <span style="color:#3dba6e;">Green</span> = good edge (1.3–1.5x)<br>
    Always verify strikes and credits with your broker before trading. Not financial advice.
  </div>

</div></body></html>"""


def send_email(results):
    if not all([GMAIL_USER, GMAIL_APP_PASSWORD, EMAIL_RECIPIENT]):
        print("Email not configured — skipping")
        return
    date_str = datetime.today().strftime("%A %b %d, %Y")
    count    = len(results)
    subject  = f"Bull Put Spreads — {count} opportunit{'ies' if count!=1 else 'y'} — {datetime.today().strftime('%Y-%m-%d')}"
    html     = build_email(results, date_str)
    try:
        msg = MIMEMultipart("alternative")
        msg["From"]    = GMAIL_USER
        msg["To"]      = EMAIL_RECIPIENT
        msg["Subject"] = subject
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.send_message(msg)
        print(f"Email sent to {EMAIL_RECIPIENT} — {count} opportunities")
    except Exception as e:
        print(f"Email error: {e}")


if __name__ == "__main__":
    results = run_screen()
    send_email(results)
