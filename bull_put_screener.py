"""
bull_put_screener.py — CLEAN VERSION

Includes:
- S&P 500 universe (CSV source)
- IV/HV >= 1.4
- Delta 0.18–0.28
- Buffer >= 7%
- Earnings filter (skip next 10 days only)
"""

import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime
import time
import warnings


warnings.filterwarnings('ignore')
import os

GMAIL_USER         = os.environ.get("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
EMAIL_RECIPIENT    = os.environ.get("EMAIL_RECIPIENT", "")
RISK_FREE_RATE = 0.05


# ─────────────────────────────
# S&P 500 TICKERS
# ─────────────────────────────

def get_sp500_tickers():
    url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    df = pd.read_csv(url)
    return [t.replace(".", "-") for t in df["Symbol"].tolist()]

TICKERS = get_sp500_tickers()


# ─────────────────────────────
# BLACK-SCHOLES
# ─────────────────────────────

def bs_put_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_put_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return None
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1


def implied_vol(price, S, K, T, r):
    if T <= 0 or price <= 0:
        return None
    try:
        def f(sig):
            return bs_put_price(S, K, T, r, sig) - price
        return brentq(f, 0.01, 5.0, maxiter=100)
    except:
        return None


# ─────────────────────────────
# VOLATILITY
# ─────────────────────────────

def get_hv_and_ivr(ticker):
    try:
        hist = ticker.history(period='1y')
        returns = hist['Close'].pct_change().dropna()

        hv = returns.tail(30).std() * np.sqrt(252)

        rolling = returns.rolling(30).std().dropna() * np.sqrt(252)
        hv_max, hv_min = rolling.max(), rolling.min()

        if hv_max == hv_min:
            return None, None

        ivr = (hv - hv_min) / (hv_max - hv_min) * 100
        return hv, ivr
    except:
        return None, None


def get_atm_iv(puts, S, T):
    try:
        puts = puts.copy()
        puts["dist"] = abs(puts["strike"] - S)
        atm = puts.nsmallest(1, "dist").iloc[0]

        bid, ask = atm["bid"], atm["ask"]
        if bid <= 0 or ask <= 0:
            return None

        mid = (bid + ask) / 2
        return implied_vol(mid, S, atm["strike"], T, RISK_FREE_RATE)
    except:
        return None


# ─────────────────────────────
# EARNINGS
# ─────────────────────────────

def get_next_earnings_date(t):
    try:
        cal = t.calendar

        if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
            val = cal.loc["Earnings Date"].values[0]
            return pd.to_datetime(val)

        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed:
                return pd.to_datetime(ed[0] if isinstance(ed, (list, tuple)) else ed)

    except:
        pass

    return None


# ─────────────────────────────
# SPREAD FINDER
# ─────────────────────────────

def find_spread(S, iv, T, puts):

    for _, put in puts.iterrows():
        K_short = put["strike"]
        bid, ask = put["bid"], put["ask"]

        if bid <= 0 or ask <= 0:
            continue

        short_mid = (bid + ask) / 2

        delta = bs_put_delta(S, K_short, T, RISK_FREE_RATE, iv)
        if delta is None:
            continue

        if not (0.18 <= abs(delta) <= 0.28):
            continue

        for width in [2, 3, 5, 10]:
            K_long = K_short - width
            row = puts[puts["strike"] == K_long]

            if row.empty:
                continue

            lb, la = row.iloc[0]["bid"], row.iloc[0]["ask"]
            if lb <= 0 or la <= 0:
                continue

            credit = short_mid - (lb + la) / 2
            if credit < 0.10:
                continue

            breakeven = K_short - credit
            buffer = (S - breakeven) / S * 100
            cw = credit / width

            if cw >= 0.33 and buffer >= 7:
                return {
                    "short": K_short,
                    "long": K_long,
                    "delta": round(abs(delta), 2),
                    "credit": round(credit, 2),
                    "cw": round(cw * 100, 1),
                    "buffer": round(buffer, 1),
                }

    return None


# ─────────────────────────────
# MAIN
# ─────────────────────────────

def run_screen():

    today = datetime.today()
    today_date = today.date()

    results = []

    print(f"Screener running — {today}")

    for ticker in TICKERS:

        try:
            t = yf.Ticker(ticker)

            hist = t.history(period="5d")
            if hist.empty:
                continue

            S = hist["Close"].iloc[-1]

            hv, ivr = get_hv_and_ivr(t)
            if hv is None or ivr is None or ivr < 50:
                continue

            exps = t.options
            if not exps:
                continue

            target = None
            dte = None

            for e in exps:
                d = (datetime.strptime(e, "%Y-%m-%d") - today).days
                if 28 <= d <= 50:
                    target = e
                    dte = d
                    break

            if not target:
                continue

            # ✅ EARNINGS FILTER (10 DAYS ONLY)
            earnings_date = get_next_earnings_date(t)

            if earnings_date is not None:
                ed = earnings_date.date()
                days = (ed - today_date).days

                if 0 <= days <= 10:
                    continue

            chain = t.option_chain(target)
            puts = chain.puts
            if puts.empty:
                continue

            T = dte / 365

            iv = get_atm_iv(puts, S, T)
            if not iv:
                continue

            iv_hv = iv / hv

            if iv_hv < 1.4:
                continue

            spread = find_spread(S, iv, T, puts)

            if spread:
                results.append({
                    "ticker": ticker,
                    "price": round(S, 2),
                    "ivr": round(ivr, 1),
                    "iv_hv": round(iv_hv, 2),
                    **spread
                })

            time.sleep(0.1)

        except:
            continue

    results.sort(key=lambda x: -x["iv_hv"])

    print(f"\nDone — {len(results)} trades found")

    for r in results[:10]:
        print(r)

    return results


if __name__ == "__main__":
    run_screen()

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
