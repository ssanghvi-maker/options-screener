"""
bull_put_screener.py
Screens for bull put spread opportunities daily.
Criteria: IVR > 50, Delta 25-35, Credit/Width > 33%, Buffer > 5%, DTE 30-45
Sends results via email.
"""

import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import smtplib
import os
import time
import warnings
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────
GMAIL_USER         = os.environ.get("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
EMAIL_RECIPIENT    = os.environ.get("EMAIL_RECIPIENT", "")

RISK_FREE_RATE = 0.05

# Universe of liquid, high-IV candidates to screen
TICKERS = [
    # Tech / Semis
    'NVDA','AMD','INTC','MU','QCOM','AVGO','ARM','SMCI','TSM',
    # Mega cap tech
    'AAPL','MSFT','GOOGL','META','AMZN','NFLX','CRM','ORCL',
    # High beta / volatile
    'TSLA','COIN','MSTR','PLTR','RBLX','HOOD','SOFI','RIVN',
    # Broader market
    'SPY','QQQ','IWM',
    # Other liquid names
    'UBER','ABNB','SNAP','PINS','LYFT',
    'BA','CAT','GE','HON','LMT','RTX',
    'JPM','GS','MS','BAC','C',
    'XOM','CVX','OXY','SLB',
    'AMGN','GILD','MRNA','PFE',
]

# ── BLACK-SCHOLES ─────────────────────────────────────────────────

def bs_put_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    try:
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return norm.cdf(d1) - 1
    except:
        return None


# ── SCREENER ──────────────────────────────────────────────────────

def get_iv_rank(ticker_obj):
    """Calculate IV rank proxy from historical volatility."""
    try:
        hist = ticker_obj.history(period='1y')
        if len(hist) < 60:
            return None, None
        returns = hist['Close'].pct_change().dropna()
        # Rolling 20-day HV
        rolling_hv = returns.rolling(20).std().dropna() * np.sqrt(252) * 100
        if rolling_hv.empty:
            return None, None
        current_hv = rolling_hv.iloc[-1]
        hv_max = rolling_hv.max()
        hv_min = rolling_hv.min()
        if hv_max == hv_min:
            return None, None
        ivr = (current_hv - hv_min) / (hv_max - hv_min) * 100
        return round(ivr, 1), round(current_hv, 1)
    except:
        return None, None


def find_best_spread(ticker_obj, S, sigma, min_dte=28, max_dte=50):
    """Find the best bull put spread for a given stock."""
    try:
        expirations = ticker_obj.options
        if not expirations:
            return None

        # Find expiry in 30-45 DTE window
        target_exp = None
        target_dte = None
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d')
            dte = (exp_date - datetime.now()).days
            if min_dte <= dte <= max_dte:
                target_dte = dte
                target_exp = exp
                break

        if not target_exp:
            return None

        chain = ticker_obj.option_chain(target_exp)
        puts = chain.puts
        if puts.empty:
            return None

        T = target_dte / 365
        best = None

        for _, put in puts.iterrows():
            K_short = put['strike']
            bid = put.get('bid', 0)
            ask = put.get('ask', 0)
            if bid <= 0 or ask <= 0:
                continue
            short_mid = (bid + ask) / 2

            delta = bs_put_delta(S, K_short, T, RISK_FREE_RATE, sigma)
            if delta is None:
                continue
            abs_delta = abs(delta)

            # Target 25-35 delta for short strike
            if not (0.23 <= abs_delta <= 0.37):
                continue

            # Try $2, $3, $5 wide spreads
            for width in [2, 3, 5]:
                K_long = K_short - width
                long_puts = puts[puts['strike'] == K_long]
                if long_puts.empty:
                    continue

                lb = long_puts.iloc[0].get('bid', 0)
                la = long_puts.iloc[0].get('ask', 0)
                if lb <= 0 or la <= 0:
                    continue
                long_mid = (lb + la) / 2

                credit = short_mid - long_mid
                if credit <= 0:
                    continue

                credit_ratio = credit / width
                breakeven = K_short - credit
                buffer_pct = (S - breakeven) / S * 100

                # Apply all criteria
                if credit_ratio >= 0.33 and buffer_pct >= 5:
                    best = {
                        'expiry': target_exp,
                        'dte': target_dte,
                        'short_strike': K_short,
                        'long_strike': K_long,
                        'width': width,
                        'delta': round(abs_delta, 2),
                        'credit': round(credit, 2),
                        'credit_ratio': round(credit_ratio * 100, 1),
                        'breakeven': round(breakeven, 2),
                        'buffer': round(buffer_pct, 1),
                        'max_profit': round(credit * 100, 0),
                        'max_loss': round((width - credit) * 100, 0),
                    }
                    break  # take first passing spread

            if best:
                break

        return best
    except:
        return None


def run_screen():
    today = datetime.today()
    results = []

    print(f"Bull Put Spread Screener — {today.strftime('%Y-%m-%d')}")
    print(f"Universe: {len(TICKERS)} tickers\n")

    for ticker in TICKERS:
        try:
            t = yf.Ticker(ticker)

            # Get current price
            hist = t.history(period='5d')
            if hist.empty:
                continue
            S = hist['Close'].iloc[-1]
            if S <= 0:
                continue

            # Get IV rank
            ivr, hv = get_iv_rank(t)
            if ivr is None or ivr < 50:
                continue

            sigma = hv / 100 if hv else 0.30

            print(f"  [{ticker}] Price: ${S:.2f} | IVR: {ivr}% | HV: {hv}% — scanning options...")

            # Find best spread
            spread = find_best_spread(t, S, sigma)
            if spread:
                results.append({
                    'ticker': ticker,
                    'price': round(S, 2),
                    'ivr': ivr,
                    'hv': hv,
                    **spread
                })
                print(f"  [{ticker}] ✓ Found: ${spread['short_strike']}/${spread['long_strike']} | Credit: ${spread['credit']} | C/W: {spread['credit_ratio']}% | Buffer: {spread['buffer']}%")
            else:
                print(f"  [{ticker}] — No qualifying spread found")

            time.sleep(0.3)  # be polite to Yahoo Finance

        except Exception as e:
            print(f"  [{ticker}] Error: {e}")
            continue

    # Sort by IVR descending
    results.sort(key=lambda x: x['ivr'], reverse=True)
    print(f"\nFound {len(results)} qualifying opportunities.")
    return results


# ── EMAIL ─────────────────────────────────────────────────────────

def build_email(results, date_str):
    def score_badge(ivr):
        if ivr >= 80: return "#f0a500", "🔥 Very High IV"
        if ivr >= 65: return "#3dba6e", "✓ High IV"
        return "#5b9cf6", "Moderate IV"

    cards = ""
    for i, r in enumerate(results[:10], 1):  # top 10 only
        color, badge = score_badge(r['ivr'])
        rr_ratio = round(r['max_profit'] / r['max_loss'], 2) if r['max_loss'] > 0 else 0

        cards += f"""
        <div style="border:1px solid #1e232b;border-left:3px solid {color};border-radius:4px;
                    padding:18px;margin-bottom:14px;background:#111418;">

          <div style="display:flex;align-items:center;justify-content:space-between;
                      margin-bottom:12px;flex-wrap:wrap;gap:8px;">
            <div style="display:flex;align-items:center;gap:10px;">
              <span style="font-family:monospace;font-size:18px;font-weight:600;
                           color:#e8e6e0;">#{i} {r['ticker']}</span>
              <span style="font-size:12px;color:#7a7870;">@ ${r['price']}</span>
            </div>
            <span style="font-size:11px;color:{color};background:rgba(0,0,0,0.3);
                         padding:3px 10px;border-radius:2px;font-family:monospace;">
              IVR {r['ivr']}% — {badge}
            </span>
          </div>

          <div style="background:#0a0c0f;border-radius:4px;padding:12px 16px;
                      margin-bottom:12px;font-family:monospace;">
            <div style="font-size:14px;color:#f0a500;margin-bottom:4px;">
              Sell ${r['short_strike']} Put / Buy ${r['long_strike']} Put
            </div>
            <div style="font-size:11px;color:#7a7870;">
              Expires {r['expiry']} · {r['dte']} days · ${r['width']} wide · Delta {r['delta']}
            </div>
          </div>

          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px;">
            <div style="background:#1e232b;padding:8px 10px;border-radius:3px;">
              <div style="font-size:9px;color:#7a7870;text-transform:uppercase;
                          letter-spacing:.08em;margin-bottom:3px;">Collect</div>
              <div style="font-family:monospace;font-size:15px;color:#3dba6e;
                          font-weight:500;">${r['credit']}</div>
              <div style="font-size:10px;color:#7a7870;">${r['max_profit']}/contract</div>
            </div>
            <div style="background:#1e232b;padding:8px 10px;border-radius:3px;">
              <div style="font-size:9px;color:#7a7870;text-transform:uppercase;
                          letter-spacing:.08em;margin-bottom:3px;">Max loss</div>
              <div style="font-family:monospace;font-size:15px;color:#e05252;
                          font-weight:500;">-${r['max_loss']}</div>
              <div style="font-size:10px;color:#7a7870;">per contract</div>
            </div>
            <div style="background:#1e232b;padding:8px 10px;border-radius:3px;">
              <div style="font-size:9px;color:#7a7870;text-transform:uppercase;
                          letter-spacing:.08em;margin-bottom:3px;">Credit/Width</div>
              <div style="font-family:monospace;font-size:15px;color:#e8e6e0;
                          font-weight:500;">{r['credit_ratio']}%</div>
              <div style="font-size:10px;color:#7a7870;">min 33%</div>
            </div>
            <div style="background:#1e232b;padding:8px 10px;border-radius:3px;">
              <div style="font-size:9px;color:#7a7870;text-transform:uppercase;
                          letter-spacing:.08em;margin-bottom:3px;">Buffer</div>
              <div style="font-family:monospace;font-size:15px;color:#e8e6e0;
                          font-weight:500;">{r['buffer']}%</div>
              <div style="font-size:10px;color:#7a7870;">to breakeven</div>
            </div>
          </div>

          <div style="font-size:11px;color:#7a7870;">
            Breakeven: <span style="color:#e8e6e0;font-family:monospace;">${r['breakeven']}</span>
            &nbsp;·&nbsp;
            R/R: <span style="color:#e8e6e0;font-family:monospace;">{rr_ratio}x</span>
            &nbsp;·&nbsp;
            HV30: <span style="color:#e8e6e0;font-family:monospace;">{r['hv']}%</span>
          </div>
        </div>"""

    no_results = "" if results else """
        <div style="text-align:center;padding:40px;color:#7a7870;font-size:13px;">
          No opportunities met all criteria today. Check again tomorrow.
        </div>"""

    return f"""<!DOCTYPE html>
<html><body style="background:#0a0c0f;font-family:-apple-system,BlinkMacSystemFont,
    'Segoe UI',sans-serif;padding:24px;max-width:680px;margin:0 auto;">

  <div style="border-bottom:1px solid #1e232b;padding-bottom:14px;margin-bottom:20px;">
    <div style="font-family:monospace;font-size:15px;font-weight:500;color:#f0a500;
                letter-spacing:.06em;">BULL PUT SPREAD SCREENER</div>
    <div style="font-size:10px;color:#7a7870;letter-spacing:.1em;text-transform:uppercase;
                margin-top:2px;">
      {date_str} &nbsp;·&nbsp; {len(results)} opportunities found
    </div>
  </div>

  <div style="background:#111418;border:1px solid #1e232b;border-radius:4px;
              padding:12px 16px;margin-bottom:18px;font-size:12px;color:#7a7870;
              line-height:1.7;">
    <strong style="color:#e8e6e0;">Criteria applied:</strong>
    IVR &gt; 50% &nbsp;·&nbsp; Delta 25–35 &nbsp;·&nbsp;
    Credit/Width &gt; 33% &nbsp;·&nbsp; Buffer &gt; 5% &nbsp;·&nbsp; DTE 30–45
  </div>

  {cards}{no_results}

  <div style="border-top:1px solid #1e232b;margin-top:20px;padding-top:10px;
              font-size:10px;color:#4a4840;line-height:1.6;">
    Always verify prices with your broker before trading.
    Options data from Yahoo Finance — may be delayed 15 minutes.
    This is not financial advice.
  </div>
</body></html>"""


def send_email(results):
    if not all([GMAIL_USER, GMAIL_APP_PASSWORD, EMAIL_RECIPIENT]):
        print("Email not configured — skipping")
        return

    date_str = datetime.today().strftime("%A %b %d, %Y")
    count = len(results)
    subject = f"Bull Put Spreads — {count} opportunit{'ies' if count != 1 else 'y'} today — {datetime.today().strftime('%Y-%m-%d')}"

    html = build_email(results, date_str)

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


# ── MAIN ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_screen()
    send_email(results)
