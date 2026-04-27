# Bull Put Spread Screener

Daily email digest of the best bull put spread opportunities across a universe of liquid US stocks.

## What it does

Runs every weekday at 9:30am ET (market open). Screens ~60 liquid tickers for:
- **IVR > 50%** — options are expensive relative to their history
- **Delta 25–35** — appropriate strike selection for high IV environments  
- **Credit/Width > 33%** — collecting at least 1/3 of the spread width
- **Buffer > 5%** — breakeven at least 5% below current price
- **DTE 30–45** — sweet spot for theta decay

Sends a clean HTML email with the top opportunities ranked by IV Rank.

## Setup

### 1. Add GitHub Secrets

Go to your repo → Settings → Secrets and variables → Actions → New repository secret

Add these three secrets (same values as your other repos):
- `GMAIL_USER` — your Gmail address
- `GMAIL_APP_PASSWORD` — your Gmail app password
- `EMAIL_RECIPIENT` — where to send the email

### 2. Upload files

```
options-screener/
├── bull_put_screener.py
└── .github/
    └── workflows/
        └── run_screener.yml
```

### 3. Run manually first

Actions → Bull Put Spread Screener → Run workflow

Check your inbox within 2-3 minutes.

## Notes

- Options data from Yahoo Finance (free, no API key needed)
- Delta calculated using Black-Scholes with 30-day historical volatility as IV proxy
- Always verify strikes and premiums with your broker before trading
- Not financial advice
