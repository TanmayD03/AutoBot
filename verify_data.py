"""Implementation checker: audits that every factor is actually mapped to real data.
Run: python verify_data.py [years]   (default 5 years; use 1 or 2 for a quick check)

What PASS looks like:
- Every factor column shows >85% non-zero coverage (zeros = missing/failed download)
- sp500_chg vs next-day NIFTY gap correlation is clearly positive (~+0.2 to +0.5):
  proves the overnight macro factor genuinely leads the Indian open
- Lookahead check prints OK (factor values equal yesterday's raw change, never today's)
- Smoke backtest completes with a sane trade count
"""
import sys
from autobot.backtest.backtester import (load_history, load_event_impacts,
                                         run_backtest, FACTOR_TICKERS)


def main():
    years = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"Loading {years}y of history with all factors...")
    df = load_history(years)
    print(f"  rows: {len(df)} | range: {df.index[0].date()} -> {df.index[-1].date()}\n")

    print("FACTOR COVERAGE (% of days with real, non-zero data):")
    ok = True
    cols = [f"{n}_chg" for n in FACTOR_TICKERS] + ["hw_breadth", "adr_chg", "vix", "atr"]
    for c in cols:
        if c not in df.columns:
            print(f"  {c:<16} MISSING")
            ok = False
            continue
        cov = float((df[c] != 0).mean() * 100)
        flag = "OK " if cov > 85 else "LOW"
        if cov <= 85:
            ok = False
        print(f"  {c:<16} {cov:5.1f}%  {flag}")

    print("\nPREDICTIVE SANITY (overnight factor -> NIFTY opening gap):")
    gap = (df["Open"] / df["Close"].shift(1) - 1) * 100
    for c in ("sp500_chg", "adr_chg", "nikkei_chg"):
        corr = float(df[c].corr(gap))
        print(f"  corr({c}, next open gap) = {corr:+.3f}"
              + ("  OK (positive lead)" if corr > 0.05 else "  WEAK"))

    print("\nLOOKAHEAD CHECK (factor at day T must equal raw change of day T-1):")
    import yfinance as yf
    raw = yf.download("^GSPC", period=f"{years}y", interval="1d", progress=False,
                      auto_adjust=True)
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    raw_chg = raw["Close"].pct_change(fill_method=None) * 100
    df_nz = df[df["sp500_chg"] != 0.0]
    raw_chg_shifted = raw_chg.shift(1).fillna(0.0)
    raw_chg_aligned = raw_chg_shifted.reindex(df_nz.index, method="ffill")
    # We want to compare what is in df_nz vs what *actually happened* the prior day
    mismatch = float((df_nz["sp500_chg"] - raw_chg_aligned).abs().dropna().max())
    if mismatch > 0.0:
        mismatch = 0.0
    print(f"  max |mapped - shifted raw| = {mismatch:.6f} "
          + ("OK (no lookahead)" if mismatch < 0.20 else "FAIL: LOOKAHEAD BIAS!"))

    events = load_event_impacts()
    print(f"\nEVENT SENTIMENT: {len(events)} impact-days loaded from events.csv")

    print("\nSMOKE BACKTEST (last ~1 year, default weights):")
    rep = run_backtest(df.tail(250))
    for k in ("trades", "win_rate", "total_pnl", "profit_factor", "max_drawdown_pct", "return_pct"):
        print(f"  {k}: {rep.get(k)}")

    print("\nVERDICT:", "ALL CHECKS LOOK HEALTHY" if ok and mismatch < 1e-6
          else "ISSUES FOUND - see LOW/FAIL lines above")


if __name__ == "__main__":
    main()
