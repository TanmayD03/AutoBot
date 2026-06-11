"""Tiered data ingestion.
Tier 1 (lowest latency): broker WebSocket — see execution/kotak_neo.py live feed.
Tier 2: NSE public endpoints (option chain, VIX, FII/DII).
Tier 3: yfinance global macro.
Pluggable: implement DataAdapter to add paid low-latency vendors (TrueData etc.).
"""
import time
import requests

MACRO_TICKERS = {
    "sp500": "^GSPC", "nasdaq": "^IXIC", "dow": "^DJI", "nikkei": "^N225",
    "kospi": "^KS11", "brent": "BZ=F", "dxy": "DX-Y.NYB", "usdinr": "USDINR=X",
    "us10y": "^TNX", "india_vix": "^INDIAVIX", "nifty": "^NSEI",
    "adr_infy": "INFY", "adr_wit": "WIT", "adr_hdb": "HDB", "adr_ibn": "IBN",
}


class DataAdapter:
    """Interface for pluggable feeds (subclass for TrueData/GlobalDatafeeds)."""
    def snapshot(self) -> dict:
        raise NotImplementedError


class MacroFeed(DataAdapter):
    """Tier 3: global macro daily % changes via yfinance."""
    def snapshot(self) -> dict:
        import yfinance as yf
        out = {}
        data = yf.download(list(MACRO_TICKERS.values()), period="5d", interval="1d",
                           progress=False, group_by="ticker", auto_adjust=True)
        for name, tkr in MACRO_TICKERS.items():
            try:
                closes = data[tkr]["Close"].dropna()
                out[name] = {"last": float(closes.iloc[-1]),
                             "chg_pct": float((closes.iloc[-1] / closes.iloc[-2] - 1) * 100)}
            except Exception:
                out[name] = {"last": None, "chg_pct": 0.0}
        return out


class NSEClient(DataAdapter):
    """Tier 2: NSE public endpoints with session/cookie warm-up and rate limiting."""
    BASE = "https://www.nseindia.com"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept": "application/json", "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain",
    }

    def __init__(self, min_interval=3.0):
        self.sess = requests.Session()
        self.sess.headers.update(self.HEADERS)
        self.min_interval = min_interval
        self._last = 0.0
        self._warm = False

    def _get(self, path):
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        if not self._warm:
            self.sess.get(self.BASE, timeout=10)
            self._warm = True
        r = self.sess.get(self.BASE + path, timeout=10)
        self._last = time.time()
        r.raise_for_status()
        return r.json()

    def option_chain(self, symbol="NIFTY"):
        """Returns (spot, chain) rows: strike, call_oi, put_oi, chg OI, IV, LTP."""
        js = self._get(f"/api/option-chain-indices?symbol={symbol}")
        spot = js["records"]["underlyingValue"]
        expiry = js["records"]["expiryDates"][0]
        chain = []
        for row in js["records"]["data"]:
            if row.get("expiryDate") != expiry:
                continue
            ce, pe = row.get("CE", {}), row.get("PE", {})
            chain.append({
                "strike": row["strikePrice"],
                "call_oi": ce.get("openInterest", 0), "put_oi": pe.get("openInterest", 0),
                "call_chg_oi": ce.get("changeinOpenInterest", 0),
                "put_chg_oi": pe.get("changeinOpenInterest", 0),
                "call_iv": ce.get("impliedVolatility", 0), "put_iv": pe.get("impliedVolatility", 0),
                "call_ltp": ce.get("lastPrice", 0), "put_ltp": pe.get("lastPrice", 0),
            })
        return spot, chain

    def snapshot(self):
        spot, chain = self.option_chain()
        return {"spot": spot, "chain": chain}
