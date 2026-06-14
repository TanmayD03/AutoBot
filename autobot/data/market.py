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
    """
    Tiered macro feed with fallback:
    1. Try yfinance (often fails)
    2. Try investing.com scrape via requests
    3. Use last known values from cache
    """
    def __init__(self):
        self._cache = {}

    def snapshot(self) -> dict:
        out = {}
        # Try yfinance first
        try:
            import yfinance as yf
            data = yf.download(
                list(MACRO_TICKERS.values()),
                period="5d", interval="1d",
                progress=False, group_by="ticker", auto_adjust=True,
                timeout=8
            )
            for name, tkr in MACRO_TICKERS.items():
                try:
                    closes = data[tkr]["Close"].dropna()
                    if len(closes) >= 2:
                        val = {"last": float(closes.iloc[-1]),
                               "chg_pct": float((closes.iloc[-1]/closes.iloc[-2]-1)*100)}
                        out[name] = val
                        self._cache[name] = val   # update cache on success
                except Exception:
                    pass
        except Exception:
            pass

        # Fill missing with cache or zero
        for name in MACRO_TICKERS:
            if name not in out:
                out[name] = self._cache.get(name, {"last": None, "chg_pct": 0.0})

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
        from ..options_math.black_scholes import gamma_exposure, max_pain, put_call_ratio, get_option_walls

        gex = gamma_exposure(chain, spot, 0.068, 3/365.0, 0.15) if chain else 0.0
        pain = max_pain(chain) if chain else spot
        pcr = put_call_ratio(chain) if chain else 1.0
        call_wall, put_wall = get_option_walls(chain)

        return {
            "spot": spot,
            "chain": chain,
            "gex": gex,
            "max_pain": pain,
            "pcr": pcr,
            "call_wall": call_wall,
            "put_wall": put_wall
        }

    def fii_dii_flow(self) -> dict:
        """Pull today's FII/DII provisional net flow from NSE."""
        try:
            js = self._get("/api/fiidiiTradeReact")
            # Returns list of records: category, buyValue, sellValue, netValue
            fii = next(r for r in js if r["category"] == "FII/FPI")
            dii = next(r for r in js if r["category"] == "DII")
            return {
                "fii_net_cr": float(fii["netValue"]),   # ₹ crore
                "dii_net_cr": float(dii["netValue"]),
            }
        except Exception:
            return {"fii_net_cr": 0.0, "dii_net_cr": 0.0}
