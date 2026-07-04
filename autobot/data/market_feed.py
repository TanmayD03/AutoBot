"""
Supplementary live data feeds that Zerodha's Kite API does not provide:
FII/DII institutional flow, GIFT Nifty (NSE IX), and a low-latency US ADR
check. Per explicit instruction, these ARE scraped/sourced externally
(unlike option-chain data, which comes from Kite itself) because there is
no Kite-native equivalent for any of them.

Reliability notes (read before relying on this in a live session):
- MarketDataFeed hits nseindia.com directly. This is not a documented public
  API — it relies on session/cookie warmup to get past NSE's bot protection,
  and NSE can change or block this at any time without notice. Treat every
  call as "best effort", never as guaranteed.
- fetch_gift_nifty_live() (NSE's own marketStatus endpoint) frequently will
  NOT contain a GIFT Nifty / NSE IX entry — that data usually isn't broadcast
  there. GiftNiftyTracker (via tvDatafeed) is the primary path; NSE is only
  a fallback.
- GiftNiftyTracker uses tvDatafeed, an unofficial library that emulates a
  TradingView guest websocket client. Same caveat as above: unofficial,
  unsupported by TradingView, can break or get rate-limited without notice.
"""
import time
import requests
import logging


class MarketDataFeed:
    def __init__(self):
        # Master session for NSE to retain cookies and bypass the Akamai firewall
        self.nse_session = requests.Session()
        self.nse_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        self.nse_session.headers.update(self.nse_headers)

        logging.info("Initializing NSE session (FII/DII feed)...")
        try:
            self.nse_session.get("https://www.nseindia.com", timeout=10)
            time.sleep(1)  # let cookies settle
            self.nse_session.headers.update({"Accept": "application/json, text/plain, */*"})
        except Exception as e:
            logging.warning(f"Failed to initialize NSE session: {e}")

    def fetch_fii_dii_morning(self):
        """Run around 9:00 AM. Finalized institutional flow from the previous session."""
        url = "https://www.nseindia.com/api/fiidiiTradeReact"
        try:
            res = self.nse_session.get(url, timeout=10)
            if res.status_code == 200:
                return res.json()
            return f"FII/DII Error: {res.status_code}"
        except Exception as e:
            return f"Exception: {e}"

    def fetch_gift_nifty_live(self):
        """
        NSE-side fallback only — see module docstring. GiftNiftyTracker
        (tvDatafeed) is the primary source; this is rarely populated.
        """
        url = "https://www.nseindia.com/api/marketStatus"
        try:
            res = self.nse_session.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                for market in data.get('marketState', []):
                    if 'GIFT' in market.get('index', '').upper() or 'NSE IX' in market.get('index', '').upper():
                        return market
                return "GIFT Nifty data not currently broadcasted in the status endpoint. (Market closed/offline)"
            return f"GIFT Nifty Error: {res.status_code}"
        except Exception as e:
            return f"Exception: {e}"

    def fetch_us_adr_live(self, ticker="INFY"):
        """Direct Yahoo Finance chart JSON — lower latency than the yfinance library."""
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1m&range=1d"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                data = res.json()
                meta = data['chart']['result'][0]['meta']
                live_price = meta['regularMarketPrice']
                prev_close = meta['chartPreviousClose']
                return {
                    "ticker": ticker,
                    "live_price": live_price,
                    "previous_close": prev_close,
                    "percent_change": round(((live_price - prev_close) / prev_close) * 100, 2)
                }
            return f"ADR Error: {res.status_code}"
        except Exception as e:
            return f"Exception: {e}"


class MarketPreFlightMatrix:
    """Compiles FII/DII + US ADR data into a single pre-market bias score/regime."""

    def __init__(self, data_feed_instance):
        self.feed = data_feed_instance
        self.fii_net_cr = 0.0
        self.dii_net_cr = 0.0
        self.adr_bias_pct = 0.0
        self.overnight_bias_score = 0.0  # -1.0 to +1.0

    def generate_morning_matrix(self):
        """Run once, right before market open (e.g. ~9:08 AM IST)."""
        logging.info("Compiling pre-flight FII/DII + ADR matrix...")

        fii_raw = self.feed.fetch_fii_dii_morning()
        if isinstance(fii_raw, list):
            for record in fii_raw:
                if record.get('category') == 'FII':
                    self.fii_net_cr = float(record.get('netValue', 0.0))
                elif record.get('category') == 'DII':
                    self.dii_net_cr = float(record.get('netValue', 0.0))
        else:
            logging.warning(f"FII/DII fetch did not return data ({fii_raw}); defaulting to 0.0/0.0.")

        # Averaged across the same 3 ADRs the backtester's historical proxy
        # uses (INFY, HDB, IBN), so live and backtest stay comparable.
        adr_changes = []
        for ticker in ("INFY", "HDB", "IBN"):
            result = self.feed.fetch_us_adr_live(ticker)
            if isinstance(result, dict):
                adr_changes.append(result['percent_change'])
            else:
                logging.warning(f"ADR fetch failed for {ticker}: {result}")

        if adr_changes:
            self.adr_bias_pct = sum(adr_changes) / len(adr_changes)

        fii_factor = max(min(self.fii_net_cr / 2000.0, 1.0), -1.0)
        adr_factor = max(min(self.adr_bias_pct / 1.5, 1.0), -1.0)
        self.overnight_bias_score = round((fii_factor * 0.4) + (adr_factor * 0.6), 2)

        regime = ("BULLISH" if self.overnight_bias_score > 0.2
                  else "BEARISH" if self.overnight_bias_score < -0.2
                  else "RANGEBOUND")

        return {
            "fii_net_cr": self.fii_net_cr,
            "dii_net_cr": self.dii_net_cr,
            "adr_bias_pct": self.adr_bias_pct,
            "bias_score": self.overnight_bias_score,
            "regime": regime,
        }


class GiftNiftyTracker:
    """
    Live GIFT Nifty (NSE IX) via tvDatafeed — emulates a TradingView guest
    websocket client. Primary GIFT Nifty source; connects lazily so a
    missing/broken tvdatafeed install doesn't crash the whole bot.
    """

    def __init__(self):
        self.tv = None
        self._init_error = None
        try:
            from tvDatafeed import TvDatafeed  # noqa: local import — optional dependency
            logging.info("Connecting to TradingView data socket (guest) for GIFT Nifty...")
            self.tv = TvDatafeed()
        except Exception as e:
            self._init_error = str(e)
            logging.warning(f"GiftNiftyTracker init failed ({e}); GIFT Nifty will be unavailable this session.")

    def fetch_live_snapshot(self):
        """Returns {"live_price": float, "timestamp": ...} or {"error": str}."""
        if self.tv is None:
            return {"error": self._init_error or "tvDatafeed not initialized"}
        try:
            from tvDatafeed import Interval
            data = self.tv.get_hist(symbol='NIFTY1!', exchange='NSEIX',
                                    interval=Interval.in_1_minute, n_bars=1)
            if data is not None and not data.empty:
                latest = data.iloc[-1]
                return {"live_price": float(latest['close']), "timestamp": data.index[-1]}
            return {"error": "empty response from tvDatafeed"}
        except Exception as e:
            return {"error": str(e)}
