import logging
import pandas as pd
from datetime import datetime
from kiteconnect import KiteConnect

class KiteAdapter:
    """Handles Zerodha API authentication, instrument mapping, and token resolution."""
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = KiteConnect(api_key=self.api_key)
        self.instruments_df = None

    def get_login_url(self) -> str:
        return self.kite.login_url()

    def set_access_token(self, request_token: str):
        data = self.kite.generate_session(request_token, api_secret=self.api_secret)
        self.kite.set_access_token(data["access_token"])
        logging.info("Zerodha Kite session generated successfully.")

    def fetch_master_instruments(self):
        """Downloads the full NSE/NFO instrument list (must run daily before open)."""
        logging.info("Fetching master instrument list from Zerodha...")
        instruments = self.kite.instruments()
        self.instruments_df = pd.DataFrame(instruments)
        logging.info(f"Loaded {len(self.instruments_df)} instruments.")

    def get_token(self, tradingsymbol: str, exchange: str = "NFO") -> int:
        """Resolve a trading symbol like 'NIFTY24JUL24500CE' to its integer token."""
        if self.instruments_df is None:
            self.fetch_master_instruments()

        match = self.instruments_df[
            (self.instruments_df['tradingsymbol'] == tradingsymbol) &
            (self.instruments_df['exchange'] == exchange)
        ]

        if match.empty:
            raise ValueError(f"Instrument not found: {tradingsymbol} on {exchange}")

        return int(match.iloc[0]['instrument_token'])

    def find_nifty_option(self, strike: int, kind: str, expiry_date: datetime.date) -> dict:
        """
        Dynamically finds the NIFTY option symbol matching the strike, CE/PE kind,
        and exact expiry date. Expiry date format matching varies, so we filter by segment and name.
        """
        if self.instruments_df is None:
            self.fetch_master_instruments()

        df = self.instruments_df
        # Filter for NIFTY options
        opts = df[(df['name'] == 'NIFTY') & (df['segment'] == 'NFO-OPT')]
        # Filter for Strike and Type (CE or PE)
        opts = opts[(opts['strike'] == float(strike)) & (opts['instrument_type'] == kind)]

        # Exact date matching (ensure expiry column is date object or string matched)
        opts['expiry_date'] = pd.to_datetime(opts['expiry']).dt.date
        opts = opts[opts['expiry_date'] == expiry_date]

        if opts.empty:
            raise ValueError(f"No NIFTY {kind} found for strike {strike} expiring on {expiry_date}")

        row = opts.iloc[0]
        return {
            "tradingsymbol": row["tradingsymbol"],
            "instrument_token": int(row["instrument_token"]),
            "lot_size": int(row["lot_size"])
        }

    def get_option_chain(self, expiry_date: datetime.date, spot: float,
                          strike_range_count: int = 15, strike_step: int = 50) -> list:
        """
        Builds an option-chain snapshot directly from Zerodha Kite — batches
        kite.quote() across the strikes around spot to pull OI and LTP per
        strike. No NSE website scraping involved.

        Returns a list of dicts: {strike, call_oi, put_oi, call_ltp, put_ltp}
        — the schema autobot.options_math.black_scholes (max_pain,
        put_call_ratio, get_option_walls) and signals.engine (oi_walls_signal
        etc.) already expect.
        """
        if self.instruments_df is None:
            self.fetch_master_instruments()

        df = self.instruments_df
        opts = df[(df['name'] == 'NIFTY') & (df['segment'] == 'NFO-OPT')].copy()
        opts['expiry_date'] = pd.to_datetime(opts['expiry']).dt.date
        opts = opts[opts['expiry_date'] == expiry_date]

        atm = round(spot / strike_step) * strike_step
        lo = atm - strike_range_count * strike_step
        hi = atm + strike_range_count * strike_step
        opts = opts[(opts['strike'] >= lo) & (opts['strike'] <= hi)]

        if opts.empty:
            logging.warning(f"get_option_chain: no NIFTY strikes found for expiry {expiry_date} "
                            f"in range [{lo}, {hi}]")
            return []

        symbols = [f"NFO:{ts}" for ts in opts['tradingsymbol'].tolist()]

        # Kite's quote() accepts a batch of instruments, but we chunk
        # conservatively to stay well under any per-request instrument limit.
        quotes = {}
        CHUNK = 200
        for i in range(0, len(symbols), CHUNK):
            batch = symbols[i:i + CHUNK]
            try:
                quotes.update(self.kite.quote(batch))
            except Exception as e:
                logging.warning(f"get_option_chain: quote() batch failed ({e}); "
                                f"continuing with strikes fetched so far.")

        chain = {}
        for _, row in opts.iterrows():
            tsym = f"NFO:{row['tradingsymbol']}"
            q = quotes.get(tsym)
            if not q:
                continue
            strike = float(row['strike'])
            entry = chain.setdefault(strike, {"strike": strike, "call_oi": 0, "put_oi": 0,
                                                "call_ltp": 0.0, "put_ltp": 0.0})
            if row['instrument_type'] == 'CE':
                entry["call_oi"] = q.get("oi", 0)
                entry["call_ltp"] = q.get("last_price", 0.0)
            else:
                entry["put_oi"] = q.get("oi", 0)
                entry["put_ltp"] = q.get("last_price", 0.0)

        return sorted(chain.values(), key=lambda c: c["strike"])
