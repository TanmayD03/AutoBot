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
