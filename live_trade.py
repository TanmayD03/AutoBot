import os
import sys
import time
import logging
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

import pandas as pd
from autobot.data.kite_adapter import KiteAdapter
from autobot.backtest.backtester import load_history, build_signals, days_to_nifty_expiry
from autobot.strategy.decision import DecisionEngine
from autobot.strategy.regime import RegimeDetector
from autobot.strategy.capital import CapitalManager
from autobot.strategy.trail import TrailManager
from autobot.execution.paper_broker import PaperBroker

from kiteconnect import KiteTicker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def get_next_thursday(today: date) -> date:
    """NIFTY weekly options typically expire on Thursday."""
    days_ahead = 3 - today.weekday()
    if days_ahead < 0: # Target next week
        days_ahead += 7
    return today + timedelta(days=days_ahead)

def setup_kite() -> KiteAdapter:
    load_dotenv()
    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")

    if not api_key or not api_secret:
        logging.error("Missing KITE_API_KEY or KITE_API_SECRET in environment variables.")
        sys.exit(1)

    kite_adapter = KiteAdapter(api_key, api_secret)

    print("\n" + "="*50)
    print("ZERODHA LOGIN REQUIRED")
    print("1. Go to this URL in your browser:")
    print(f"   {kite_adapter.get_login_url()}")
    print("2. Login and you will be redirected.")
    print("3. Copy the 'request_token' parameter from the redirected URL.")
    print("="*50 + "\n")

    req_token = input("Enter request_token: ").strip()
    kite_adapter.set_access_token(req_token)
    kite_adapter.fetch_master_instruments()
    return kite_adapter

class LivePaperTrader:
    def __init__(self, kite_adapter: KiteAdapter):
        self.kite = kite_adapter.kite
        self.adapter = kite_adapter
        self.broker = PaperBroker(capital=100000.0) # Rs 1 Lakh paper capital
        self.decision_engine = DecisionEngine()
        self.regime_detector = RegimeDetector()
        self.capital_manager = CapitalManager()
        self.trail_manager = None
        self.active_position = None
        self.option_meta = None
        self.spot_token = 256265 # Hardcoded standard token for NIFTY 50 index (NSE:NIFTY 50)
        self.option_token = None
        self.ticker = None

        # Real-time state
        self.latest_spot = None
        self.latest_opt_price = None
        self.peak_opt_price = 0.0

    def run_premarket_analysis(self) -> dict:
        """Download history and decide direction before market opens (based on overnight macro)."""
        logging.info("Downloading historical macro data to build pre-market signals...")
        df = load_history(1) # Load 1 year

        if len(df) < 4:
            logging.error("Not enough historical data to generate signals.")
            sys.exit(1)

        today = df.iloc[-1]
        prev = df.iloc[-2]
        prev2_close = df.iloc[-3].Close
        prev3_close = df.iloc[-4].Close if len(df) >= 4 else None

        signals = build_signals(prev, today, prev2_close, prev3_close)
        adx_val = getattr(today, "adx", 18.0)
        regime = self.regime_detector.classify(vix=today.vix, adx=adx_val, atr_pct=(today.atr / prev.Close) * 100)

        logging.info(f"Regime Detected: {regime}")
        for s in signals:
            logging.info(f"  Signal {s.name}: Score {s.score:.2f}, Conf {s.confidence:.2f}")

        # The actual 'spot' price isn't fully known until 09:15 open, but we use prev close to build the plan direction
        plan = self.decision_engine.decide(signals, prev.Close, sentiment_impact=0.0)

        if plan is None or plan.action == "NO_TRADE":
            logging.warning("Pre-market analysis dictates NO TRADE for today.")
            return None

        logging.info(f"PRE-MARKET PLAN: {plan.action} (Confidence: {plan.confidence:.2f})")
        return plan

    def wait_for_market_open(self):
        logging.info("Waiting for market open to capture spot price...")
        # Simplistic wait - in reality you might just poll the LTP of NIFTY 50 every 5 seconds until it updates for today
        while True:
            # Note: Hardcoding 'NSE:NIFTY 50' as the exchange symbol for spot. Check if correct.
            quote = self.kite.quote("NSE:NIFTY 50")
            if "NSE:NIFTY 50" in quote:
                spot = quote["NSE:NIFTY 50"]["last_price"]
                if spot > 0:
                    logging.info(f"Market is Open! Spot Price: {spot}")
                    self.latest_spot = spot
                    return spot
            time.sleep(2)

    def execute_paper_trade(self, plan, spot):
        """Map the plan to a specific option, size it, and buy via PaperBroker."""
        atm = round(spot / 50) * 50
        kind = "CE" if plan.action == "BUY_CE" else "PE"
        today_date = datetime.now().date()
        expiry_date = get_next_thursday(today_date)

        logging.info(f"Resolving instrument for NIFTY {atm} {kind} exp {expiry_date}...")
        try:
            self.option_meta = self.adapter.find_nifty_option(atm, kind, expiry_date)
            self.option_token = self.option_meta["instrument_token"]
        except Exception as e:
            logging.error(f"Failed to resolve option token: {e}")
            sys.exit(1)

        # Get entry premium via REST API quote before kicking off websocket
        tsym = "NFO:" + self.option_meta["tradingsymbol"]
        quote = self.kite.quote(tsym)
        entry_prem = quote[tsym]["last_price"]

        logging.info(f"Current Premium for {tsym}: Rs {entry_prem}")
        if entry_prem <= 0:
            logging.error("Invalid premium.")
            return

        # Simplified sizing for live paper trading script (can be connected to full delta sizing later)
        lots = self.capital_manager.calculate_lots(self.broker.capital, entry_prem)
        qty = lots * self.option_meta["lot_size"]

        disaster = max(1.0, entry_prem * 0.5) # Hardcoded disaster stop at 50% for live harness
        self.active_position = self.broker.buy(tsym, qty, entry_prem, stop=disaster, target=entry_prem*3)
        self.peak_opt_price = entry_prem

        # Initialize TrailManager
        is_expiry_day = (today_date == expiry_date)
        self.trail_manager = TrailManager(expiry_mode=is_expiry_day)

        logging.info(f"EXECUTED PAPER TRADE: Bought {qty} of {tsym} at {entry_prem}")

    def start_websocket(self):
        """Streams live prices and manages the trailing stop."""
        self.ticker = KiteTicker(self.adapter.api_key, self.kite.access_token)

        def on_ticks(ws, ticks):
            for tick in ticks:
                token = tick['instrument_token']
                ltp = tick['last_price']

                if token == self.spot_token:
                    self.latest_spot = ltp
                elif token == self.option_token:
                    self.latest_opt_price = ltp
                    self.peak_opt_price = max(self.peak_opt_price, ltp)
                    self.evaluate_trail(ltp)

        def on_connect(ws, response):
            logging.info("WebSocket connected. Subscribing to tokens...")
            ws.subscribe([self.spot_token, self.option_token])
            ws.set_mode(ws.MODE_LTP, [self.spot_token, self.option_token])

        def on_close(ws, code, reason):
            logging.info("WebSocket closed.")

        self.ticker.on_ticks = on_ticks
        self.ticker.on_connect = on_connect
        self.ticker.on_close = on_close

        logging.info("Starting WebSocket stream... (Press Ctrl+C to stop)")
        self.ticker.connect(threaded=False) # Blocking loop

    def evaluate_trail(self, current_prem: float):
        if not self.active_position:
            return

        action, _ = self.trail_manager.update(current_prem, self.active_position.entry,
                                              self.active_position.qty, self.active_position.qty)

        # Force stop out if we hit disaster stop
        if current_prem <= self.active_position.stop:
            action = "DISASTER_STOP"

        if action in ["FULL_EXIT", "STOP_LOSS", "DISASTER_STOP"]:
            logging.info(f"TRAILING STOP TRIGGERED ({action}). Current price: {current_prem}. Closing position.")
            pnl = self.broker.close(self.active_position, current_prem)
            logging.info(f"Position Closed. Realized PnL: Rs {pnl:.2f}. New Capital: Rs {self.broker.capital:.2f}")
            self.active_position = None

            # Close websocket gracefully
            if self.ticker:
                self.ticker.close()


def main():
    print("Initializing AutoBot Live Paper Trading Harness...")
    kite_adapter = setup_kite()

    trader = LivePaperTrader(kite_adapter)
    plan = trader.run_premarket_analysis()

    if plan:
        spot = trader.wait_for_market_open()
        trader.execute_paper_trade(plan, spot)
        if trader.active_position:
            trader.start_websocket()
    else:
        print("No trade planned for today. Exiting.")


if __name__ == "__main__":
    main()
