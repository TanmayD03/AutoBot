import os
import sys
import time
import threading
import logging
from datetime import datetime, date, time as dtime, timedelta
from dotenv import load_dotenv

import pandas as pd
from autobot.data.kite_adapter import KiteAdapter
from autobot.backtest.backtester import load_history, build_signals, days_to_nifty_expiry
from autobot.strategy.decision import DecisionEngine
from autobot.strategy.regime import RegimeDetector
from autobot.strategy.capital import CapitalManager
from autobot.strategy.trail import TrailManager
from autobot.strategy.risk import RiskManager
from autobot.sentiment.ensemble import SentimentEnsemble
from autobot.options_math.black_scholes import implied_vol, greeks, max_pain, put_call_ratio
from autobot.signals.engine import pcr_signal, max_pain_signal, oi_walls_signal, iv_skew_signal
from autobot.execution.paper_broker import PaperBroker

from kiteconnect import KiteTicker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def get_next_tuesday(today: date) -> date:
    """NIFTY weekly options typically expire on Tuesday."""
    days_ahead = 1 - today.weekday()  # 1 = Tuesday
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

        # Risk manager: daily kill switch, profit lock, R:R gate, EOD squareoff time.
        # This existed in the codebase (used by the backtester) but was never
        # instantiated here, so none of its protections were active live.
        self.risk_manager = RiskManager(
            daily_profit_target=self.broker.capital * 0.02,   # lock in day at +2%
            daily_max_loss=self.broker.capital * 0.01,        # kill switch at -1%
            reward_risk_min=2.5,
            max_open_positions=1,
            squareoff=dtime(15, 15),
        )
        self.risk_manager.new_day()

        self.sentiment = SentimentEnsemble()
        self.current_regime = "TRENDING"  # updated by run_premarket_analysis each session
        self.premarket_signals = []
        self.sent_score = 0.0

        # Reconnect / shutdown state for the websocket
        self._intentional_close = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._watchdog_thread = None

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
        self.current_regime = regime  # stashed for regime-aware sizing in execute_paper_trade

        logging.info(f"Regime Detected: {regime}")
        for s in signals:
            logging.info(f"  Signal {s.name}: Score {s.score:.2f}, Conf {s.confidence:.2f}")

        # Pull overnight headlines and score them — previously hardcoded to 0.0,
        # which meant the sentiment layer in the architecture was never actually
        # influencing live decisions.
        try:
            headlines = self.sentiment.fetch_headlines()
            sent_score, sent_conf = self.sentiment.score(headlines)
            logging.info(f"Sentiment: score={sent_score:+.2f} conf={sent_conf:.2f} "
                         f"from {len(headlines)} headlines")
        except Exception as e:
            logging.warning(f"Sentiment fetch failed ({e}); proceeding neutral.")
            sent_score = 0.0

        # The actual 'spot' price isn't fully known until 09:15 open, but we use prev close to build the plan direction
        plan = self.decision_engine.decide(signals, prev.Close, sentiment_impact=sent_score)

        # Stashed for post-open refinement in execute_paper_trade — PCR/max-pain/
        # OI-walls/IV-skew need a real live spot and can only be computed after
        # market open, so this pre-market plan is a coarse bias, not the final say.
        self.premarket_signals = signals
        self.sent_score = sent_score

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

    def refine_plan_with_option_chain(self, spot, expiry_date):
        """
        Refines the pre-market plan using a LIVE option-chain snapshot built
        entirely from Zerodha Kite's own quote() API (see
        KiteAdapter.get_option_chain) — no NSE website scraping. PCR,
        max-pain, OI walls, and ATM IV skew all need a real spot price, which
        only exists after market open, so this step happens here rather than
        in run_premarket_analysis.
        """
        extra_signals = list(self.premarket_signals)
        try:
            chain = self.adapter.get_option_chain(expiry_date, spot)
            if chain:
                pcr = put_call_ratio(chain)
                pain = max_pain(chain)
                extra_signals.append(pcr_signal(pcr))
                extra_signals.append(max_pain_signal(spot, pain))
                extra_signals.append(oi_walls_signal(chain, spot))

                atm_strike = round(spot / 50) * 50
                atm_row = min(chain, key=lambda c: abs(c["strike"] - atm_strike))
                t = days_to_nifty_expiry(datetime.now())
                if atm_row["call_ltp"] > 0 and atm_row["put_ltp"] > 0 and t > 0:
                    iv_ce = implied_vol(atm_row["call_ltp"], spot, atm_row["strike"], r=0.068, t=t, kind="C")
                    iv_pe = implied_vol(atm_row["put_ltp"], spot, atm_row["strike"], r=0.068, t=t, kind="P")
                    extra_signals.append(iv_skew_signal(iv_pe, iv_ce))

                logging.info(f"Live Kite option chain: PCR={pcr:.2f} MaxPain={pain:.0f} "
                            f"({len(chain)} strikes from {expiry_date})")
            else:
                logging.warning("Live option chain came back empty; proceeding on macro+sentiment signals only.")
        except Exception as e:
            logging.warning(f"Option chain fetch/merge failed ({e}); proceeding on macro+sentiment signals only.")

        refined = self.decision_engine.decide(extra_signals, spot, sentiment_impact=self.sent_score)
        logging.info(f"REFINED PLAN (post-open, live option chain): {refined.action} "
                    f"(confidence {refined.confidence:.2f}) — {refined.reason}")
        return refined

    def execute_paper_trade(self, plan, spot):
        """Map the plan to a specific option, size it, and buy via PaperBroker."""
        today_date = datetime.now().date()
        expiry_date = get_next_tuesday(today_date)

        # Refine using a live spot + live Kite option chain — the pre-market
        # plan was only a coarse macro/sentiment bias since PCR/max-pain/OI
        # walls/IV skew can't be known before 09:15.
        plan = self.refine_plan_with_option_chain(spot, expiry_date)
        if plan is None or plan.action == "NO_TRADE":
            logging.warning(f"Post-open refinement is NO_TRADE ({plan.reason if plan else 'n/a'}). Skipping trade.")
            return

        atm = round(spot / 50) * 50
        kind = "CE" if plan.action == "BUY_CE" else "PE"

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

        # Risk gate: refuses to trade if kill switch/profit-lock is active,
        # a position is already open, or we're past the squareoff window.
        if not self.risk_manager.can_trade(open_positions=0, now_time=datetime.now().time()):
            logging.warning("RiskManager blocked entry (halted, max positions, or past squareoff). Skipping trade.")
            return

        # Delta-based sizing: back out implied vol from the observed premium,
        # then get delta from Black-Scholes greeks, then size with the same
        # calculate_lots_by_delta() the backtester uses (now with a real
        # affordability check — see capital.py fix).
        t = days_to_nifty_expiry(datetime.now())
        opt_kind = "C" if kind == "CE" else "P"
        try:
            iv = implied_vol(entry_prem, spot, atm, r=0.068, t=t, kind=opt_kind)
            delta = greeks(spot, atm, r=0.068, sigma=iv, t=t, kind=opt_kind).delta
        except Exception as e:
            logging.warning(f"IV/delta solve failed ({e}); falling back to ATM delta=0.5")
            delta = 0.5 if opt_kind == "C" else -0.5

        risk_pct = self.capital_manager.regime_confidence_risk_pct(self.current_regime, plan.confidence)
        lots = self.capital_manager.calculate_lots_by_delta(
            self.broker.capital, entry_prem, delta,
            lot_size=self.option_meta["lot_size"], risk_pct_override=risk_pct)

        if lots <= 0:
            logging.warning(f"Position sizing returned 0 lots (1 lot of {tsym} not affordable "
                            f"at Rs {entry_prem} within risk cap). Skipping trade.")
            return

        qty = lots * self.option_meta["lot_size"]

        disaster = max(1.0, entry_prem * 0.5) # Hardcoded disaster stop at 50% for live harness
        self.active_position = self.broker.buy(tsym, qty, entry_prem, stop=disaster, target=entry_prem*3)
        self.peak_opt_price = entry_prem

        # Scale the daily kill-switch loss threshold to this position's actual size
        self.risk_manager.set_dynamic_daily_loss(entry_prem, qty)

        # Initialize TrailManager
        is_expiry_day = (today_date == expiry_date)
        self.trail_manager = TrailManager(expiry_mode=is_expiry_day)

        logging.info(f"EXECUTED PAPER TRADE: Bought {qty} ({lots} lots, delta={delta:.2f}, "
                    f"regime={self.current_regime}, risk_pct={risk_pct*100:.1f}%) "
                    f"of {tsym} at {entry_prem}")

    def _start_watchdog(self):
        """Runs independently of ticks so EOD squareoff fires even if the feed
        goes quiet (e.g. low liquidity option with no trades for a while)."""
        def loop():
            while not self._intentional_close:
                now_t = datetime.now().time()
                if self.active_position and now_t >= self.risk_manager.squareoff:
                    price = self.latest_opt_price or self.active_position.entry
                    logging.warning("Watchdog: squareoff time reached with no recent tick trigger — forcing close.")
                    self.evaluate_trail(price)
                if now_t >= dtime(15, 35):  # safety cutoff well past close either way
                    self._intentional_close = True
                    if self.ticker:
                        self.ticker.close()
                    break
                time.sleep(15)
        self._watchdog_thread = threading.Thread(target=loop, daemon=True)
        self._watchdog_thread.start()

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
            self._reconnect_attempts = 0
            ws.subscribe([self.spot_token, self.option_token])
            ws.set_mode(ws.MODE_LTP, [self.spot_token, self.option_token])
            if not self._watchdog_thread:
                self._start_watchdog()

        def on_close(ws, code, reason):
            logging.info(f"WebSocket closed (code={code}, reason={reason}).")
            if self._intentional_close:
                return  # we closed it ourselves (position exited) — do not reconnect

            if self._reconnect_attempts >= self._max_reconnect_attempts:
                logging.error("Max reconnect attempts hit. Position may be UNMONITORED — "
                              "check Kite app / broker terminal manually right now.")
                return

            self._reconnect_attempts += 1
            backoff = min(30, 2 ** self._reconnect_attempts)
            logging.warning(f"Unexpected disconnect with an open position. "
                            f"Reconnecting in {backoff}s (attempt {self._reconnect_attempts}/{self._max_reconnect_attempts})...")
            time.sleep(backoff)
            try:
                self.ticker.connect(threaded=False)
            except Exception as e:
                logging.error(f"Reconnect attempt failed: {e}")

        def on_error(ws, code, reason):
            logging.error(f"WebSocket error: code={code} reason={reason}")

        self.ticker.on_ticks = on_ticks
        self.ticker.on_connect = on_connect
        self.ticker.on_close = on_close
        self.ticker.on_error = on_error

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

        # EOD forced square-off: RiskManager.can_trade() only blocks NEW entries
        # past squareoff time — it never closed an already-open position. Without
        # this, a position that never hits stop/target just sits open indefinitely.
        now_t = datetime.now().time()
        if now_t >= self.risk_manager.squareoff:
            action = "EOD_SQUAREOFF"

        # Scale-out: TrailManager already computes this (40% gain, 20% on expiry
        # day) — it just wasn't being acted on here. Book half, move the stop to
        # breakeven on the remainder so the rest of the trade can run risk-free.
        if action == "PARTIAL_EXIT" and self.active_position:
            exit_qty = self.active_position.qty // 2
            if exit_qty > 0:
                pnl = self.broker.close_partial(self.active_position, exit_qty, current_prem)
                self.risk_manager.register_pnl(pnl)
                self.active_position.stop = self.active_position.entry  # breakeven on the runner
                logging.info(f"PARTIAL PROFIT BOOKED: {exit_qty} qty @ {current_prem}. "
                            f"Realized: Rs {pnl:.2f}. Remaining qty {self.active_position.qty} "
                            f"now stopped at breakeven ({self.active_position.stop:.2f}).")
            return  # position still open — do not fall through to full-close logic

        if action in ["FULL_EXIT", "STOP_LOSS", "DISASTER_STOP", "EOD_SQUAREOFF"]:
            logging.info(f"TRAILING STOP TRIGGERED ({action}). Current price: {current_prem}. Closing position.")
            pnl = self.broker.close(self.active_position, current_prem)
            logging.info(f"Position Closed. Realized PnL: Rs {pnl:.2f}. New Capital: Rs {self.broker.capital:.2f}")
            self.active_position = None

            # Feed the realized P&L into the daily kill switch / profit lock
            self.risk_manager.register_pnl(pnl)
            if self.risk_manager.halted:
                logging.info(f"RiskManager HALTED for the day. Day PnL: Rs {self.risk_manager.day_pnl:.2f}")

            # Close websocket gracefully — this IS an intentional close, so
            # the reconnect handler below should not try to reconnect.
            self._intentional_close = True
            if self.ticker:
                self.ticker.close()


def main():
    print("Initializing AutoBot Live Paper Trading Harness...")
    kite_adapter = setup_kite()

    trader = LivePaperTrader(kite_adapter)
    trader.risk_manager.new_day()  # reset kill switch / profit lock / day_pnl for this session
    plan = trader.run_premarket_analysis()

    if plan:
        spot = trader.wait_for_market_open()
        trader.execute_paper_trade(plan, spot)
        if trader.active_position:
            try:
                trader.start_websocket()
            except KeyboardInterrupt:
                logging.warning("Ctrl+C received — closing open position at last known price before exit.")
                trader._intentional_close = True
                if trader.active_position and trader.latest_opt_price:
                    pnl = trader.broker.close(trader.active_position, trader.latest_opt_price)
                    trader.risk_manager.register_pnl(pnl)
                    logging.info(f"Manual close PnL: Rs {pnl:.2f}")
                if trader.ticker:
                    trader.ticker.close()
    else:
        print("No trade planned for today. Exiting.")


if __name__ == "__main__":
    main()
