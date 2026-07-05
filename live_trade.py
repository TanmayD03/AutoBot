import os
import sys
import time
import threading
import logging
from datetime import datetime, date, time as dtime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

IST = ZoneInfo("Asia/Kolkata")


def now_ist() -> datetime:
    """All trading-hours/squareoff comparisons use this, not naive
    datetime.now() — if the machine running this isn't set to IST, every
    time-of-day check in this file would silently be wrong."""
    return datetime.now(IST)


import pandas as pd
import yaml
from autobot.data.kite_adapter import KiteAdapter
from autobot.backtest.backtester import load_history, build_signals, days_to_nifty_expiry
from autobot.strategy.decision import DecisionEngine
from autobot.strategy.regime import RegimeDetector
from autobot.strategy.capital import CapitalManager
from autobot.strategy.trail import TrailManager
from autobot.strategy.risk import RiskManager
from autobot.strategy.intraday import ChoppyMarketStrategy, SpreadStrategy
from autobot.strategy.expiry import ExpiryDayStrategy
from autobot.nature.immune import ImmuneSystem
from autobot.sentiment.ensemble import SentimentEnsemble
from autobot.options_math.black_scholes import implied_vol, greeks, max_pain, put_call_ratio, get_option_walls
from autobot.signals.engine import (pcr_signal, max_pain_signal, oi_walls_signal, iv_skew_signal,
                                    fii_flow_signal, SignalScore, _clip)
from autobot.data.market_feed import MarketDataFeed, MarketPreFlightMatrix, GiftNiftyTracker
from autobot.execution.paper_broker import PaperBroker
from autobot.terminal.dashboard import Dashboard

from kiteconnect import KiteTicker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def get_next_tuesday(today: date) -> date:
    """NIFTY weekly options typically expire on Tuesday."""
    days_ahead = 1 - today.weekday()  # 1 = Tuesday
    if days_ahead < 0:  # Target next week
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

    print("\n" + "=" * 50)
    print("ZERODHA LOGIN REQUIRED")
    print("1. Go to this URL in your browser:")
    print(f"   {kite_adapter.get_login_url()}")
    print("2. Login and you will be redirected.")
    print("3. Copy the 'request_token' parameter from the redirected URL.")
    print("=" * 50 + "\n")

    req_token = input("Enter request_token: ").strip()
    kite_adapter.set_access_token(req_token)
    kite_adapter.fetch_master_instruments()
    return kite_adapter


class LivePaperTrader:
    def __init__(self, kite_adapter: KiteAdapter):
        self.kite = kite_adapter.kite
        self.adapter = kite_adapter
        self.broker = PaperBroker(capital=100000.0)  # Rs 1 Lakh paper capital

        # Anomaly halt (Artificial Immune System): flags flash-crash-like
        # statistical outliers in spot returns / VIX and refuses new entries
        # when tripped. Existed in the codebase, was never instantiated before.
        self.immune = ImmuneSystem(z_threshold=4.0, window=300)
        self.decision_engine = DecisionEngine(immune=self.immune)

        self.regime_detector = RegimeDetector()
        self.capital_manager = CapitalManager()

        # All four strategies, now actually instantiated and dispatched to
        # based on regime / expiry-day (see execute_paper_trade router below).
        self.choppy_strategy = ChoppyMarketStrategy()
        self.spread_strategy = SpreadStrategy()
        self.expiry_strategy = ExpiryDayStrategy()

        self.trail_manager = None
        self.active_position = None          # single-leg naked Position (TRENDING regime)
        self.active_multi_position = None    # MultiLegPosition (spread/condor)
        self.option_meta = None
        self.spot_token = 256265  # Hardcoded standard token for NIFTY 50 index (NSE:NIFTY 50)
        self.option_token = None
        self.leg_tokens = {}          # token -> "NFO:tradingsymbol", for multi-leg tracking
        self.latest_leg_prices = {}   # "NFO:tradingsymbol" -> last price
        self.ticker = None

        # Load config.yaml for real — previously ONLY the backtester read this
        # file; live_trade.py had its own hardcoded values entirely
        # disconnected from it, including CapitalManager's default 50%
        # max_risk_per_trade_pct (vs config.yaml's 1.5-2.0% per-regime values).
        # Editing config.yaml used to have zero effect on live trading.
        try:
            with open("config.yaml", "r") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logging.warning(f"config.yaml not found/unreadable ({e}); using built-in defaults.")
            self.config = {}

        risk_cfg = self.config.get("risk", {})
        squareoff_str = risk_cfg.get("squareoff_time", "15:15")
        squareoff_h, squareoff_m = (int(x) for x in squareoff_str.split(":"))

        # Risk manager: daily kill switch, profit lock, R:R gate, EOD squareoff time.
        self.risk_manager = RiskManager(
            daily_profit_target=self.broker.capital * risk_cfg.get("daily_profit_target_pct", 2.5) / 100.0,
            daily_max_loss=self.broker.capital * risk_cfg.get("daily_max_loss_pct", 3.0) / 100.0,
            reward_risk_min=risk_cfg.get("reward_risk_min", 2.5),
            max_open_positions=risk_cfg.get("max_open_positions", 1),
            squareoff=dtime(squareoff_h, squareoff_m),
        )
        self.risk_manager.new_day()

        self.sentiment = SentimentEnsemble()
        self.current_regime = "TRENDING"  # updated by run_premarket_analysis each session
        self.premarket_signals = []
        self.sent_score = 0.0
        self.today_row = None  # today's daily indicator row (vix/dma/rsi/adx), stashed for reuse

        # FII/DII (NSE, no Kite equivalent) + US ADR (direct Yahoo JSON, lower
        # latency than the yfinance library) + GIFT Nifty (tvDatafeed primary,
        # NSE marketStatus fallback). Explicit exception to the Kite-only rule
        # per instruction — nothing else in this system scrapes anything.
        self.raw_feed = MarketDataFeed()
        self.matrix_processor = MarketPreFlightMatrix(self.raw_feed)
        self.gift_tracker = GiftNiftyTracker()
        self.overnight_risk_multiplier = 1.0   # halved on a bearish overnight FII/ADR matrix
        self.allow_flipper = False              # True once GIFT Nifty is confirmed reachable

        # Live monitoring dashboard (rich terminal panels). Runs as a periodic
        # full-state snapshot from a background thread rather than a
        # persistent redrawing pane — a Live-updating pane fighting with
        # normal logging.info() output from another thread is a real rich
        # limitation, and a snapshot every N seconds is more robust for a
        # session where correctness matters more than polish.
        self.dashboard = Dashboard()
        self.trade_log = []           # human-readable entry/exit lines, newest last
        self.last_headlines = []
        self.last_pcr = None
        self.last_max_pain = None
        self.last_call_wall = None
        self.last_put_wall = None
        self._dashboard_thread = None
        self.last_tick_time = time.time()

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
        df = load_history(2)  # Load 2 years — extra headroom on top of the min_periods fix below

        if len(df) < 4:
            logging.error("Not enough historical data to generate signals.")
            sys.exit(1)

        today = df.iloc[-1]
        prev = df.iloc[-2]
        prev2_close = df.iloc[-3].Close
        prev3_close = df.iloc[-4].Close if len(df) >= 4 else None
        self.today_row = today  # reused later for condor VIX check / expiry fallback

        # Seed the anomaly detector with recent daily history so it isn't
        # blind on day 1 — otherwise it needs ~30 live observations before
        # it can ever flag anything (see ImmuneSystem.is_anomalous).
        recent = df.tail(60)
        daily_ret = recent["Close"].pct_change() * 100
        for r, v in zip(daily_ret.dropna().tolist(), recent["vix"].iloc[1:].tolist()):
            self.immune.observe(ret_pct=r, vix=v)

        signals = build_signals(prev, today, prev2_close, prev3_close)
        adx_val = getattr(today, "adx", 18.0)
        regime = self.regime_detector.classify(vix=today.vix, adx=adx_val, atr_pct=(today.atr / prev.Close) * 100)
        self.current_regime = regime  # stashed for regime-aware sizing/strategy selection

        logging.info(f"Regime Detected: {regime}")

        # --- Live FII/DII + ADR pre-flight matrix (NSE FII/DII, direct Yahoo ADR JSON) ---
        # Replaces the historical ADR-based proxy build_signals() uses for both
        # "fii_flow" and "adr" — that proxy exists only because real FII/DII
        # history isn't available for years of backtesting. Live mode has the
        # real thing, so use it.
        try:
            morning_state = self.matrix_processor.generate_morning_matrix()
            logging.info(f"[PRE-FLIGHT] Regime: {morning_state['regime']} (bias {morning_state['bias_score']}), "
                        f"FII net Rs {morning_state['fii_net_cr']} Cr, DII net Rs {morning_state['dii_net_cr']} Cr, "
                        f"ADR bias {morning_state['adr_bias_pct']:+.2f}%")

            real_fii_signal = fii_flow_signal(morning_state['fii_net_cr'], morning_state['dii_net_cr'])
            signals = [s for s in signals if s.name != "fii_flow"] + [real_fii_signal]

            real_adr_signal = SignalScore("adr", _clip(morning_state['adr_bias_pct'] / 1.5), 0.65)
            signals = [s for s in signals if s.name != "adr"] + [real_adr_signal]

            # From the reference design: halve size on a bearish overnight
            # matrix rather than blocking trades outright — this multiplies
            # into risk_pct at execution time in each _execute_* method.
            self.overnight_risk_multiplier = 0.5 if morning_state['regime'] == "BEARISH" else 1.0
            if morning_state['regime'] == "BEARISH":
                logging.info("[RISK] Bearish overnight FII/DII+ADR bias — position sizing halved today.")
        except Exception as e:
            logging.warning(f"Pre-flight FII/DII/ADR matrix failed ({e}); using historical proxy signals instead.")
            self.overnight_risk_multiplier = 1.0

        # --- GIFT Nifty: tvDatafeed primary, NSE marketStatus fallback ---
        gift_price = None
        gift_snapshot = self.gift_tracker.fetch_live_snapshot()
        if "error" not in gift_snapshot:
            gift_price = gift_snapshot["live_price"]
            logging.info(f"[GUARD] GIFT Nifty (NSEIX NIFTY1! via tvDatafeed): {gift_price} "
                        f"@ {gift_snapshot['timestamp']}")
        else:
            logging.warning(f"tvDatafeed GIFT Nifty failed ({gift_snapshot['error']}); trying NSE fallback...")
            nse_gift = self.raw_feed.fetch_gift_nifty_live()
            if isinstance(nse_gift, dict) and nse_gift.get("last"):
                gift_price = float(nse_gift["last"])
                logging.info(f"[GUARD] GIFT Nifty (NSE marketStatus fallback): {gift_price}")

        self.allow_flipper = gift_price is not None
        if not self.allow_flipper:
            logging.warning("[GUARD] GIFT Nifty unreachable from both sources this session. "
                            "(IntradayFlipper is not yet wired into execution, so this flag is "
                            "informational only for now — see prior notes on why.)")
        else:
            # Real overnight gap: GIFT Nifty futures vs YESTERDAY's actual NIFTY
            # close (today.Close here, since 'today' is the most recently
            # completed session — today's own bar doesn't exist pre-market).
            # This replaces build_signals()'s gift_gap, which used today.Open —
            # not populated until AFTER market open, so pre-market it was
            # silently reflecting a stale, already-known historical gap.
            real_gap_pct = (gift_price / today.Close - 1) * 100
            real_gift_signal = SignalScore("gift_gap", _clip(real_gap_pct), 0.70) if abs(real_gap_pct) >= 0.10 \
                else SignalScore("gift_gap", 0.0, 0.0)
            signals = [s for s in signals if s.name != "gift_gap"] + [real_gift_signal]
            logging.info(f"[PRE-FLIGHT] Real overnight gap vs GIFT Nifty: {real_gap_pct:+.2f}%")

        for s in signals:
            logging.info(f"  Signal {s.name}: Score {s.score:.2f}, Conf {s.confidence:.2f}")

        # Pull overnight headlines and score them — the one external data
        # source in this system; everything else comes from Zerodha itself.
        try:
            headlines = self.sentiment.fetch_headlines()
            sent_score, sent_conf = self.sentiment.score(headlines)
            self.last_headlines = headlines  # cached for the dashboard's sentiment panel
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
            logging.warning("Pre-market analysis dictates NO TRADE for today (coarse bias only — "
                            "regime/expiry-day strategies may still trade after open).")
            return None

        logging.info(f"PRE-MARKET PLAN: {plan.action} (Confidence: {plan.confidence:.2f})")
        return plan

    def wait_for_market_open(self):
        """
        Waits for actual NSE trading hours (9:15-15:30 IST, Mon-Fri) before
        treating a quote as live. The old check was just "is LTP > 0", which
        is trivially true at ANY time of day or night — Kite still returns
        the last traded price from the previous session even when the market
        is closed.
        """
        now = now_ist()
        if now.weekday() >= 5:
            logging.error(f"Today ({now:%A}) is a weekend — NSE is closed. Exiting rather than "
                          f"proceeding on stale data.")
            sys.exit(0)
        if not (dtime(9, 15) <= now.time() <= dtime(15, 30)):
            logging.error(f"Current IST time is {now:%H:%M:%S}, outside NSE trading hours "
                          f"(9:15-15:30 IST, Mon-Fri). Exiting rather than proceeding on stale data.")
            sys.exit(0)

        logging.info("Waiting for market open to capture spot price...")
        while True:
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
        only exists after market open.
        """
        extra_signals = list(self.premarket_signals)
        try:
            chain = self.adapter.get_option_chain(expiry_date, spot)
            if chain:
                pcr = put_call_ratio(chain)
                pain = max_pain(chain)
                call_wall, put_wall = get_option_walls(chain)
                self.last_pcr, self.last_max_pain = pcr, pain
                self.last_call_wall, self.last_put_wall = call_wall, put_wall
                extra_signals.append(pcr_signal(pcr))
                extra_signals.append(max_pain_signal(spot, pain))
                extra_signals.append(oi_walls_signal(chain, spot))

                atm_strike = round(spot / 50) * 50
                atm_row = min(chain, key=lambda c: abs(c["strike"] - atm_strike))
                t = days_to_nifty_expiry(now_ist())
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

        # Feed the anomaly detector today's opening gap so a genuinely
        # freakish open (flash-crash-like) can gate the trade.
        anomaly_inputs = None
        if self.today_row is not None:
            gap_ret = (spot / self.today_row.Close - 1) * 100
            anomaly_inputs = {"ret_pct": gap_ret, "vix": self.today_row.vix}

        refined = self.decision_engine.decide(extra_signals, spot, sentiment_impact=self.sent_score,
                                              anomaly_inputs=anomaly_inputs)
        logging.info(f"REFINED PLAN (post-open, live option chain): {refined.action} "
                    f"(confidence {refined.confidence:.2f}) — {refined.reason}")
        return refined

    def _regime_config(self, regime=None):
        """Returns config.yaml's per-regime block (risk_per_trade_pct,
        theta_timeout_minutes, etc.) — falls back to an empty dict (which
        makes callers fall back to their own hardcoded defaults) if
        config.yaml is missing or the regime key isn't present."""
        regime = regime or self.current_regime
        key = regime.lower()
        return self.config.get("regimes", {}).get(key, {})

    def _get_live_vix(self):
        """Live India VIX quote — used on intraday re-checks so the condor
        viability test reflects current volatility, not the premarket value
        frozen at market open."""
        try:
            q = self.kite.quote("NSE:INDIA VIX")
            return q["NSE:INDIA VIX"]["last_price"]
        except Exception as e:
            logging.warning(f"Live VIX quote failed ({e}); using premarket VIX as fallback.")
            return self.today_row.vix if self.today_row is not None else 15.0

    def _resolve_leg(self, strike, kind, expiry_date):
        """Resolve + quote one option leg via Zerodha. Returns a dict or None."""
        try:
            meta = self.adapter.find_nifty_option(strike, kind, expiry_date)
        except Exception as e:
            logging.warning(f"Leg resolution failed for {strike}{kind}: {e}")
            return None
        tsym = "NFO:" + meta["tradingsymbol"]
        try:
            q = self.kite.quote(tsym)
            ltp = q[tsym]["last_price"]
        except Exception as e:
            logging.warning(f"Leg quote failed for {tsym}: {e}")
            return None
        return {"tradingsymbol": meta["tradingsymbol"], "tsym": tsym, "token": meta["instrument_token"],
                "lot_size": meta["lot_size"], "ltp": ltp}

    def execute_paper_trade(self, spot):
        """
        Strategy router: picks ExpiryDayStrategy / iron condor / defined-risk
        spread / naked CE-PE depending on expiry-day status and the detected
        regime, then delegates to the matching _execute_* method. A failure
        or non-viable setup in one path falls back to the next rather than
        just giving up for the day.

        Fully re-derives its decision from the CURRENT spot and a fresh
        Kite option-chain snapshot every time it's called — nothing here is
        cached from market open — which is what makes it safe to call this
        repeatedly throughout the day (see the retry loop in main()) rather
        than only once at 9:15.
        """
        today_date = now_ist().date()
        expiry_date = get_next_tuesday(today_date)
        is_expiry_day = (today_date == expiry_date)

        if not self.risk_manager.can_trade(open_positions=0, now_time=now_ist().time()):
            logging.warning("RiskManager blocked entry (halted, max positions, or past squareoff). Skipping trade.")
            return

        if is_expiry_day and self.expiry_strategy.is_valid_window(now_ist().time()):
            self._execute_expiry(spot, expiry_date)
            return

        refined_plan = self.refine_plan_with_option_chain(spot, expiry_date)
        if refined_plan is None or refined_plan.action == "NO_TRADE":
            logging.warning(f"No qualifying setup right now ({refined_plan.reason if refined_plan else 'n/a'}). "
                            f"Will check again next cycle rather than giving up for the day.")
            return

        if self.current_regime == "CHOPPY":
            executed = self._execute_condor(spot, expiry_date)
            if not executed:
                logging.info("Condor not viable (VIX too high, unresolved legs, or unaffordable) — "
                            "falling back to a defined-risk spread instead of sitting out.")
                self._execute_spread(spot, refined_plan, expiry_date)
        elif self.current_regime == "HIGH_VOLATILITY":
            self._execute_spread(spot, refined_plan, expiry_date)
        else:  # TRENDING
            self._execute_naked(spot, refined_plan, expiry_date, today_date)

    def _execute_naked(self, spot, plan, expiry_date, today_date):
        """TRENDING regime: single-leg naked CE/PE, sized by delta."""
        atm = round(spot / 50) * 50
        kind = "CE" if plan.action == "BUY_CE" else "PE"

        leg = self._resolve_leg(atm, kind, expiry_date)
        if not leg:
            logging.error("Failed to resolve naked leg — skipping trade.")
            return

        tsym, entry_prem = leg["tsym"], leg["ltp"]
        logging.info(f"[TRENDING/NAKED] Current Premium for {tsym}: Rs {entry_prem}")
        if entry_prem <= 0:
            logging.error("Invalid premium.")
            return

        t = days_to_nifty_expiry(now_ist())
        opt_kind = "C" if kind == "CE" else "P"
        try:
            iv = implied_vol(entry_prem, spot, atm, r=0.068, t=t, kind=opt_kind)
            delta = greeks(spot, atm, r=0.068, sigma=iv, t=t, kind=opt_kind).delta
        except Exception as e:
            logging.warning(f"IV/delta solve failed ({e}); falling back to ATM delta=0.5")
            delta = 0.5 if opt_kind == "C" else -0.5

        base_pct = self._regime_config().get("risk_per_trade_pct", 2.0) / 100.0
        risk_pct = self.capital_manager.regime_confidence_risk_pct(self.current_regime, plan.confidence, base_pct=base_pct) \
            * self.overnight_risk_multiplier
        lots = self.capital_manager.calculate_lots_by_delta(
            self.broker.capital, entry_prem, delta,
            lot_size=leg["lot_size"], risk_pct_override=risk_pct)

        if lots <= 0:
            logging.warning(f"Position sizing returned 0 lots for {tsym}. Skipping trade.")
            return

        qty = lots * leg["lot_size"]
        disaster = max(1.0, entry_prem * 0.5)
        self.active_position = self.broker.buy(tsym, qty, entry_prem, stop=disaster, target=entry_prem * 3)
        if self.active_position:
            self.active_position.entry_time = now_ist()
        if not self.active_position:
            logging.warning("Broker rejected naked entry (capital circuit breaker).")
            return

        self.option_meta = {"tradingsymbol": leg["tradingsymbol"], "instrument_token": leg["token"],
                            "lot_size": leg["lot_size"]}
        self.option_token = leg["token"]
        self.peak_opt_price = entry_prem
        self.risk_manager.set_dynamic_daily_loss(entry_prem, qty)
        self.trail_manager = TrailManager(expiry_mode=(today_date == expiry_date))
        self.leg_tokens = {}

        logging.info(f"EXECUTED NAKED TRADE: Bought {qty} ({lots} lots, delta={delta:.2f}, "
                    f"regime={self.current_regime}, risk_pct={risk_pct*100:.1f}%) of {tsym} at {entry_prem}")
        self.trade_log.append(f"{now_ist():%H:%M:%S} ENTRY NAKED {tsym} qty={qty} @ Rs{entry_prem:.1f}")

    def _execute_condor(self, spot, expiry_date) -> bool:
        """
        CHOPPY regime: sell an iron condor, sized off the live Kite chain +
        VIX. Returns True if placed, False if it backed off — the caller
        then falls back to a defined-risk spread rather than sitting idle.
        """
        vix_today = self._get_live_vix()
        classification = self.choppy_strategy.evaluate_iron_condor(vix_today)
        if classification == "NO_TRADE":
            logging.info(f"Condor skipped: VIX {vix_today:.1f} too high for a defined-risk condor.")
            return False

        atm = round(spot / 50) * 50
        short_wing = 50 if classification == "SELL_NARROW_CONDOR" else 100
        long_wing = 50

        ce_short = self._resolve_leg(atm + short_wing, "CE", expiry_date)
        ce_long = self._resolve_leg(atm + short_wing + long_wing, "CE", expiry_date)
        pe_short = self._resolve_leg(atm - short_wing, "PE", expiry_date)
        pe_long = self._resolve_leg(atm - short_wing - long_wing, "PE", expiry_date)

        if not all([ce_short, ce_long, pe_short, pe_long]):
            logging.warning("Condor leg resolution incomplete — skipping condor.")
            return False

        net_credit = (ce_short["ltp"] + pe_short["ltp"]) - (ce_long["ltp"] + pe_long["ltp"])
        max_loss_per_unit = long_wing - net_credit
        if max_loss_per_unit <= 0:
            logging.warning("Condor pricing looks inverted (max_loss<=0) — skipping.")
            return False

        lot_size = ce_short["lot_size"]
        base_pct = self._regime_config("CHOPPY").get("risk_per_trade_pct", 1.5) / 100.0
        risk_pct = self.capital_manager.regime_confidence_risk_pct("CHOPPY", 0.70, base_pct=base_pct) * self.overnight_risk_multiplier
        lots = self.capital_manager.calculate_lots_by_max_loss(
            self.broker.capital, max_loss_per_unit, lot_size=lot_size, risk_pct_override=risk_pct)
        if lots <= 0:
            logging.warning("Condor not affordable within risk cap — skipping.")
            return False

        legs = [
            {"symbol": ce_short["tsym"], "side": "SELL", "price": ce_short["ltp"]},
            {"symbol": ce_long["tsym"], "side": "BUY", "price": ce_long["ltp"]},
            {"symbol": pe_short["tsym"], "side": "SELL", "price": pe_short["ltp"]},
            {"symbol": pe_long["tsym"], "side": "BUY", "price": pe_long["ltp"]},
        ]
        pos = self.broker.buy_multi(legs, max_loss_per_unit, lots * lot_size, kind="CREDIT_CONDOR")
        if pos:
            pos.entry_time = now_ist()
        if not pos:
            logging.warning("Broker rejected condor entry (capital circuit breaker).")
            return False

        self.active_multi_position = pos
        self.leg_tokens = {ce_short["token"]: ce_short["tsym"], ce_long["token"]: ce_long["tsym"],
                            pe_short["token"]: pe_short["tsym"], pe_long["token"]: pe_long["tsym"]}
        self.latest_leg_prices = {ce_short["tsym"]: ce_short["ltp"], ce_long["tsym"]: ce_long["ltp"],
                                   pe_short["tsym"]: pe_short["ltp"], pe_long["tsym"]: pe_long["ltp"]}
        self.risk_manager.set_dynamic_daily_loss(max_loss_per_unit, lots * lot_size)
        logging.info(f"EXECUTED IRON CONDOR ({classification}): {lots} lots, net credit {net_credit:.2f}, "
                    f"max loss/unit {max_loss_per_unit:.2f}. SELL {ce_short['tradingsymbol']}, "
                    f"BUY {ce_long['tradingsymbol']}, SELL {pe_short['tradingsymbol']}, "
                    f"BUY {pe_long['tradingsymbol']}.")
        self.trade_log.append(f"{now_ist():%H:%M:%S} ENTRY CONDOR ({classification}) "
                              f"lots={lots} net_credit={net_credit:.1f}")
        return True

    def _execute_spread(self, spot, plan, expiry_date):
        """HIGH_VOLATILITY regime (or condor fallback): defined-risk debit spread."""
        atm = round(spot / 50) * 50
        bullish = plan.action == "BUY_CE" if plan else spot >= atm
        wing = 100

        spec = self.spread_strategy.bull_call_spread(spot, atm, wing=wing) if bullish \
            else self.spread_strategy.bear_put_spread(spot, atm, wing=wing)
        leg_kind = spec["kind"] + "E"  # "C"/"P" -> "CE"/"PE"

        buy_leg = self._resolve_leg(spec["buy"], leg_kind, expiry_date)
        sell_leg = self._resolve_leg(spec["sell"], leg_kind, expiry_date)
        if not buy_leg or not sell_leg:
            logging.warning("Spread leg resolution failed — skipping trade.")
            return

        net_debit = buy_leg["ltp"] - sell_leg["ltp"]
        if net_debit <= 0:
            logging.warning("Spread pricing looks inverted (net_debit<=0) — skipping.")
            return
        max_loss_per_unit = net_debit  # a debit spread's max loss IS the debit paid

        lot_size = buy_leg["lot_size"]
        base_pct = self._regime_config().get("risk_per_trade_pct", 2.0) / 100.0
        risk_pct = self.capital_manager.regime_confidence_risk_pct(
            self.current_regime, plan.confidence if plan else 0.70, base_pct=base_pct) * self.overnight_risk_multiplier
        lots = self.capital_manager.calculate_lots_by_max_loss(
            self.broker.capital, max_loss_per_unit, lot_size=lot_size, risk_pct_override=risk_pct)
        if lots <= 0:
            logging.warning("Spread not affordable within risk cap — skipping trade.")
            return

        legs = [
            {"symbol": buy_leg["tsym"], "side": "BUY", "price": buy_leg["ltp"]},
            {"symbol": sell_leg["tsym"], "side": "SELL", "price": sell_leg["ltp"]},
        ]
        pos = self.broker.buy_multi(legs, max_loss_per_unit, lots * lot_size, kind="DEBIT_SPREAD")
        if pos:
            pos.entry_time = now_ist()
        if not pos:
            logging.warning("Broker rejected spread entry (capital circuit breaker).")
            return

        self.active_multi_position = pos
        self.leg_tokens = {buy_leg["token"]: buy_leg["tsym"], sell_leg["token"]: sell_leg["tsym"]}
        self.latest_leg_prices = {buy_leg["tsym"]: buy_leg["ltp"], sell_leg["tsym"]: sell_leg["ltp"]}
        self.risk_manager.set_dynamic_daily_loss(max_loss_per_unit, lots * lot_size)
        direction = "BULL CALL" if bullish else "BEAR PUT"
        logging.info(f"EXECUTED {direction} SPREAD: {lots} lots, net debit {net_debit:.2f} (max loss/unit). "
                    f"BUY {buy_leg['tradingsymbol']}, SELL {sell_leg['tradingsymbol']}.")
        self.trade_log.append(f"{now_ist():%H:%M:%S} ENTRY {direction} SPREAD lots={lots} "
                              f"net_debit={net_debit:.1f}")

    def _execute_expiry(self, spot, expiry_date):
        """Expiry-day max-pain reversion, using the live Kite chain (not NSE)."""
        try:
            chain = self.adapter.get_option_chain(expiry_date, spot)
            pain = max_pain(chain) if chain else spot
        except Exception as e:
            logging.warning(f"Expiry chain fetch failed ({e}); using spot as max-pain fallback.")
            pain = spot

        action, strike = self.expiry_strategy.evaluate(now_ist().time(), spot, pain)
        if action == "NO_TRADE":
            logging.info(f"Expiry-day max-pain check: no edge (spot {spot:.0f} vs max pain {pain:.0f}).")
            return

        kind = "CE" if action == "BUY_CE" else "PE"
        leg = self._resolve_leg(strike, kind, expiry_date)
        if not leg:
            logging.error("Failed to resolve expiry-day leg — skipping trade.")
            return

        entry_prem = leg["ltp"]
        if entry_prem <= 0:
            logging.error("Invalid premium on expiry leg.")
            return

        # Expiry-day naked options decay fast and violently — size extra
        # conservatively (half the normal risk cap); TrailManager's tighter
        # expiry_mode (20% partial target vs 40%) handles the exit side.
        base_pct = self.config.get("risk", {}).get("risk_per_trade_pct", 2.0) / 100.0
        lots = self.capital_manager.calculate_lots_by_delta(
            self.broker.capital, entry_prem, delta=0.35 if kind == "CE" else -0.35,
            lot_size=leg["lot_size"],
            risk_pct_override=base_pct * 0.5 * self.overnight_risk_multiplier)
        if lots <= 0:
            logging.warning("Expiry-day trade not affordable within risk cap — skipping.")
            return

        qty = lots * leg["lot_size"]
        tsym = leg["tsym"]
        disaster = max(1.0, entry_prem * 0.5)
        self.active_position = self.broker.buy(tsym, qty, entry_prem, stop=disaster, target=entry_prem * 2)
        if self.active_position:
            self.active_position.entry_time = now_ist()
        if not self.active_position:
            logging.warning("Broker rejected expiry-day entry (capital circuit breaker).")
            return

        self.option_meta = {"tradingsymbol": leg["tradingsymbol"], "instrument_token": leg["token"],
                            "lot_size": leg["lot_size"]}
        self.option_token = leg["token"]
        self.peak_opt_price = entry_prem
        self.risk_manager.set_dynamic_daily_loss(entry_prem, qty)
        self.trail_manager = TrailManager(expiry_mode=True)
        self.leg_tokens = {}

        logging.info(f"EXECUTED EXPIRY-DAY MAX-PAIN TRADE: {lots} lots of {tsym} at {entry_prem} "
                    f"(spot {spot:.0f}, max pain {pain:.0f}).")
        self.trade_log.append(f"{now_ist():%H:%M:%S} ENTRY EXPIRY {tsym} qty={qty} @ Rs{entry_prem:.1f}")

    def _dashboard_update(self, state):
        """Pulls current system state into the Dashboard's expected shape.
        Called from a background thread, roughly every `refresh` seconds."""
        state["status"] = (f"{self.current_regime} | "
                           f"{'HALTED' if self.risk_manager.halted else 'LIVE'} | "
                           f"overnight x{self.overnight_risk_multiplier:.1f} | "
                           f"flipper_guard={'ARMED' if self.allow_flipper else 'DISARMED'}")
        state["day_pnl"] = self.risk_manager.day_pnl
        state["kill_limit"] = -self.risk_manager.current_dynamic_max_loss
        state["profit_lock"] = self.risk_manager.daily_profit_target
        state["signals"] = self.premarket_signals
        state["latency_ms"] = (time.time() - self.last_tick_time) * 1000 if self.latest_spot else 0.0

        # Sentiment panel: real headlines, tagged with the aggregate score —
        # SentimentEnsemble only returns one combined score, not per-headline,
        # so every row intentionally shows the same number.
        state["sentiment"] = [(h, self.sent_score) for h in self.last_headlines[:5]]

        # Macro panel: overnight % moves already computed in build_signals()
        if self.today_row is not None:
            macro_cols = ["sp500_chg", "nasdaq_chg", "nikkei_chg", "kospi_chg",
                         "brent_chg", "dxy_chg", "us10y_chg", "usdinr_chg", "banknifty_chg"]
            state["macro"] = {c.replace("_chg", ""): {"last": "-", "chg_pct": getattr(self.today_row, c, 0.0)}
                              for c in macro_cols if hasattr(self.today_row, c)}

        state["chain"] = {
            "spot": self.latest_spot or "-",
            "pcr": f"{self.last_pcr:.2f}" if self.last_pcr is not None else "-",
            "max_pain": self.last_max_pain if self.last_max_pain is not None else "-",
            "ceiling": self.last_call_wall if self.last_call_wall is not None else "-",
            "floor": self.last_put_wall if self.last_put_wall is not None else "-",
            "vix": self.today_row.vix if self.today_row is not None else "-",
            "gex": "-",  # gamma exposure not currently computed live — see options_math.gamma_exposure if needed
        }

        # Positions: naked Position already matches Dashboard's expected shape.
        # MultiLegPosition doesn't (different fields), so adapt it with a
        # lightweight shim rather than changing Dashboard's render() contract.
        positions = []
        if self.active_position:
            positions.append(self.active_position)
        if self.active_multi_position:
            pos = self.active_multi_position
            leg_desc = "/".join(f"{l['side'][0]}{l['symbol'].split(':')[-1][-9:]}" for l in pos.legs)
            shim = type("DisplayPosition", (), {})()
            shim.symbol = f"{pos.kind}[{leg_desc}]"
            shim.qty = pos.qty
            shim.entry = pos.entry_net_cash / pos.qty if pos.qty else 0.0
            shim.stop = -pos.max_loss_per_unit
            shim.target = pos.max_loss_per_unit * 0.5
            positions.append(shim)
        state["positions"] = positions
        state["trades"] = self.trade_log

    def _start_dashboard(self, refresh=20.0):
        """Prints a full-state snapshot every `refresh` seconds from a
        background thread. Deliberately NOT a persistent Live-updating pane —
        that would fight with normal logging.info() calls from other threads
        for terminal ownership. A periodic snapshot coexists cleanly with
        the regular event log instead."""
        def loop():
            while not self._intentional_close:
                try:
                    self._dashboard_update(self.dashboard.state)
                    self.dashboard.snapshot()
                except Exception as e:
                    logging.warning(f"Dashboard render failed ({e})")
                time.sleep(refresh)
        self._dashboard_thread = threading.Thread(target=loop, daemon=True)
        self._dashboard_thread.start()

    def _start_watchdog(self):
        """Runs independently of ticks so EOD squareoff AND theta-timeout fire
        even if the feed goes quiet (e.g. a low-liquidity option with no
        trades for a while)."""
        def loop():
            while not self._intentional_close:
                now_t = now_ist().time()

                # Calling these every cycle (not just at squareoff) is what
                # lets theta-timeout fire even when ticks have gone quiet —
                # both functions already contain the full exit-decision logic
                # and are safe to call repeatedly with the latest known price.
                if self.active_position:
                    price = self.latest_opt_price or self.active_position.entry
                    self.evaluate_trail(price)

                if self.active_multi_position:
                    pos = self.active_multi_position
                    missing = [l["symbol"] for l in pos.legs if l["symbol"] not in self.latest_leg_prices]
                    if missing:
                        try:
                            q = self.kite.quote(missing)
                            for sym, data in q.items():
                                self.latest_leg_prices[sym] = data["last_price"]
                        except Exception as e:
                            logging.warning(f"Watchdog: failed to refresh missing leg quotes ({e}).")
                    self.evaluate_multi_trail()

                if now_t >= dtime(15, 35):  # safety cutoff well past close either way
                    self._intentional_close = True
                    if self.ticker:
                        self.ticker.close()
                    break
                time.sleep(15)
        self._watchdog_thread = threading.Thread(target=loop, daemon=True)
        self._watchdog_thread.start()

    def start_websocket(self):
        """Streams live prices and manages exits for whichever position type is open."""
        self.ticker = KiteTicker(self.adapter.api_key, self.kite.access_token)

        def on_ticks(ws, ticks):
            self.last_tick_time = time.time()
            for tick in ticks:
                token = tick['instrument_token']
                ltp = tick['last_price']

                if token == self.spot_token:
                    self.latest_spot = ltp
                elif token == self.option_token and self.active_position:
                    self.latest_opt_price = ltp
                    self.peak_opt_price = max(self.peak_opt_price, ltp)
                    self.evaluate_trail(ltp)
                elif token in self.leg_tokens:
                    sym = self.leg_tokens[token]
                    self.latest_leg_prices[sym] = ltp
                    if self.active_multi_position:
                        self.evaluate_multi_trail()

        def on_connect(ws, response):
            logging.info("WebSocket connected. Subscribing to tokens...")
            self._reconnect_attempts = 0
            tokens = [self.spot_token]
            if self.option_token:
                tokens.append(self.option_token)
            tokens.extend(self.leg_tokens.keys())
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_LTP, tokens)
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
        self.ticker.connect(threaded=False)  # Blocking loop

    def evaluate_trail(self, current_prem: float):
        """Exit logic for the single-leg naked position (TRENDING / expiry-day)."""
        if not self.active_position:
            return

        action, _ = self.trail_manager.update(current_prem, self.active_position.entry,
                                              self.active_position.qty, self.active_position.qty)

        if current_prem <= self.active_position.stop:
            action = "DISASTER_STOP"

        now_t = now_ist().time()
        if now_t >= self.risk_manager.squareoff:
            action = "EOD_SQUAREOFF"

        # Theta timeout: config.yaml has had theta_timeout_minutes per regime
        # for a while, but nothing ever read it. A trade that's gone nowhere
        # for that long is just bleeding time decay — free the capital for a
        # better setup instead of parking it till 15:15. Only fires on a
        # trade that ISN'T currently profitable — a winner is allowed to run.
        timeout_min = self._regime_config().get("theta_timeout_minutes", 90)
        elapsed_min = (now_ist() - self.active_position.entry_time).total_seconds() / 60.0
        if elapsed_min >= timeout_min and current_prem <= self.active_position.entry and action == "HOLD":
            action = "THETA_TIMEOUT"
            logging.info(f"Theta timeout: {elapsed_min:.0f} min elapsed (limit {timeout_min}) with no "
                        f"progress ({current_prem:.2f} vs entry {self.active_position.entry:.2f}).")

        if action == "PARTIAL_EXIT" and self.active_position:
            exit_qty = self.active_position.qty // 2
            if exit_qty > 0:
                pnl = self.broker.close_partial(self.active_position, exit_qty, current_prem)
                self.risk_manager.register_pnl(pnl)
                self.active_position.stop = self.active_position.entry  # breakeven on the runner
                logging.info(f"PARTIAL PROFIT BOOKED: {exit_qty} qty @ {current_prem}. "
                            f"Realized: Rs {pnl:.2f}. Remaining qty {self.active_position.qty} "
                            f"now stopped at breakeven ({self.active_position.stop:.2f}).")
                self.trade_log.append(f"{now_ist():%H:%M:%S} PARTIAL EXIT {exit_qty}qty @ Rs{current_prem:.1f} "
                                      f"PnL={pnl:+.1f}")
            return

        if action in ["FULL_EXIT", "STOP_LOSS", "DISASTER_STOP", "EOD_SQUAREOFF", "THETA_TIMEOUT"]:
            logging.info(f"TRAILING STOP TRIGGERED ({action}). Current price: {current_prem}. Closing position.")
            pnl = self.broker.close(self.active_position, current_prem)
            logging.info(f"Position Closed. Realized PnL: Rs {pnl:.2f}. New Capital: Rs {self.broker.capital:.2f}")
            self.trade_log.append(f"{now_ist():%H:%M:%S} EXIT ({action}) PnL={pnl:+.1f} "
                                  f"Capital=Rs{self.broker.capital:.0f}")
            self.active_position = None

            self.risk_manager.register_pnl(pnl)
            if self.risk_manager.halted:
                logging.info(f"RiskManager HALTED for the day. Day PnL: Rs {self.risk_manager.day_pnl:.2f}")

            self._intentional_close = True
            if self.ticker:
                self.ticker.close()

    def evaluate_multi_trail(self):
        """
        Exit logic for spreads/condors: profit target at +50% of max
        loss-equivalent, stop at -90% of max loss, or EOD squareoff.
        Needs a fresh price for every leg — start_websocket subscribes to
        all leg tokens whenever a multi-leg position is open.
        """
        pos = self.active_multi_position
        if not pos:
            return
        if not all(l["symbol"] in self.latest_leg_prices for l in pos.legs):
            return  # wait for a fresh quote on every leg before evaluating

        exit_prices = {l["symbol"]: self.latest_leg_prices[l["symbol"]] for l in pos.legs}

        mtm_flow = 0.0
        for l in pos.legs:
            price = exit_prices[l["symbol"]]
            closing_side = "SELL" if l["side"] == "BUY" else "BUY"
            mtm_flow += (price if closing_side == "SELL" else -price) * pos.qty
        unrealized_pnl = pos.entry_net_cash + mtm_flow
        max_loss_total = pos.max_loss_per_unit * pos.qty

        now_t = now_ist().time()
        reason = None
        if now_t >= self.risk_manager.squareoff:
            reason = "EOD_SQUAREOFF"
        elif unrealized_pnl <= -max_loss_total * 0.9:
            reason = "STOP_LOSS"
        elif unrealized_pnl >= max_loss_total * 0.5:
            reason = "PROFIT_TARGET"
        else:
            timeout_min = self._regime_config().get("theta_timeout_minutes", 90)
            elapsed_min = (now_ist() - pos.entry_time).total_seconds() / 60.0
            if elapsed_min >= timeout_min and unrealized_pnl <= 0:
                reason = "THETA_TIMEOUT"
                logging.info(f"Theta timeout: {elapsed_min:.0f} min elapsed (limit {timeout_min}) with no "
                            f"progress (unrealized PnL Rs {unrealized_pnl:.2f}).")

        if reason:
            pnl = self.broker.close_multi(pos, exit_prices)
            logging.info(f"MULTI-LEG EXIT ({reason}, {pos.kind}): Realized PnL Rs {pnl:.2f}. "
                        f"New capital Rs {self.broker.capital:.2f}")
            self.trade_log.append(f"{now_ist():%H:%M:%S} EXIT {pos.kind} ({reason}) PnL={pnl:+.1f} "
                                  f"Capital=Rs{self.broker.capital:.0f}")
            self.active_multi_position = None
            self.leg_tokens = {}

            self.risk_manager.register_pnl(pnl)
            if self.risk_manager.halted:
                logging.info(f"RiskManager HALTED for the day. Day PnL: Rs {self.risk_manager.day_pnl:.2f}")

            self._intentional_close = True
            if self.ticker:
                self.ticker.close()


def main():
    print("Initializing AutoBot Live Paper Trading Harness...")
    kite_adapter = setup_kite()

    trader = LivePaperTrader(kite_adapter)
    trader.risk_manager.new_day()  # reset kill switch / profit lock / day_pnl for this session
    plan = trader.run_premarket_analysis()

    if plan is None:
        logging.info("Premarket bias is NO_TRADE, but the system will keep checking for a "
                     "qualifying setup throughout the day rather than stopping here.")

    trader._start_dashboard(refresh=20.0)  # periodic full-state snapshot for the rest of the session

    spot = trader.wait_for_market_open()

    # Retry loop: the system does NOT get just one look at the market at
    # 9:15 and then give up for the day. It re-checks for a qualifying setup
    # — across regimes, using a fresh Kite option-chain snapshot and the
    # current spot every time — until either a trade opens, the entry
    # cutoff passes, or RiskManager halts. This does not lower the
    # confidence threshold or force a trade on a genuinely quiet day; it
    # just gives the system many chances instead of one, which is the
    # honest way to reduce "zero trades all week" without weakening any
    # risk gate.
    NEW_ENTRY_CUTOFF = dtime(14, 45)   # leaves 30 min of runway before the 15:15 squareoff
    RETRY_POLL_SECONDS = 900           # re-check every 15 minutes when nothing qualifies

    while True:
        now = now_ist()
        if now.time() >= NEW_ENTRY_CUTOFF:
            logging.info(f"Past the new-entry cutoff ({NEW_ENTRY_CUTOFF}) with no open position — "
                        f"not enough runway left before squareoff for a fresh trade. Ending session.")
            break
        if trader.risk_manager.halted:
            logging.info("RiskManager has halted trading for the day (kill switch or profit lock hit). "
                        "No further entries will be attempted.")
            break

        try:
            quote = trader.kite.quote("NSE:NIFTY 50")
            spot = quote["NSE:NIFTY 50"]["last_price"]
            trader.latest_spot = spot
        except Exception as e:
            logging.warning(f"Spot quote refresh failed ({e}); retrying next cycle.")
            time.sleep(RETRY_POLL_SECONDS)
            continue

        trader.execute_paper_trade(spot)

        if trader.active_position or trader.active_multi_position:
            # Fresh per-position session state so the websocket/watchdog for
            # THIS trade start cleanly (a prior trade today may have already
            # used and torn down its own instance of both).
            trader._intentional_close = False
            trader._reconnect_attempts = 0
            trader._watchdog_thread = None
            try:
                trader.start_websocket()  # blocks until THIS position closes
            except KeyboardInterrupt:
                logging.warning("Ctrl+C received — closing any open position at last known price before exit.")
                trader._intentional_close = True
                if trader.active_position and trader.latest_opt_price:
                    pnl = trader.broker.close(trader.active_position, trader.latest_opt_price)
                    trader.risk_manager.register_pnl(pnl)
                    logging.info(f"Manual close PnL: Rs {pnl:.2f}")
                if trader.active_multi_position:
                    pos = trader.active_multi_position
                    try:
                        q = trader.kite.quote([l["symbol"] for l in pos.legs])
                        exit_prices = {sym: data["last_price"] for sym, data in q.items()}
                    except Exception:
                        exit_prices = {l["symbol"]: l["entry"] for l in pos.legs}  # last resort
                    pnl = trader.broker.close_multi(pos, exit_prices)
                    trader.risk_manager.register_pnl(pnl)
                    logging.info(f"Manual multi-leg close PnL: Rs {pnl:.2f}")
                if trader.ticker:
                    trader.ticker.close()
                break  # Ctrl+C ends the whole session, not just this one trade

            logging.info("Position closed. Checking whether another opportunity exists "
                        "for the rest of the day...")
            continue  # loop back and look for another trade
        else:
            logging.info(f"No qualifying setup this check. Re-checking in "
                        f"{RETRY_POLL_SECONDS // 60} minutes...")
            time.sleep(RETRY_POLL_SECONDS)

    print("Session loop ended (cutoff reached, halted, or manually stopped).")


if __name__ == "__main__":
    main()
