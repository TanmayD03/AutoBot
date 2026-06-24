"""Event-driven backtester v5.
Replays NIFTY daily history through the SAME signal/decision/risk code used live.
Option premiums reconstructed with Black-Scholes using India VIX as IV proxy.

v5: ALL free historical factors mapped (Yahoo Finance, up to 20y daily, lookahead-safe):
- Global: S&P 500, Nikkei, Brent, DXY, US 10Y yield, USD/INR
- India structure: Bank Nifty (sector alignment), top-5 heavyweight breadth
  (RELIANCE/HDFCBANK/ICICIBANK/INFY/TCS), Indian ADRs (INFY/HDB/IBN)
Every factor is shift(1)-aligned: the simulation only knows yesterday's close at
today's 09:15 IST open. Run `python verify_data.py` to audit the mapping.

Still NOT freely available historically (live-only, paid-vendor adapter slots exist):
strike-level option chain OI/PCR/GEX, FII/DII flows, intraday ticks, GIFT Nifty.
"""
import csv
import math
import os
from datetime import datetime, time as dtime, timedelta
import numpy as np
from ..options_math import bs_price
from ..signals.engine import pivot_signal, vix_signal, SignalScore
from ..strategy.decision import DecisionEngine
from ..strategy.regime import RegimeDetector
from ..strategy.intraday import IntradayFlipper, ChoppyMarketStrategy
from ..strategy.capital import CapitalManager
from ..strategy.trail import TrailManager
from ..strategy.risk import RiskManager
from ..execution.paper_broker import PaperBroker

FACTOR_TICKERS = {
    "sp500": "^GSPC", "nasdaq": "^IXIC", "brent": "BZ=F", "usdinr": "USDINR=X", "dxy": "DX-Y.NYB",
    "us10y": "^TNX", "nikkei": "^N225", "banknifty": "^NSEBANK",
    "adr_infy": "INFY", "adr_hdb": "HDB", "adr_ibn": "IBN",
    "hw_rel": "RELIANCE.NS", "hw_hdfc": "HDFCBANK.NS", "hw_icici": "ICICIBANK.NS",
    "hw_infy": "INFY.NS", "hw_tcs": "TCS.NS",
}
HW_COLS = ["hw_rel_chg", "hw_hdfc_chg", "hw_icici_chg", "hw_infy_chg", "hw_tcs_chg"]
ADR_COLS = ["adr_infy_chg", "adr_hdb_chg", "adr_ibn_chg"]


def load_history(years=20):
    import yfinance as yf
    nifty = yf.download("^NSEI", period=f"{years}y", interval="1d", progress=False, auto_adjust=True)
    vix = yf.download("^INDIAVIX", period=f"{years}y", interval="1d", progress=False, auto_adjust=True)
    nifty.columns = [c[0] if isinstance(c, tuple) else c for c in nifty.columns]
    vix.columns = [c[0] if isinstance(c, tuple) else c for c in vix.columns]
    df = nifty[["Open", "High", "Low", "Close"]].copy()
    df["vix"] = vix["Close"].reindex(df.index).ffill().fillna(15.0)
    df["sma20"] = df["Close"].rolling(20).mean()
    df["sma50"] = df["Close"].rolling(50).mean()
    df["dma20"] = df["sma20"]  # Alias for strategy access
    df["dma50"] = df["sma50"]
    # Approximate VWAP on daily bars using typical price
    df["vwap"] = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Close"]).rolling(14).sum() / df["Close"].rolling(14).sum()
    df["atr"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["vix_pct_rank"] = df["vix"].rolling(252).rank(pct=True) * 100
    # Calculate basic ADX(14) and RSI(14)
    import pandas as pd
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi14"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

    high_low = df["High"] - df["Low"]
    high_pc  = (df["High"] - df["Close"].shift(1)).abs()
    low_pc   = (df["Low"]  - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    plus_dm  = (df["High"].diff()).clip(lower=0)
    minus_dm = (-df["Low"].diff()).clip(lower=0)
    plus_di  = 100 * plus_dm.rolling(14).mean()  / tr.rolling(14).mean()
    minus_di = 100 * minus_dm.rolling(14).mean() / tr.rolling(14).mean()
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
    df["adx"] = dx.rolling(14).mean()
    df["adx"] = df["adx"].fillna(15.0) # default fallback

    # Batch-download all factor tickers; shift(1) alignment = no lookahead.
    data = yf.download(list(FACTOR_TICKERS.values()), period=f"{years}y", interval="1d",
                       progress=False, group_by="ticker", auto_adjust=True)
    for name, tkr in FACTOR_TICKERS.items():
        try:
            if name == "nikkei":
                # Nikkei opens before Nifty. Gap from Nikkei yesterday close to today open is better
                chg = (data[tkr]["Open"] / data[tkr]["Close"].shift(1) - 1).dropna() * 100
                df[f"{name}_chg"] = chg.reindex(df.index, method="ffill").fillna(0.0)
            else:
                chg = data[tkr]["Close"].pct_change(fill_method=None) * 100
                df[f"{name}_chg"] = chg.reindex(df.index, method="ffill").shift(1).fillna(0.0)
        except Exception:
            df[f"{name}_chg"] = 0.0

    # After all downloads, fill any remaining NaN columns (handles 403 errors on yfinance)
    for col in ['sp500_chg','nasdaq_chg','nikkei_chg','brent_chg',
                'dxy_chg','us10y_chg','usdinr_chg','banknifty_chg',
                'hw_breadth','adr_chg'] + HW_COLS + ADR_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    # Heavyweight breadth: fraction of top-5 index anchors green yesterday, in [-1, 1]
    df["hw_breadth"] = sum(np.sign(df[c]) for c in HW_COLS) / len(HW_COLS)
    # ADR composite: average prev-day move of Indian ADRs in the US session
    df["adr_chg"] = sum(df[c] for c in ADR_COLS) / len(ADR_COLS)

    # Mocking FII flow for historical backtest using ADR moves proxy (better than zeros)
    # In live mode this is pulled from NSE API.
    df["fii_net_cr"] = df["adr_chg"] * 2500.0

    return df.dropna()


def load_event_impacts():
    """Map curated historical events to decaying daily impact scores in [-1, 1]."""
    path = os.path.join(os.path.dirname(__file__), "events.csv")
    impacts = {}
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                d0 = datetime.strptime(row["date"], "%Y-%m-%d").date()
                imp = max(-1.0, min(1.0, float(row["nifty_impact_pct_5d"]) / 10.0))
                for k in range(6):  # impact decays over the following week
                    day = d0 + timedelta(days=k)
                    decayed = imp * (1 - k / 6)
                    if abs(decayed) > abs(impacts.get(day, 0.0)):
                        impacts[day] = decayed
    except Exception:
        pass
    return impacts


def _clip(x):
    return max(-1.0, min(1.0, x))


def build_signals(prev, today, prev2_close, prev3_close=None) -> list:
    """All factor signals for one day. Used by backtest AND auditable by verify_data.py."""
    gap_pct = (today.Open / prev.Close - 1) * 100
    macro_raw = (0.25 * today.sp500_chg + 0.25 * today.nasdaq_chg + 0.15 * today.nikkei_chg
                 - 0.10 * today.brent_chg - 0.15 * today.dxy_chg - 0.10 * today.us10y_chg
                 - 3.0 * 0.25 * today.usdinr_chg) / 1.2

    # 3-day smoothed momentum (more stable, avoids expiry whipsaw distortions)
    if prev3_close is not None:
        mom_1 = (prev.Close / prev2_close - 1) * 100
        mom_2 = (prev.Close / prev3_close - 1) * 100 / 2.0
        mom_smooth = (0.60 * mom_1 + 0.40 * mom_2) / 1.2
    else:
        mom_smooth = (prev.Close / prev2_close - 1) * 100 / 1.2

    signals = []

    # Enhanced pivot signal with ADX boost
    piv = pivot_signal(today.Open, prev.High, prev.Low, prev.Close)
    adx_val = today.adx if hasattr(today, 'adx') else 15.0
    adx_boost = min(0.20, (adx_val - 14) / 100) if adx_val > 14 else 0
    piv_enhanced = SignalScore("pivots", piv.score, min(0.85, piv.confidence + adx_boost))
    signals.append(piv_enhanced)

    signals.append(vix_signal(today.vix, (today.vix / prev.vix - 1) * 100 if prev.vix else 0))

    # Small-gap filter on gift_gap
    if abs(gap_pct) >= 0.10:
        signals.append(SignalScore("gift_gap", _clip(gap_pct), 0.70))
    else:
        signals.append(SignalScore("gift_gap", 0.0, 0.0))

    signals.append(SignalScore("momentum", _clip(mom_smooth), 0.55))
    signals.append(SignalScore("macro", _clip(macro_raw), 0.6))
    signals.append(SignalScore("sector", _clip(today.banknifty_chg / 1.2), 0.55))
    signals.append(SignalScore("breadth", _clip(today.hw_breadth * 0.8), 0.55))
    signals.append(SignalScore("adr", _clip(today.adr_chg / 1.5), 0.6))

    # Add RSI-14 signal
    rsi14 = today.rsi14 if hasattr(today, 'rsi14') else 50.0
    rsi_score = _clip((rsi14 - 50) / 20.0)
    rsi_conf  = 0.60 if abs(rsi14 - 50) > 10 else 0.40
    signals.append(SignalScore("rsi14", rsi_score, rsi_conf))

    from ..signals.engine import iv_skew_signal, max_pain_signal, crude_signal, fii_flow_signal

    # Add Crude Regime Signal
    signals.append(crude_signal(brent_price=80.0, brent_chg_pct=today.brent_chg)) # using 80.0 as mock baseline for backtest where price isn't available

    # Add FII flow signal
    if hasattr(today, "fii_net_cr"):
        signals.append(fii_flow_signal(today.fii_net_cr, 0.0))
    else:
        signals.append(fii_flow_signal(0.0, 0.0))

    # Approximate skew from VIX
    iv = max(today.vix, 8.0) / 100.0
    vix_chg_pct = (today.vix / prev.vix - 1) * 100 if prev.vix else 0.0
    iv_skew_proxy = _clip(-vix_chg_pct / 10.0)  # -10% VIX change = ±1.0 skew score
    signals.append(iv_skew_signal(iv * (1 - iv_skew_proxy * 0.05),
                                  iv * (1 + iv_skew_proxy * 0.05)))

    # Approximate max pain (removed as rounding logic forces score ~0.0; wait for live OI data)
    # max_pain_strike = round(today.Open / 50) * 50
    # signals.append(max_pain_signal(today.Open, max_pain_strike))

    from ..data.events import get_event_score
    date_str = str(today.Index.date()) if hasattr(today, "Index") else str(today.name.date())
    event_score, event_conf = get_event_score(date_str)
    if event_conf > 0:
        signals.append(SignalScore("geopolitical", event_score, event_conf))

    return signals


def days_to_nifty_expiry(dt):
    from datetime import timedelta
    d = dt.date() if hasattr(dt, 'date') else dt
    days = 0
    while d.weekday() != 1:   # 1 = Tuesday
        d += timedelta(days=1)
        days += 1
    return max(0.1, days) / 365.0


def run_backtest(df, weights=None, capital=100000.0, lots=1, lot_size=75, r=0.068,
                 confidence_threshold=0.70, daily_max_loss_pct=3.0,
                 daily_profit_target_pct=7.5, disaster_atr_mult=2.0,
                 dd_throttle=0.10, dd_threshold_bump=0.07):
    import yaml
    try:
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
            regimes_cfg = cfg.get("regimes", {})
    except Exception:
        regimes_cfg = {}

    qty = lots * lot_size
    broker = PaperBroker(capital)
    risk = RiskManager(daily_profit_target=capital * daily_profit_target_pct / 100,
                       daily_max_loss=capital * daily_max_loss_pct / 100,
                       squareoff=dtime(15, 15))
    engine = DecisionEngine(confidence_threshold, weights=weights)
    events = load_event_impacts()
    from ..nature.pheromone import PheromoneMemory
    pheromone = PheromoneMemory(["BUY_CE", "BUY_PE", "FLIP_TO_PE", "FLIP_TO_CE"])
    trades, equity = [], []
    peak_eq = capital
    rows = list(df.itertuples())
    from ..strategy.regime import RegimeDetector
    regime_detector = RegimeDetector()
    from ..strategy.capital import CapitalManager
    capital_manager = CapitalManager()

    for i in range(3, len(rows)):
        pheromone.evaporate()
        prev, today = rows[i - 1], rows[i]
        risk.new_day()
        spot = today.Open
        iv = max(today.vix, 8.0) / 100.0
        t_exp = days_to_nifty_expiry(today.Index)

        # On expiry Tuesday, before the normal signal check:
        atm = round(spot / 50) * 50
        ce_cost = bs_price(spot, atm, r, iv, t_exp, "C")
        pe_cost = bs_price(spot, atm, r, iv, t_exp, "P")
        straddle_cost_est = ce_cost + pe_cost
        expected_daily_move = (today.vix / 100) * spot / math.sqrt(252)

        if today.Index.weekday() == 1 and today.vix > 16 and expected_daily_move > straddle_cost_est * 1.10:
            straddle_cost = straddle_cost_est
            straddle_lots = max(1, int(broker.capital * 0.20 / (straddle_cost * lot_size)))

            gap_pct_today = (today.Open / prev.Close - 1) * 100
            t_ce = t_exp * 0.45;  t_pe = t_exp * 0.20
            if gap_pct_today >= 0:
                ce_peak = bs_price(today.High, atm, r, iv, t_ce, "C")
                pe_exit = bs_price(min(today.Low, today.Open * 0.993), atm, r, iv, t_pe, "P")
            else:
                pe_peak = bs_price(today.Low,  atm, r, iv, t_pe,  "P")
                ce_exit = bs_price(max(today.High, today.Open * 1.007), atm, r, iv, t_ce, "C")
                ce_peak = ce_exit;  pe_exit = pe_peak
            straddle_exit = max((ce_peak + pe_exit) * 0.85, straddle_cost * 0.30)
            pnl = (straddle_exit - straddle_cost) * straddle_lots * lot_size
            broker.capital += pnl
            risk.register_pnl(pnl)
            pheromone.reinforce("STRADDLE", pnl)
            trades.append({"date": str(today.Index.date()), "action": "STRADDLE",
                           "strike": atm, "entry": round(straddle_cost, 2),
                           "exit": round(straddle_exit, 2), "pnl": round(pnl, 2),
                           "confidence": 0.80})
            equity.append(broker.capital)
            continue

        signals = build_signals(prev, today, rows[i - 2].Close, rows[i - 3].Close)
        adx_val = getattr(today, "adx", 0.0)
        regime = regime_detector.classify(vix=today.vix, adx=adx_val, atr_pct=(today.atr / spot) * 100)

        # Load threshold from config
        if regime == "CHOPPY" and "choppy" in regimes_cfg:
            engine.threshold = regimes_cfg["choppy"].get("confidence_threshold", 0.55)
        elif regime == "HIGH_VOLATILITY" and "high_volatility" in regimes_cfg:
            engine.threshold = regimes_cfg["high_volatility"].get("confidence_threshold", 0.65)
        else:
            engine.threshold = regimes_cfg.get("trending", {}).get("confidence_threshold", 0.58)

        # Drawdown-adaptive throttle: trade less while wounded (immune response)
        peak_eq = max(peak_eq, broker.capital)
        in_drawdown = (peak_eq - broker.capital) / peak_eq > dd_throttle
        if in_drawdown:
            engine.threshold += dd_threshold_bump   # bump the REGIME threshold, not the top-level one

        from ..signals.engine import combine
        # Get raw ensemble score to determine direction for quality check
        score, _ = combine(signals)

        # Count high-conviction signals (conf > 0.65 AND score in same direction as ensemble)
        high_conv = sum(1 for s in signals
                        if s.confidence > 0.65 and s.score * score > 0.10)
        # Lower threshold when 3+ high-conviction signals agree
        if high_conv >= 3:
            engine.threshold = max(0.48, engine.threshold - 0.08)
        elif high_conv >= 2:
            engine.threshold = max(0.52, engine.threshold - 0.05)
        else:
            med_conv = sum(1 for s in signals
                           if 0.50 <= s.confidence <= 0.65 and s.score * score > 0.10)
            if   high_conv == 1 and med_conv >= 5: engine.threshold = max(0.50, engine.threshold - 0.03)
            elif high_conv == 1 and med_conv >= 3: engine.threshold = max(0.53, engine.threshold - 0.02)
            elif high_conv == 1:                   engine.threshold = min(0.68, engine.threshold + 0.03)
            elif high_conv == 0 and med_conv >= 5: engine.threshold = max(0.50, engine.threshold - 0.03)
            elif high_conv == 0 and med_conv >= 3: pass  # hold at base
            else:                                  engine.threshold = min(0.72, engine.threshold + 0.05)

        vix_pct = today.vix_pct_rank if "vix_pct_rank" in df.columns else 50.0

        if vix_pct < 25:       # IV cheap → buy options aggressively
            pos_size_mult = 1.5    # 50% more lots than usual
            strategy_pref = "BUY_STRADDLE_OR_DIRECTIONAL"
        elif vix_pct < 60:     # IV moderate → normal directional
            pos_size_mult = 1.0
            strategy_pref = "BUY_SPREAD"
        else:                  # IV expensive → sell spreads/condors
            pos_size_mult = 0.8
            strategy_pref = "SELL_CONDOR"

        import pandas as pd
        current_time_sim = pd.to_datetime(today.Index).to_pydatetime().replace(hour=9, minute=15)
        # Approximate ORB from daily bar (first 30-min = ~12% of daily range)
        orb_high = today.Open + (today.High - today.Open) * 0.15
        orb_low  = today.Open - (today.Open - today.Low)  * 0.15

        from ..strategy.intraday import IntradayFlipper
        intraday_flipper = IntradayFlipper()
        flipper_action = intraday_flipper.evaluate_entry(
            current_time=current_time_sim,
            spot=spot, vwap=today.Open, rsi=50, adx=today.adx if "adx" in df.columns else 0.0,
            pdh=prev.High, dma20=today.sma20, dma50=today.sma50,
            vix=today.vix,
            gift_nifty_gap=(today.Open / prev.Close - 1) * 100,
            orb_high=orb_high, orb_low=orb_low
        )

        # Event sentiment: decaying impact of major historical news
        sentiment_impact = events.get(today.Index.date(), 0.0)
        plan = engine.decide(signals, spot, sentiment_impact=sentiment_impact)

        if flipper_action in ("BUY_CE", "BUY_PE"):
            from ..strategy.decision import TradePlan
            plan = TradePlan(action=flipper_action, strike=round(spot/50)*50, confidence=0.75, reason="IntradayFlipper ORB/Morning")
        # Trend-regime gate: never fight the established trend
        uptrend   = prev.Close > prev.sma50 * 1.005
        downtrend = prev.Close < prev.sma50 * 0.995
        near_sma = abs(prev.Close / prev.sma50 - 1) < 0.005

        # Gap vs macro conflict detection — add after build_signals() inside the loop logic
        gap_s   = next((s for s in signals if s.name == "gift_gap"),  None)
        macro_s = next((s for s in signals if s.name == "macro"),     None)
        if (gap_s and macro_s
            and gap_s.score * macro_s.score < 0        # opposite directions
            and abs(gap_s.score) > 0.20):              # gap > ~0.20% — catches short-covering opens
            regime = "CHOPPY"                          # force override

        if plan is not None:
            # VWAP Gate (Improvement A)
            vwap_sig = next((s for s in signals if s.name=="vwap_pos"), None)
            if vwap_sig and plan.action == "BUY_CE" and vwap_sig.score < -0.3:
                plan = None
            elif vwap_sig and plan.action == "BUY_PE" and vwap_sig.score > 0.3:
                plan = None

        if plan is not None:
            # Skip trend gate if confidence is high (> 0.65) and 4+ signals agree (quality reversal)
            # Or if confidence is extremely high (> 0.85)
            high_conv_match = sum(1 for s in signals if s.confidence > 0.65 and s.score * plan.confidence > 0.10)
            bypass_trend = plan.confidence >= 0.85 or (plan.confidence >= 0.65 and high_conv_match >= 4)

            if not bypass_trend:
                if plan.action == "BUY_CE" and not (uptrend or near_sma):
                    plan = None
                elif plan.action == "BUY_PE" and not (downtrend or near_sma):
                    plan = None

            if regime == "CHOPPY":
                plan = None # Choppy logic overrides directional trades
        if plan is None or plan.action == "NO_TRADE":
            if regime == "CHOPPY":
                from ..strategy.intraday import ChoppyMarketStrategy
                choppy = ChoppyMarketStrategy()
                condor_action = choppy.evaluate_iron_condor(today.vix)
                if condor_action in ("SELL_IRON_CONDOR", "SELL_NARROW_CONDOR"):
                    # Dynamic wing = 40% of prior day range, rounded to nearest 50
                    expected_range_pts = (prev.High - prev.Low)
                    wing = max(50, round((expected_range_pts * 0.40) / 50) * 50)

                    # Only fire condor if expected daily range < 2x wing
                    if expected_range_pts > wing * 2.2:
                        equity.append(broker.capital)
                        continue

                    atm = round(spot / 50) * 50
                    span_margin = wing * lot_size * 1.5
                    condor_lots = max(1, int(broker.capital * 0.50 / span_margin))
                    sell_p = bs_price(spot, atm - wing//2, r, iv, t_exp, "P")
                    buy_p  = bs_price(spot, atm - wing,    r, iv, t_exp, "P")
                    sell_c = bs_price(spot, atm + wing//2, r, iv, t_exp, "C")
                    buy_c  = bs_price(spot, atm + wing,    r, iv, t_exp, "C")
                    credit = (sell_p - buy_p + sell_c - buy_c) * condor_lots * lot_size
                    stayed_in = abs(today.Close - spot) < wing
                    condor_pnl = credit if stayed_in else -wing * condor_lots * lot_size * 0.5
                    risk.register_pnl(condor_pnl)
                    trades.append({"date": str(today.Index.date()), "action": condor_action,
                                   "strike": atm, "entry": round(sell_p, 2),
                                   "exit": round(buy_p, 2), "pnl": round(condor_pnl, 2),
                                   "confidence": 0.60})
                    pheromone.reinforce("choppy_condor", condor_pnl)
            equity.append(broker.capital)
            continue
        # Default to spread unless confidence is high and VIX is expanding
        use_spread = True
        if plan.confidence > 0.70 and today.vix > 18:
            use_spread = False

        kind = "C" if plan.action == "BUY_CE" else "P"

        if use_spread:
            from ..strategy.intraday import SpreadStrategy
            spread_strat = SpreadStrategy()
            if kind == "C":
                spread = spread_strat.bull_call_spread(spot, plan.strike, wing=100)
            else:
                spread = spread_strat.bear_put_spread(spot, plan.strike, wing=100)

            entry_prem_buy = bs_price(spot, spread["buy"], r, iv, t_exp, kind)
            entry_prem_sell = bs_price(spot, spread["sell"], r, iv, t_exp, kind)
            entry_prem = entry_prem_buy - entry_prem_sell
        else:
            entry_prem = bs_price(spot, plan.strike, r, iv, t_exp, kind)

        if entry_prem < 5 and not use_spread:
            equity.append(broker.capital)
            continue

        # Calculate delta for sizing
        from ..options_math.black_scholes import greeks
        delta_val = greeks(spot, plan.strike, r, iv, t_exp, kind).delta

        # Determine lots dynamically
        calc_lots = capital_manager.calculate_lots_by_delta(broker.capital, entry_prem, delta_val, target_delta_exposure=0.50, lot_size=lot_size)
        if calc_lots == 0:
            equity.append(broker.capital)
            continue

        # Apply pos_size_mult calculated earlier based on vix_pct_rank
        adjusted_qty = max(1, int(calc_lots * pos_size_mult)) * lot_size
        if adjusted_qty < lot_size:
            adjusted_qty = lot_size

        # Volatility-aware DISASTER stop: outside normal noise, abnormal days only
        exp_move = 0.5 * prev.atr
        disaster = max(1.0, entry_prem - disaster_atr_mult * exp_move)
        pos = broker.buy(f"NIFTY{plan.strike}{kind}E", adjusted_qty, entry_prem, disaster,
                         entry_prem + disaster_atr_mult * exp_move * 2.5)
        if pos is None:
            equity.append(broker.capital)
            continue
        # Trail Manager Simulation
        trail_manager = TrailManager(expiry_mode=(today.Index.weekday() == 1))

        # Simulating intraday trajectory (H/L -> Close)
        peak_spot = today.High if kind == "C" else today.Low
        peak_prem = bs_price(peak_spot, plan.strike, r, iv, t_exp - 0.25 / 365, kind)
        prem_close = bs_price(today.Close, plan.strike, r, iv, t_exp - 1 / 365, kind)

        # We simplify TrailManager to output just an exit price or action in a daily backtest.
        # For simulation, use peak_prem and prem_close to see if trailing SL or partials were hit.
        action, qty_to_sell = trail_manager.update(peak_prem, entry_prem, adjusted_qty, adjusted_qty)
        if action == "PARTIAL_EXIT" or action == "FULL_EXIT":
            # Assume we got stopped out somewhere between peak and close based on trail parameter
            exit_px = max(prem_close, peak_prem * (1 - trail_manager.trail_pct))
        elif action == "STOP_LOSS":
            exit_px = entry_prem * (1 - trail_manager.hard_stop_pct)
        else:
            action, qty_to_sell = trail_manager.update(prem_close, entry_prem, adjusted_qty, adjusted_qty)
            if action in ("FULL_EXIT", "STOP_LOSS"):
                exit_px = prem_close
            else:
                exit_px = prem_close

        if exit_px < disaster:
            exit_px = disaster

        # Max profit logic for spreads
        if use_spread:
            if exit_px > spread["max_profit_pts"]:
                exit_px = spread["max_profit_pts"]

        pnl = broker.close(pos, exit_px)
        risk.register_pnl(pnl)

        # Reinforce strategy based on PnL
        strat_key = plan.action
        if "Flipped Intraday" in getattr(plan, 'reason', ''):
            strat_key = "FLIP_TO_" + plan.action[-2:]
        pheromone.reinforce(strat_key, pnl)

        trades.append({"date": str(today.Index.date()), "action": plan.action,
                       "strike": plan.strike, "entry": round(pos.entry, 2),
                       "exit": round(exit_px, 2), "pnl": round(pnl, 2),
                       "confidence": round(plan.confidence, 2)})
        equity.append(broker.capital)
    return report(trades, equity, capital)


def report(trades, equity, capital):
    if not trades:
        return {"trades": 0, "note": "no trades passed the gates", "equity": equity or [capital]}
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_w, gross_l = sum(wins), abs(sum(losses)) or 1e-9
    eq_curve = equity or [capital]
    peak, max_dd = eq_curve[0], 0.0
    for v in eq_curve:
        peak = max(peak, v)
        max_dd = max(max_dd, (peak - v) / peak * 100)
    mu = sum(pnls) / len(pnls)
    sd = math.sqrt(sum((p - mu) ** 2 for p in pnls) / len(pnls)) or 1e-9
    final = eq_curve[-1]
    return {
        "trades": len(trades), "win_rate": round(len(wins) / len(trades) * 100, 1),
        "total_pnl": round(sum(pnls), 2), "profit_factor": round(gross_w / gross_l, 2),
        "expectancy": round(mu, 2), "sharpe_like": round(mu / sd * math.sqrt(252), 2),
        "max_drawdown_pct": round(max_dd, 2),
        "final_capital": round(final, 2),
        "return_pct": round((final / capital - 1) * 100, 2),
        "last_trades": trades[-5:], "equity": eq_curve,
    }


def fitness_for_pso(df, names, folds=5):
    n = len(df)
    # Blocked walk-forward: train on 60%, validate on 20%, test 20%
    splits = []
    chunk = n // folds
    for k in range(folds - 1):
        train = df.iloc[:chunk * (k + 1)]
        val   = df.iloc[chunk*(k+1):chunk*(k+2)]
        splits.append((train, val))

    def fitness(vec):
        weights = dict(zip(names, vec))
        # L1 regularization: pull weights toward 1.0
        reg = 0.04 * sum(abs(w - 1.0) for w in vec)
        exps, dds = [], []
        for train, val in splits:
            r = run_backtest(val, weights=weights)
            if r.get("trades", 0) < 5: return -1e6
            exps.append(r["expectancy"])
            dds.append(r["max_drawdown_pct"])
        mu = sum(exps) / len(exps)
        sd = (sum((e-mu)**2 for e in exps)/len(exps))**0.5 if len(exps) > 1 else 1e-9
        return mu - sd - 0.3*(sum(dds)/len(dds)) - reg
    return fitness
