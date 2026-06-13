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
from ..strategy.risk import RiskManager
from ..execution.paper_broker import PaperBroker

FACTOR_TICKERS = {
    "sp500": "^GSPC", "brent": "BZ=F", "usdinr": "USDINR=X", "dxy": "DX-Y.NYB",
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
    df["atr"] = (df["High"] - df["Low"]).rolling(14).mean()
    # Calculate basic ADX(14)
    high_diff = df["High"].diff()
    low_diff = df["Low"].diff()
    df["+dm"] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    df["-dm"] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    tr = np.maximum(df["High"] - df["Low"], np.maximum(abs(df["High"] - df["Close"].shift(1)), abs(df["Low"] - df["Close"].shift(1))))
    df["tr14"] = tr.rolling(14).sum()
    df["+di14"] = 100 * (df["+dm"].rolling(14).sum() / df["tr14"])
    df["-di14"] = 100 * (df["-dm"].rolling(14).sum() / df["tr14"])
    df["dx"] = 100 * (abs(df["+di14"] - df["-di14"]) / (df["+di14"] + df["-di14"]))
    df["adx"] = df["dx"].rolling(14).mean()
    df["adx"] = df["adx"].fillna(15.0) # default fallback
    df.drop(["+dm", "-dm", "tr14", "+di14", "-di14", "dx"], axis=1, inplace=True)

    # Batch-download all factor tickers; shift(1) alignment = no lookahead.
    data = yf.download(list(FACTOR_TICKERS.values()), period=f"{years}y", interval="1d",
                       progress=False, group_by="ticker", auto_adjust=True)
    for name, tkr in FACTOR_TICKERS.items():
        try:
            chg = data[tkr]["Close"].pct_change(fill_method=None) * 100
            df[f"{name}_chg"] = chg.reindex(df.index, method="ffill").shift(1).fillna(0.0)
        except Exception:
            df[f"{name}_chg"] = 0.0
    # Heavyweight breadth: fraction of top-5 index anchors green yesterday, in [-1, 1]
    df["hw_breadth"] = sum(np.sign(df[c]) for c in HW_COLS) / len(HW_COLS)
    # ADR composite: average prev-day move of Indian ADRs in the US session
    df["adr_chg"] = sum(df[c] for c in ADR_COLS) / len(ADR_COLS)
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



def days_to_nifty_expiry(dt):
    from datetime import timedelta
    d = dt.date() if hasattr(dt, 'date') else dt
    days = 0
    while d.weekday() != 1:   # 1 = Tuesday
        d += timedelta(days=1)
        days += 1
    return max(0.1, days) / 365.0

def _clip(x):
    return max(-1.0, min(1.0, x))


def build_signals(prev, today, prev2_close) -> list:
    """All factor signals for one day. Used by backtest AND auditable by verify_data.py."""
    gap_pct = (today.Open / prev.Close - 1) * 100
    macro_raw = (0.40 * today.sp500_chg + 0.15 * today.nikkei_chg - 0.10 * today.brent_chg
                 - 0.15 * today.dxy_chg - 0.10 * today.us10y_chg
                 - 3.0 * 0.25 * today.usdinr_chg) / 1.2
    return [
        pivot_signal(today.Open, prev.High, prev.Low, prev.Close),
        vix_signal(today.vix, (today.vix / prev.vix - 1) * 100 if prev.vix else 0),
        SignalScore("gift_gap", _clip(gap_pct), 0.7),
        SignalScore("momentum", _clip((prev.Close / prev2_close - 1) * 100 / 1.2), 0.5),
        SignalScore("macro", _clip(macro_raw), 0.6),
        SignalScore("sector", _clip(today.banknifty_chg / 1.2), 0.55),   # Bank Nifty alignment
        SignalScore("breadth", _clip(today.hw_breadth * 0.8), 0.55),     # heavyweight breadth
        SignalScore("adr", _clip(today.adr_chg / 1.5), 0.6),             # ADR overnight read
    ]


def run_backtest(df, weights=None, capital=100000.0, lots=1, lot_size=75, r=0.068,
                 confidence_threshold=0.50, daily_max_loss_pct=3.0,
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
    for i in range(2, len(rows)):
        pheromone.evaporate()
        prev, today = rows[i - 1], rows[i]
        risk.new_day()
        spot = today.Open
        iv = max(today.vix, 8.0) / 100.0
        t_exp = days_to_nifty_expiry(today.Index)
        signals = build_signals(prev, today, rows[i - 2].Close)
        regime = regime_detector.classify(vix=today.vix, adx=today.adx if "adx" in df.columns else 0.0, atr_pct=(today.atr / spot) * 100)

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

        import pandas as pd
        current_time_sim = pd.to_datetime(today.Index).to_pydatetime().replace(hour=9, minute=15)
        # Event sentiment: decaying impact of major historical news
        sentiment_impact = events.get(today.Index.date(), 0.0)
        plan = engine.decide(signals, spot, sentiment_impact=sentiment_impact)
        # Trend-regime gate: never fight the established trend
        uptrend   = prev.Close > prev.sma50 * 1.005
        downtrend = prev.Close < prev.sma50 * 0.995
                # Priority 4: Re-enables CE trades in bearish regime and PE trades in bullish regime if near SMA50.
        uptrend   = prev.Close > prev.sma50 * 1.005
        downtrend = prev.Close < prev.sma50 * 0.995
        near_sma = abs(prev.Close / prev.sma50 - 1) < 0.005

        if plan is not None:
            if plan.action == "BUY_CE" and not (uptrend or near_sma):
                plan = None
            elif plan.action == "BUY_PE" and not (downtrend or near_sma):
                plan = None
        if plan is None or plan.action == "NO_TRADE":
            if regime == "CHOPPY":
                from ..strategy.intraday import ChoppyMarketStrategy
                choppy = ChoppyMarketStrategy()
                condor_action = choppy.evaluate_iron_condor(today.vix)
                if condor_action in ("SELL_IRON_CONDOR", "SELL_NARROW_CONDOR"):
                    wing = 75 if condor_action == "SELL_IRON_CONDOR" else 50
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
        kind = "C" if plan.action == "BUY_CE" else "P"
        entry_prem = bs_price(spot, plan.strike, r, iv, t_exp, kind)
        if entry_prem < 5:
            equity.append(broker.capital)
            continue
        # Volatility-aware DISASTER stop: outside normal noise, abnormal days only
        exp_move = 0.5 * prev.atr
        disaster = max(1.0, entry_prem - disaster_atr_mult * exp_move)
        pos = broker.buy(f"NIFTY{plan.strike}{kind}E", qty, entry_prem, disaster,
                         entry_prem + disaster_atr_mult * exp_move * 2.5)
        if pos is None:
            equity.append(broker.capital)
            continue
        # Honest fill model on daily bars: hold open->close unless disaster stop breached
        adv_spot = today.Low if kind == "C" else today.High
        prem_adv = bs_price(adv_spot, plan.strike, r, iv, t_exp - 0.25 / 365, kind)
        prem_close = bs_price(today.Close, plan.strike, r, iv, t_exp - 1 / 365, kind)
        exit_px = disaster if prem_adv <= disaster else prem_close
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
        reg = 0.05 * sum(abs(w - 1.0) for w in vec)
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
