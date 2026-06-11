"""Event-driven backtester v3.
Replays NIFTY daily history through the SAME signal/decision/risk code used live.
Option premiums reconstructed with Black-Scholes using India VIX as IV proxy.

v3 changes (volatility-aware exits + honest fill model):
- v2's percent-of-capital stops produced ~13-point stops on ~180-point premiums while
  ATM premiums swing 50-80 points intraday -> stops sat inside market noise and the
  conservative stop-first fill rule stopped out 100% of trades. Structural flaw, fixed.
- Exits are now ATR-based: a DISASTER stop at 2x the expected daily premium move
  (delta ~0.5 x ATR), which only triggers on abnormal adverse days.
- Baseline fill is open-to-close hold: the only execution daily OHLC bars can model
  honestly (intraday stop/target ordering is unknowable from daily data).
- Reality check: 1 NIFTY lot (75 qty) at 100k capital implies ~2-4% effective risk per
  trade. That is the physical minimum; tighter is noise, not risk management.
"""
import math
from datetime import time as dtime
from ..options_math import bs_price
from ..signals.engine import pivot_signal, vix_signal, SignalScore
from ..strategy.decision import DecisionEngine
from ..strategy.risk import RiskManager
from ..execution.paper_broker import PaperBroker


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
    return df.dropna()


def run_backtest(df, weights=None, capital=100000.0, lots=1, lot_size=75, r=0.068,
                 confidence_threshold=0.70, daily_max_loss_pct=3.0,
                 daily_profit_target_pct=7.5, disaster_atr_mult=2.0):
    qty = lots * lot_size
    broker = PaperBroker(capital)
    risk = RiskManager(daily_profit_target=capital * daily_profit_target_pct / 100,
                       daily_max_loss=capital * daily_max_loss_pct / 100,
                       squareoff=dtime(15, 15))
    engine = DecisionEngine(confidence_threshold, weights=weights)
    trades, equity = [], []
    rows = list(df.itertuples())
    for i in range(2, len(rows)):
        prev, today = rows[i - 1], rows[i]
        risk.new_day()
        spot = today.Open
        iv = max(today.vix, 8.0) / 100.0
        t_exp = 3 / 365.0  # weekly option, ~3 days to expiry on average
        gap_pct = (today.Open / prev.Close - 1) * 100
        signals = [
            pivot_signal(spot, prev.High, prev.Low, prev.Close),
            vix_signal(today.vix, (today.vix / prev.vix - 1) * 100 if prev.vix else 0),
            SignalScore("gift_gap", max(-1, min(1, gap_pct)), 0.7),
            SignalScore("momentum", max(-1, min(1, (prev.Close / rows[i - 2].Close - 1) * 100 / 1.2)), 0.5),
        ]
        plan = engine.decide(signals, spot)
        # Trend-regime gate: never fight the established trend
        uptrend = prev.Close > prev.sma50 and prev.sma20 > prev.sma50
        downtrend = prev.Close < prev.sma50 and prev.sma20 < prev.sma50
        if plan.action == "BUY_CE" and not uptrend:
            plan = None
        elif plan.action == "BUY_PE" and not downtrend:
            plan = None
        if plan is None or plan.action == "NO_TRADE" or not risk.can_trade(len(broker.positions)):
            equity.append(broker.capital)
            continue
        kind = "C" if plan.action == "BUY_CE" else "P"
        entry_prem = bs_price(spot, plan.strike, r, iv, t_exp, kind)
        if entry_prem < 5:
            equity.append(broker.capital)
            continue
        # Volatility-aware DISASTER stop: 2x expected daily premium move (delta~0.5 x ATR).
        # Sits outside normal noise; only abnormal adverse days trigger it.
        exp_move = 0.5 * prev.atr
        disaster = max(1.0, entry_prem - disaster_atr_mult * exp_move)
        pos = broker.buy(f"NIFTY{plan.strike}{kind}E", qty, entry_prem, disaster,
                         entry_prem + disaster_atr_mult * exp_move * 2.5)
        if pos is None:
            equity.append(broker.capital)
            continue
        # Honest fill model on daily bars: hold open->close (EOD square-off).
        # Only exception: adverse extreme breaches the disaster stop.
        adv_spot = today.Low if kind == "C" else today.High
        prem_adv = bs_price(adv_spot, plan.strike, r, iv, t_exp - 0.25 / 365, kind)
        prem_close = bs_price(today.Close, plan.strike, r, iv, t_exp - 1 / 365, kind)
        exit_px = disaster if prem_adv <= disaster else prem_close
        pnl = broker.close(pos, exit_px)
        risk.register_pnl(pnl)
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


def fitness_for_pso(df, names, folds=3):
    """K-fold walk-forward fitness: weights must hold up across multiple regimes.
    fitness = mean(expectancy across folds) - std(expectancy) - 0.3*mean(drawdown).
    Any fold with too few trades disqualifies the particle.
    """
    n = len(df)
    chunks = [df.iloc[int(n * k / folds):int(n * (k + 1) / folds)] for k in range(folds)]

    def fitness(vec):
        weights = dict(zip(names, vec))
        exps, dds = [], []
        for chunk in chunks:
            rep = run_backtest(chunk, weights=weights)
            if rep.get("trades", 0) < 5:
                return -1e6
            exps.append(rep["expectancy"])
            dds.append(rep["max_drawdown_pct"])
        mu = sum(exps) / len(exps)
        sd = math.sqrt(sum((e - mu) ** 2 for e in exps) / len(exps))
        return mu - sd - 0.3 * (sum(dds) / len(dds))
    return fitness
