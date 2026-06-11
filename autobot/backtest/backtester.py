"""Event-driven backtester. Replays NIFTY daily history (up to ~20y via yfinance) through
the SAME signal/decision/risk code used live. Option premiums are reconstructed with
Black-Scholes using India VIX as the IV proxy (documented approximation; plug a paid
historical option-chain vendor into data.DataAdapter for exact premiums).
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
    return df.dropna()


def run_backtest(df, weights=None, capital=100000.0, qty=75, r=0.068,
                 confidence_threshold=0.70, daily_profit_target=50.0, daily_max_loss=20.0):
    broker = PaperBroker(capital)
    risk = RiskManager(daily_profit_target, daily_max_loss, squareoff=dtime(15, 15))
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
            vix_signal(today.vix, (today.vix / rows[i - 1].vix - 1) * 100 if rows[i - 1].vix else 0),
            SignalScore("gift_gap", max(-1, min(1, gap_pct)), 0.7),
            SignalScore("momentum", max(-1, min(1, (prev.Close / rows[i - 2].Close - 1) * 100 / 1.2)), 0.5),
        ]
        plan = engine.decide(signals, spot)
        if plan.action == "NO_TRADE" or not risk.can_trade(len(broker.positions)):
            equity.append(broker.capital)
            continue
        kind = "C" if plan.action == "BUY_CE" else "P"
        entry_prem = bs_price(spot, plan.strike, r, iv, t_exp, kind)
        if entry_prem < 5:
            continue
        stop, target = risk.levels(entry_prem)
        pos = broker.buy(f"NIFTY{plan.strike}{kind}E", qty, entry_prem, stop, target)
        if pos is None:
            continue
        # Intraday path approximation: premium at day's favourable/adverse extremes, then close
        fav_spot = today.High if kind == "C" else today.Low
        adv_spot = today.Low if kind == "C" else today.High
        prem_fav = bs_price(fav_spot, plan.strike, r, iv, t_exp - 0.25 / 365, kind)
        prem_adv = bs_price(adv_spot, plan.strike, r, iv, t_exp - 0.25 / 365, kind)
        prem_close = bs_price(today.Close, plan.strike, r, iv, t_exp - 1 / 365, kind)
        if prem_adv <= pos.stop:
            exit_px = pos.stop          # stop hit (conservative: stop checked first)
        elif prem_fav >= pos.target:
            exit_px = pos.target        # target hit
        else:
            exit_px = prem_close        # EOD square-off
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
        return {"trades": 0, "note": "no trades passed the confidence gate"}
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
    return {
        "trades": len(trades), "win_rate": round(len(wins) / len(trades) * 100, 1),
        "total_pnl": round(sum(pnls), 2), "profit_factor": round(gross_w / gross_l, 2),
        "expectancy": round(mu, 2), "sharpe_like": round(mu / sd * math.sqrt(252), 2),
        "max_drawdown_pct": round(max_dd, 2), "last_trades": trades[-5:],
    }


def fitness_for_pso(df, names):
    """Fitness closure for PSO weight evolution: expectancy minus drawdown penalty."""
    def fitness(vec):
        weights = dict(zip(names, vec))
        rep = run_backtest(df, weights=weights)
        if rep.get("trades", 0) < 10:
            return -1e6
        return rep["expectancy"] - 0.5 * rep["max_drawdown_pct"]
    return fitness
