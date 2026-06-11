"""Signal engine. Every signal returns SignalScore(score in [-1,+1], confidence in [0,1]).
Score > 0 = bullish (favours CE), score < 0 = bearish (favours PE).
"""
from dataclasses import dataclass


@dataclass
class SignalScore:
    name: str
    score: float       # -1 .. +1
    confidence: float  # 0 .. 1


def _clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def macro_bias(macro: dict) -> SignalScore:
    """Overnight US, Asia, ADRs, Brent (inverse), DXY (inverse), US10Y (inverse)."""
    w = {"sp500": 0.20, "nasdaq": 0.15, "nikkei": 0.10, "kospi": 0.05,
         "adr_infy": 0.10, "adr_hdb": 0.10, "adr_ibn": 0.05,
         "brent": -0.10, "dxy": -0.10, "us10y": -0.05, "usdinr": -0.10}
    score = sum(wt * macro.get(k, {}).get("chg_pct", 0.0) for k, wt in w.items())
    score = _clip(score / 1.5)
    conf = min(1.0, abs(score) + 0.3)
    return SignalScore("macro_bias", score, conf)


def gap_signal(gift_gap_pct: float) -> SignalScore:
    """GIFT Nifty implied opening gap %. Strong gaps fade in confidence (gap-fill risk)."""
    score = _clip(gift_gap_pct / 1.0)
    conf = 0.8 if abs(gift_gap_pct) < 0.8 else 0.5
    return SignalScore("gift_gap", score, conf)


def pivot_signal(price, pdh, pdl, pdc) -> SignalScore:
    """PDH/PDL/PDC tripwires."""
    if price > pdh:
        return SignalScore("pivots", 0.8, 0.7)
    if price < pdl:
        return SignalScore("pivots", -0.8, 0.7)
    rng = max(pdh - pdl, 1e-9)
    return SignalScore("pivots", _clip(2 * (price - pdc) / rng * 0.5), 0.5)


def pcr_signal(pcr: float) -> SignalScore:
    """PCR regime: ~1 neutral, <0.8 bearish, <=0.6 contrarian bullish (short-cover spring)."""
    if pcr <= 0.6:
        return SignalScore("pcr", 0.6, 0.6)   # contrarian
    if pcr < 0.8:
        return SignalScore("pcr", -0.5, 0.6)
    if pcr > 1.3:
        return SignalScore("pcr", -0.4, 0.5)  # contrarian overbought
    return SignalScore("pcr", _clip((pcr - 1.0) * 1.5), 0.5)


def vix_signal(vix: float, vix_chg_pct: float) -> SignalScore:
    """Rising VIX = bearish bias + Vega expansion; falling VIX warns IV crush for buyers."""
    score = _clip(-vix_chg_pct / 8.0)
    conf = 0.7 if abs(vix_chg_pct) > 4 else 0.4
    return SignalScore("vix", score, conf)


def oi_walls_signal(chain, spot) -> SignalScore:
    """Max Call OI = ceiling, Max Put OI = floor. Position of spot between walls."""
    if not chain:
        return SignalScore("oi_walls", 0.0, 0.0)
    ceiling = max(chain, key=lambda c: c["call_oi"])["strike"]
    floor = max(chain, key=lambda c: c["put_oi"])["strike"]
    if ceiling <= floor:
        return SignalScore("oi_walls", 0.0, 0.3)
    pos = (spot - floor) / (ceiling - floor)        # 0 at floor, 1 at ceiling
    score = _clip((0.5 - pos) * 2 * 0.7)            # near floor -> bullish bounce bias
    return SignalScore("oi_walls", score, 0.65)


def combine(signals, weights=None):
    """Confidence-weighted ensemble -> (score, confidence). Weights evolved by PSO."""
    if not signals:
        return 0.0, 0.0
    weights = weights or {}
    num = den = 0.0
    for s in signals:
        w = weights.get(s.name, 1.0) * s.confidence
        num += w * s.score
        den += w
    score = num / den if den else 0.0
    agree = sum(1 for s in signals if s.score * score > 0) / len(signals)
    conf = min(1.0, abs(score) * 0.5 + agree * 0.5)
    return _clip(score), conf
