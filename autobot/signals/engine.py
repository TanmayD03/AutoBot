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


def crude_signal(brent_price: float, brent_chg_pct: float) -> SignalScore:
    """
    Two components:
    1. Price level: above $80 = bearish for India, below $70 = bullish
    2. Daily change: amplifies the level signal
    """
    # Level component: normalize $50–$100 range to [-1, +1]
    level_score = _clip((75.0 - brent_price) / 25.0)   # $75 = neutral
    # Change component: -4% crash = very bullish for India
    change_score = _clip(-brent_chg_pct / 5.0)
    # Combined: level sets context, change gives the day's signal
    score = 0.20 * level_score + 0.80 * change_score
    conf  = 0.70 if abs(brent_chg_pct) > 2.0 else 0.50
    return SignalScore("crude_regime", _clip(score), conf)

def fii_flow_signal(fii_net_cr: float, dii_net_cr: float) -> SignalScore:
    """
    FII/DII combined flow. ₹3000cr+ net buy = strong bullish.
    FII dominates (0.70 weight), DII is counter-cyclical (0.30).
    """
    combined = fii_net_cr * 0.70 + dii_net_cr * 0.30
    # Normalize: ±5000 crore = ±1.0 signal
    score = _clip(combined / 5000.0)
    conf  = min(0.90, 0.50 + abs(combined) / 10000.0)
    return SignalScore("fii_flow", score, conf)


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
    vix_level_factor = min(1.0, vix / 18.0)
    score = _clip(-vix_chg_pct / 8.0 * vix_level_factor)
    # LOW-VIX NOISE: when VIX is below 16, a rise is usually reversion not fear
    if vix < 16.0 and vix_chg_pct > 0:
        conf = min(0.35, 0.40)   # cap at 0.35 — counts as low-weight noise
    elif abs(vix_chg_pct) > 4 and vix > 15:
        conf = 0.70
    elif abs(vix_chg_pct) > 4:
        conf = 0.55
    else:
        conf = 0.40
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

    # Only count signals that have a meaningful opinion (|score| > 0.10)
    active = [s for s in signals if abs(s.score) > 0.10]
    if active:
        # Give double weight to high-confidence agreeing signals
        weighted_agree = sum(s.confidence for s in active if s.score * score > 0)
        weighted_total = sum(s.confidence for s in active)
        agree = weighted_agree / weighted_total if weighted_total else 0.5
    else:
        agree = 0.5  # no active signals = uncertain

    conf = min(1.0, abs(score) * 0.5 + agree * 0.5)
    return _clip(score), conf

def iv_skew_signal(iv_pe, iv_ce):
    """
    Positive skew (PE IV > CE IV) = bearish bias.
    Negative skew (CE IV > PE IV) = unusual bullish pressure.
    """
    skew = iv_pe - iv_ce
    score = _clip(-skew / 0.05)  # 5% skew = maximum signal
    return SignalScore("iv_skew", score, 0.65)

def bollinger_signal(pctb: float) -> SignalScore:
    """
    Bollinger %B: 0 = price at lower band, 0.5 = at the mean, 1 = at upper band.
    Mean-reversion framing: pinned at a band edge tends to snap back toward
    the mean, so score is the INVERSE of the extremity (contrarian).
    """
    extremity = _clip((pctb - 0.5) * 2)          # -1 at lower band, +1 at upper band
    score = _clip(-extremity * 0.6)               # contrarian: near upper band -> bearish lean
    conf = 0.60 if abs(extremity) > 0.6 else 0.35  # only confident when actually near a band
    return SignalScore("bollinger", score, conf)


def stochastic_signal(k: float, d: float) -> SignalScore:
    """
    Stochastic Oscillator (14,3): overbought >80, oversold <20 — contrarian
    framing, same spirit as Bollinger. %K crossing above %D adds a small
    momentum kicker in the same direction as the contrarian call.
    """
    if k >= 80:
        base = -0.6
    elif k <= 20:
        base = 0.6
    else:
        base = _clip((50 - k) / 40.0 * 0.4)   # mild pull toward mean elsewhere
    cross_kicker = 0.15 if (k > d and base > 0) or (k < d and base < 0) else 0.0
    score = _clip(base + cross_kicker)
    conf = 0.55 if (k >= 80 or k <= 20) else 0.30
    return SignalScore("stochastic", score, conf)


def ma_trend_signal(price: float, ma_fast: float, ma_mid: float, ma_slow: float) -> SignalScore:
    """
    Trend-alignment across three moving averages (e.g. 20/50/200 SMA or EMA).
    Fully bullish stack (price > fast > mid > slow) or fully bearish stack
    score at the extremes; a tangled/crossed stack scores near zero with low
    confidence, since that's a genuinely undecided/choppy setup.
    """
    bull_stack = price > ma_fast > ma_mid > ma_slow
    bear_stack = price < ma_fast < ma_mid < ma_slow
    if bull_stack:
        return SignalScore("ma_trend", 0.7, 0.65)
    if bear_stack:
        return SignalScore("ma_trend", -0.7, 0.65)
    # Partial alignment: score by how many of the 3 pairwise comparisons agree
    comparisons = [price > ma_fast, ma_fast > ma_mid, ma_mid > ma_slow]
    agree_bull = sum(comparisons)
    score = _clip((agree_bull - 1.5) / 1.5 * 0.5)  # -0.5..+0.5 for partial stacks
    return SignalScore("ma_trend", score, 0.30)


def max_pain_signal(spot, max_pain_strike):
    """Distance from max pain normalized to [-1, +1]."""
    distance_pct = (spot - max_pain_strike) / spot * 100
    # Only enforce max pain gravity near expiry (within 50pts)
    if abs(spot - max_pain_strike) > 200:
        return SignalScore("max_pain", 0.0, 0.0)  # too far to matter
    return SignalScore("max_pain", _clip(-distance_pct / 1.5), 0.65)
