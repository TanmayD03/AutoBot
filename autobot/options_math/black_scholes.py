"""Black-Scholes engine: pricing, full Greeks, IV solver, Max Pain, PCR, GEX.
Dependency-light (math.erf, no scipy).
"""
import math
from dataclasses import dataclass

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)


def _N(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def _n(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / SQRT2PI


def _d1d2(S, K, r, sigma, t):
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    return d1, d1 - sigma * math.sqrt(t)


def bs_price(S, K, r, sigma, t, kind="C") -> float:
    if t <= 0 or sigma <= 0:
        return max(0.0, (S - K) if kind == "C" else (K - S))
    d1, d2 = _d1d2(S, K, r, sigma, t)
    if kind == "C":
        return S * _N(d1) - K * math.exp(-r * t) * _N(d2)
    return K * math.exp(-r * t) * _N(-d2) - S * _N(-d1)


@dataclass
class Greeks:
    delta: float
    gamma: float
    theta_per_day: float
    vega: float
    rho: float


def greeks(S, K, r, sigma, t, kind="C") -> Greeks:
    d1, d2 = _d1d2(S, K, r, sigma, t)
    pdf = _n(d1)
    gamma = pdf / (S * sigma * math.sqrt(t))
    vega = S * math.sqrt(t) * pdf / 100.0  # per 1 vol point
    if kind == "C":
        delta = _N(d1)
        theta = (-(S * pdf * sigma) / (2 * math.sqrt(t)) - r * K * math.exp(-r * t) * _N(d2))
        rho = K * t * math.exp(-r * t) * _N(d2) / 100.0
    else:
        delta = _N(d1) - 1.0
        theta = (-(S * pdf * sigma) / (2 * math.sqrt(t)) + r * K * math.exp(-r * t) * _N(-d2))
        rho = -K * t * math.exp(-r * t) * _N(-d2) / 100.0
    return Greeks(delta=delta, gamma=gamma, theta_per_day=theta / 365.0, vega=vega, rho=rho)


def implied_vol(price, S, K, r, t, kind="C", tol=1e-6, max_iter=100) -> float:
    """Newton-Raphson with bisection fallback."""
    sigma = 0.2
    for _ in range(max_iter):
        diff = bs_price(S, K, r, sigma, t, kind) - price
        if abs(diff) < tol:
            return sigma
        d1, _ = _d1d2(S, K, r, sigma, t)
        v = S * math.sqrt(t) * _n(d1)
        if v < 1e-10:
            break
        sigma = max(1e-4, sigma - diff / v)
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if bs_price(S, K, r, mid, t, kind) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def max_pain(chain):
    """chain: list of dicts {strike, call_oi, put_oi}. Returns strike of minimum option-writer pain."""
    strikes = [c["strike"] for c in chain]
    best, best_pain = None, float("inf")
    for s in strikes:
        pain = sum(max(0, s - c["strike"]) * c["call_oi"] + max(0, c["strike"] - s) * c["put_oi"]
                   for c in chain)
        if pain < best_pain:
            best, best_pain = s, pain
    return best


def put_call_ratio(chain) -> float:
    calls = sum(c["call_oi"] for c in chain) or 1
    return sum(c["put_oi"] for c in chain) / calls


def gamma_exposure(chain, S, r, t, iv=0.15) -> float:
    """Dealer GEX estimate: calls add positive gamma, puts negative (dealer-short-put convention)."""
    gex = 0.0
    for c in chain:
        g = greeks(S, c["strike"], r, iv, t, "C").gamma
        gex += g * c["call_oi"] * S
        gex -= g * c["put_oi"] * S
    return gex
