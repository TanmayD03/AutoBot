import math
from autobot.options_math import bs_price, greeks, implied_vol, max_pain, put_call_ratio

S, K, r, sigma, t = 100.0, 100.0, 0.05, 0.2, 1.0


def test_call_price_reference():
    assert math.isclose(bs_price(S, K, r, sigma, t, "C"), 10.4506, abs_tol=1e-3)


def test_put_price_reference():
    assert math.isclose(bs_price(S, K, r, sigma, t, "P"), 5.5735, abs_tol=1e-3)


def test_put_call_parity():
    c = bs_price(S, K, r, sigma, t, "C")
    p = bs_price(S, K, r, sigma, t, "P")
    assert math.isclose(c - p, S - K * math.exp(-r * t), abs_tol=1e-9)


def test_delta_bounds():
    g = greeks(S, K, r, sigma, t, "C")
    assert 0.6 < g.delta < 0.65
    gp = greeks(S, K, r, sigma, t, "P")
    assert math.isclose(g.delta - gp.delta, 1.0, abs_tol=1e-9)


def test_implied_vol_roundtrip():
    price = bs_price(S, K, r, 0.27, t, "C")
    assert math.isclose(implied_vol(price, S, K, r, t, "C"), 0.27, abs_tol=1e-3)


def test_max_pain_and_pcr():
    chain = [{"strike": 90, "call_oi": 10, "put_oi": 100},
             {"strike": 100, "call_oi": 50, "put_oi": 50},
             {"strike": 110, "call_oi": 100, "put_oi": 10}]
    assert max_pain(chain) == 100
    assert math.isclose(put_call_ratio(chain), 1.0, abs_tol=1e-9)
