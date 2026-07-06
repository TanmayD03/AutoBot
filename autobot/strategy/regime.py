class RegimeDetector:
    def __init__(self, vix_low=14.0, vix_high=20.0, atr_pct_choppy=1.0):
        self.vix_low = vix_low
        self.vix_high = vix_high
        # Was 0.30 — checked against 63 real trading days of daily NIFTY ATR%:
        # the MINIMUM observed value was 0.74%, making 0.30% structurally
        # unreachable. 1.0% sits within the real observed distribution.
        self.atr_pct_choppy = atr_pct_choppy

    def classify(self, vix, adx=0, atr_pct=None):
        if adx > 28:
            return "TRENDING"
        if vix < self.vix_low and (atr_pct is None or atr_pct < self.atr_pct_choppy):
            return "CHOPPY"
        elif vix > self.vix_high:
            return "HIGH_VOLATILITY"
        return "TRENDING"
