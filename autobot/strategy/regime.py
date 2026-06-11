class RegimeDetector:
    """Classifies the market regime using VIX and ATR."""

    def __init__(self, vix_low=14.0, vix_high=20.0):
        self.vix_low = vix_low
        self.vix_high = vix_high

    def classify(self, vix, atr_pct=None):
        if vix < self.vix_low:
            return "CHOPPY"
        elif vix > self.vix_high:
            return "HIGH_VOLATILITY"
        else:
            return "TRENDING"
