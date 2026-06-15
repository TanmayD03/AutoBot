class RegimeDetector:
    def __init__(self, vix_low=14.0, vix_high=20.0):
        self.vix_low = vix_low
        self.vix_high = vix_high

    def classify(self, vix, adx=0, atr_pct=None):
        if adx > 28:
            return "TRENDING"
        if vix < self.vix_low and (atr_pct is None or atr_pct < 0.30):
            return "CHOPPY"
        elif vix > self.vix_high:
            return "HIGH_VOLATILITY"
        return "TRENDING"
