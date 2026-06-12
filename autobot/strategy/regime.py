class RegimeDetector:
    """Classifies the market regime using VIX, ADX, and ATR."""

    def classify(self, vix, adx=0, atr_pct=0):
        # ADX takes priority — VIX alone is insufficient (Priority 5)
        if adx > 25: # Using 25 based on Priority 5 text "ADX>25 -> TRENDING" (even though drop-in code had 28, sticking to the textual instruction)
            return "TRENDING"
        if vix > 20 or atr_pct > 0.65:
            return "HIGH_VOLATILITY"
        if vix < 14 and atr_pct < 0.30 and adx < 18:
            return "CHOPPY"
        return "TRENDING"  # default: attempt the trade
