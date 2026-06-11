class IntradayFlipper:
    def __init__(self):
        pass

    def evaluate_9_15_to_10_00(self, spot, vwap, rsi, pdh):
        # Gap scan + VIX check. GIFT Nifty bias read
        pass

    def evaluate_10_00_to_11_30(self, spot, bullish, adx):
        # If bullish + ADX > 20 -> buy CE. Trail SL every 10 move up.
        if bullish and adx > 20:
            return "BUY_CE"
        return "NO_TRADE"

    def evaluate_exit(self, rsi, price, pdh_or_oi_wall):
        # Exit CE: Target hit, or RSI > 72 on 15min, or reversal candle at PDH/OI wall
        if rsi > 72 or price >= pdh_or_oi_wall:
            return "EXIT"
        return "HOLD"

    def evaluate_flip(self, price, vwap, adx):
        # Flip to PE: CE profit locked. If Nifty breaks VWAP + ADX confirms -> buy PE
        if price < vwap and adx > 20:
            return "BUY_PE"
        return "NO_TRADE"

class ChoppyMarketStrategy:
    def evaluate_credit_spread(self, spot, range_low, range_high):
        # Sell a credit spread at the range boundaries
        if spot > range_high - 50:
            return "SELL_CALL_SPREAD"
        elif spot < range_low + 50:
            return "SELL_PUT_SPREAD"
        return "NO_TRADE"

    def evaluate_iron_condor(self, vix):
        if vix < 13:
            return "SELL_IRON_CONDOR"
        return "NO_TRADE"

    def evaluate_vwap_scalp(self, spot, vwap, rsi):
        if spot < vwap * 0.9975 and rsi < 35:
            return "BUY_CE_SCALP"
        return "NO_TRADE"
