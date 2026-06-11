from datetime import time

class IntradayFlipper:
    def __init__(self):
        pass

    def evaluate_entry(self, current_time, spot, vwap, rsi, adx, pdh, dma20, dma50, vix, gift_nifty_gap):
        """
        Evaluate entry considering time of day and technicals.
        Enhanced to capture 9:15-9:45 AM volatility and anytime reversals using DMA.
        """
        # 1. Capture opening volatility (09:15 - 09:45)
        if current_time and getattr(current_time, 'hour', 0) == 9 and getattr(current_time, 'minute', 0) <= 45:
            # High volatility morning trades
            if vix > 14 and abs(gift_nifty_gap) > 0.3:
                if spot > vwap and gift_nifty_gap > 0:
                    return "BUY_CE"
                elif spot < vwap and gift_nifty_gap < 0:
                    return "BUY_PE"

        # 2. General Trend Entry (Anytime after morning settlement)
        if adx > 20:
            bullish = spot > dma20 and spot > vwap
            bearish = spot < dma20 and spot < vwap

            if bullish:
                return "BUY_CE"
            elif bearish:
                return "BUY_PE"

        return "NO_TRADE"

    def evaluate_exit(self, position_type, rsi, spot, pdh_or_oi_wall, pdl_or_put_wall):
        """
        Evaluate if we should exit based on momentum fading or hitting walls.
        """
        if position_type == "CE":
            if rsi > 72 or spot >= pdh_or_oi_wall:
                return "EXIT"
        elif position_type == "PE":
            if rsi < 28 or spot <= pdl_or_put_wall:
                return "EXIT"

        return "HOLD"

    def evaluate_flip(self, current_position, spot, vwap, adx, dma20):
        """
        Flip to the opposite side if the market sharply breaks VWAP & DMA with ADX confirmation.
        Works at any time.
        """
        if current_position == "CE":
            if spot < vwap and spot < dma20 and adx > 20:
                return "FLIP_TO_PE"
        elif current_position == "PE":
            if spot > vwap and spot > dma20 and adx > 20:
                return "FLIP_TO_CE"

        return "HOLD"


class ChoppyMarketStrategy:
    def evaluate_credit_spread(self, spot, range_low, range_high):
        """
        When Nifty is range-bound, sell a credit spread at the range boundaries.
        """
        if spot > range_high - 50:
            return "SELL_CALL_SPREAD"
        elif spot < range_low + 50:
            return "SELL_PUT_SPREAD"
        return "NO_TRADE"

    def evaluate_iron_condor(self, vix):
        """
        In truly compressed VIX days (<13), sell a narrow iron condor.
        """
        if vix < 13:
            return "SELL_IRON_CONDOR"
        return "NO_TRADE"

    def evaluate_vwap_scalp(self, spot, vwap, rsi):
        """
        Mean reversion entries: buy CE when Nifty drops 0.25% below VWAP with RSI < 35.
        Target = VWAP.
        """
        if spot < vwap * 0.9975 and rsi < 35:
            return "BUY_CE_SCALP"
        elif spot > vwap * 1.0025 and rsi > 65:
            return "BUY_PE_SCALP"
        return "NO_TRADE"
