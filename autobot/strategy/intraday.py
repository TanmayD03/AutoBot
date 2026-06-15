from datetime import time

class IntradayFlipper:
    def __init__(self):
        pass

    def evaluate_entry(self, current_time, spot, vwap, rsi, adx, pdh, dma20, dma50, vix, gift_nifty_gap, orb_high=None, orb_low=None):
        if current_time and getattr(current_time, 'hour', 0) == 9 and getattr(current_time, 'minute', 0) <= 45:
            if vix > 14 and abs(gift_nifty_gap) > 0.3:
                if spot > vwap and gift_nifty_gap > 0:
                    return "BUY_CE"
                elif spot < vwap and gift_nifty_gap < 0:
                    return "BUY_PE"

        # ORB breakout check (fires after 9:45 AM)
        if (current_time and getattr(current_time, 'hour', 0) >= 9
            and getattr(current_time, 'minute', 0) > 45
            and orb_high and orb_low):
            if spot > orb_high and adx > 20 and gift_nifty_gap > 0.3:
                return "BUY_CE"   # upside breakout confirmed
            elif spot < orb_low and adx > 20 and gift_nifty_gap < -0.3:
                return "BUY_PE"   # downside breakout confirmed

        if adx > 20:
            bullish = spot > dma20 and spot > vwap
            bearish = spot < dma20 and spot < vwap
            if bullish:
                return "BUY_CE"
            elif bearish:
                return "BUY_PE"
        return "NO_TRADE"

    def evaluate_exit(self, position_type, rsi, spot, pdh_or_oi_wall, pdl_or_put_wall):
        if position_type == "CE":
            if rsi > 72 or spot >= pdh_or_oi_wall:
                return "EXIT"
        elif position_type == "PE":
            if rsi < 28 or spot <= pdl_or_put_wall:
                return "EXIT"
        return "HOLD"

    def evaluate_flip(self, current_position, spot, vwap, adx, dma20):
        if current_position == "CE":
            if spot < vwap and spot < dma20 and adx > 20:
                return "FLIP_TO_PE"
        elif current_position == "PE":
            if spot > vwap and spot > dma20 and adx > 20:
                return "FLIP_TO_CE"
        return "HOLD"


class ChoppyMarketStrategy:
    def evaluate_credit_spread(self, spot, range_low, range_high):
        if spot > range_high - 50:
            return "SELL_CALL_SPREAD"
        elif spot < range_low + 50:
            return "SELL_PUT_SPREAD"
        return "NO_TRADE"

    def evaluate_iron_condor(self, vix):
        if vix < 17:          # practical threshold for current Indian market
            return "SELL_IRON_CONDOR"
        elif vix < 20:        # elevated vol — use tighter wings (50pt instead of 75pt)
            return "SELL_NARROW_CONDOR"
        return "NO_TRADE"     # VIX > 20: too much vol risk for condor

    def evaluate_vwap_scalp(self, spot, vwap, rsi):
        if spot < vwap * 0.9975 and rsi < 35:
            return "BUY_CE_SCALP"
        elif spot > vwap * 1.0025 and rsi > 65:
            return "BUY_PE_SCALP"
        return "NO_TRADE"

class SpreadStrategy:
    def bear_put_spread(self, spot, atm, wing=100):
        # Buy ATM PE, Sell (ATM-wing) PE
        return {"buy": atm, "sell": atm - wing, "kind": "P",
                "max_profit_pts": wing, "note": "capped upside, defined loss"}

    def bull_call_spread(self, spot, atm, wing=100):
        # Buy ATM CE, Sell (ATM+wing) CE
        return {"buy": atm, "sell": atm + wing, "kind": "C",
                "max_profit_pts": wing, "note": "capped upside, defined loss"}
