class TrailManager:
    def __init__(self, partial_profit_target_pct=40.0, partial_profit_booking_pct=50.0,
                 trail_retracement_pct=20.0, hard_stop_pct=60.0):
        self.partial_profit_target_pct = partial_profit_target_pct / 100.0
        self.partial_profit_booking_pct = partial_profit_booking_pct / 100.0
        self.trail_retracement_pct = trail_retracement_pct / 100.0
        self.hard_stop_pct = hard_stop_pct / 100.0

    def evaluate(self, entry_price, current_price, peak_price, qty_remaining, initial_qty):
        """
        Returns action ('HOLD', 'PARTIAL_EXIT', 'FULL_EXIT', 'STOP_LOSS')
        and the quantity to exit.
        """
        gain_pct = (current_price - entry_price) / entry_price
        peak_gain_pct = (peak_price - entry_price) / entry_price
        drawdown_from_peak = (peak_price - current_price) / peak_price

        # Hard Stop (60%)
        if gain_pct <= -self.hard_stop_pct:
            return "STOP_LOSS", qty_remaining

        # Partial Profit Booking (Book 50% at 40% gain)
        if peak_gain_pct >= self.partial_profit_target_pct and qty_remaining == initial_qty:
            qty_to_exit = int(initial_qty * self.partial_profit_booking_pct)
            return "PARTIAL_EXIT", qty_to_exit

        # Trailing Stop (20% retracement from peak, but only if we've taken partials or are well in profit)
        if qty_remaining < initial_qty and drawdown_from_peak >= self.trail_retracement_pct:
            return "FULL_EXIT", qty_remaining

        return "HOLD", 0
