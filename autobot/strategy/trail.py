class TrailManager:
    def __init__(self, partial_profit_target_pct=40.0, partial_profit_booking_pct=50.0,
                 trail_retracement_pct=20.0, hard_stop_pct=60.0, expiry_mode=False):
        if expiry_mode:
            partial_profit_target_pct = 20.0
            trail_retracement_pct = 15.0

        self.peak = None  # running peak, updated each candle
        self.expiry_mode = expiry_mode
        self.partial_trigger = partial_profit_target_pct / 100.0
        self.trail_pct = trail_retracement_pct / 100.0
        self.hard_stop_pct = hard_stop_pct / 100.0
        self.partial_profit_booking_pct = partial_profit_booking_pct / 100.0
        self.booked = False

    def evaluate(self, entry, current, peak_prem, qty, initial_qty):
        return self.update(current, entry, qty, initial_qty)

    def update(self, current, entry, qty, initial_qty):
        self.peak = max(self.peak or entry, current)
        gain = (current - entry) / entry
        if gain >= self.partial_trigger and not self.booked:
            self.booked = True
            return "PARTIAL_EXIT", qty // 2
        trail_sl = self.peak * (1 - self.trail_pct)
        if self.booked and current < trail_sl:
            return "FULL_EXIT", qty
        if gain <= -self.hard_stop_pct:
            return "STOP_LOSS", qty
        return "HOLD", 0