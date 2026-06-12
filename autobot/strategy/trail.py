class TrailManager:
    def __init__(self, expiry_mode=False):
        self.peak = None  # running peak, updated each candle
        self.expiry_mode = expiry_mode
        self.partial_trigger = 0.20 if expiry_mode else 0.40
        self.trail_pct = 0.15 if expiry_mode else 0.20
        self.booked = False

    def update(self, current, entry, qty, initial_qty):
        self.peak = max(self.peak or entry, current)
        gain = (current - entry) / entry

        if gain >= self.partial_trigger and not self.booked:
            self.booked = True
            return "PARTIAL_EXIT", qty // 2

        trail_sl = self.peak * (1 - self.trail_pct)
        if self.booked and current < trail_sl:
            return "FULL_EXIT", qty

        if gain <= -0.60: # 60% hard stop from the original spec
            return "STOP_LOSS", qty

        return "HOLD", 0
