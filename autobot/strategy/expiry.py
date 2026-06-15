from datetime import time

class ExpiryDayStrategy:
    def __init__(self):
        pass

    def is_valid_window(self, current_time):
        """Trade only 9:15-10:45 or 14:00-15:15 windows."""
        if not current_time:
            return True # Backtest daily mode fallback

        t = current_time
        morning = time(9, 15) <= t <= time(10, 45)
        afternoon = time(14, 0) <= t <= time(15, 15)
        return morning or afternoon

    def evaluate(self, current_time, spot, max_pain):
        """Use max_pain from NSE chain as signal, prefer OTM strikes."""
        if not self.is_valid_window(current_time):
            return "NO_TRADE", 0

        # Very simple max pain reversion logic
        if spot > max_pain + 20:
            return "BUY_PE", round(spot / 50) * 50 - 50 # OTM
        elif spot < max_pain - 20:
            return "BUY_CE", round(spot / 50) * 50 + 50 # OTM

        return "NO_TRADE", 0
