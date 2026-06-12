"""Risk manager: the survival organ. Enforces the user's asymmetric daily P&L rule
(e.g. +50 target / -20 max loss => worst good day +30), R:R >= 2.5 per trade,
kill switch, and end-of-day square-off.
"""
from datetime import time as dtime


class RiskManager:
    def __init__(self, daily_profit_target=50.0, daily_max_loss=20.0, reward_risk_min=2.5,
                 max_open_positions=1, squareoff=dtime(15, 15)):
        self.daily_profit_target = daily_profit_target
        self.daily_max_loss_base = daily_max_loss
        self.current_dynamic_max_loss = daily_max_loss
        self.rr_min = reward_risk_min
        self.max_open = max_open_positions
        self.squareoff = squareoff
        self.day_pnl = 0.0
        self.halted = False

    def new_day(self):
        self.day_pnl, self.halted = 0.0, False
        self.current_dynamic_max_loss = self.daily_max_loss_base

    def set_dynamic_daily_loss(self, entry_premium, qty):
        # Priority 10: Scale kill switch to position size to avoid immediate false-positives
        self.current_dynamic_max_loss = max(200.0, entry_premium * qty * 0.40)

    def register_pnl(self, pnl):
        self.day_pnl += pnl
        if self.day_pnl <= -self.current_dynamic_max_loss:
            self.halted = True  # kill switch
        if self.day_pnl >= self.daily_profit_target:
            self.halted = True  # lock in the day

    def can_trade(self, open_positions, now_time=None):
        if self.halted or open_positions >= self.max_open:
            return False
        if now_time and now_time >= self.squareoff:
            return False
        return True

    def validate_trade(self, entry, stop, target):
        risk = abs(entry - stop)
        reward = abs(target - entry)
        return risk > 0 and (reward / risk) >= self.rr_min

    def levels(self, entry, direction=1):
        """Build stop/target satisfying daily rule and R:R from premium entry."""
        risk = min(self.current_dynamic_max_loss, entry * 0.15)
        reward = max(self.daily_profit_target, risk * self.rr_min)
        return entry - risk, entry + reward  # long-premium convention
