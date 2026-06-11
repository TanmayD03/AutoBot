"""Artificial Immune System anomaly detector (negative selection).
Learns the statistical 'self' of normal market behaviour; any live observation that no
detector recognizes as self is flagged 'non-self' (anomaly) -> trading is halted.
Protects against flash crashes, circuit moves, and data-feed corruption.
"""
import statistics


class ImmuneSystem:
    def __init__(self, z_threshold=4.0, window=500):
        self.z = z_threshold
        self.window = window
        self.history = {"ret": [], "vix": [], "spread": []}

    def observe(self, ret_pct=None, vix=None, spread=None):
        for k, v in (("ret", ret_pct), ("vix", vix), ("spread", spread)):
            if v is not None:
                self.history[k].append(v)
                self.history[k] = self.history[k][-self.window:]

    def is_anomalous(self, ret_pct=None, vix=None, spread=None):
        """Returns (anomalous: bool, reasons: list)."""
        reasons = []
        for k, v in (("ret", ret_pct), ("vix", vix), ("spread", spread)):
            h = self.history[k]
            if v is None or len(h) < 30:
                continue
            mu, sd = statistics.fmean(h), statistics.pstdev(h) or 1e-9
            if abs(v - mu) / sd > self.z:
                reasons.append(f"{k} z-score {(v - mu) / sd:.1f} exceeds {self.z}")
        return bool(reasons), reasons
