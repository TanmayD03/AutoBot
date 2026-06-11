"""Decision engine: fuses signals + sentiment + nature layer into one action:
BUY_CE, BUY_PE, or NO_TRADE — gated by confidence threshold.
"""
from dataclasses import dataclass
from ..signals.engine import combine


@dataclass
class TradePlan:
    action: str          # BUY_CE | BUY_PE | NO_TRADE
    strike: float
    confidence: float
    score: float
    reason: str


class DecisionEngine:
    def __init__(self, confidence_threshold=0.70, min_delta=0.45, weights=None,
                 immune=None, pheromone=None):
        self.threshold = confidence_threshold
        self.min_delta = min_delta
        self.weights = weights or {}
        self.immune = immune
        self.pheromone = pheromone

    def decide(self, signals, spot, sentiment_impact=0.0, strike_step=50,
               anomaly_inputs=None) -> TradePlan:
        # 1) Immune check: refuse to trade in non-self (anomalous) conditions
        if self.immune and anomaly_inputs:
            bad, reasons = self.immune.is_anomalous(**anomaly_inputs)
            if bad:
                return TradePlan("NO_TRADE", 0, 0.0, 0.0, "immune halt: " + "; ".join(reasons))
        # 2) Ensemble
        score, conf = combine(signals, self.weights)
        score = max(-1.0, min(1.0, score + 0.25 * sentiment_impact))
        if conf < self.threshold or abs(score) < 0.15:
            return TradePlan("NO_TRADE", 0, conf, score,
                             f"confidence {conf:.2f} below gate {self.threshold}")
        # 3) Direction & strike: ATM (delta ~0.5 satisfies min_delta gate)
        atm = round(spot / strike_step) * strike_step
        action = "BUY_CE" if score > 0 else "BUY_PE"
        return TradePlan(action, atm, conf, score,
                         f"score {score:+.2f} conf {conf:.2f} ATM {atm}")
