KNOWN_RISK_ON_EVENTS = {
    # Format: "YYYY-MM-DD": score (positive = risk-on = bullish)
    "2026-06-12": +0.75,   # Trump ceasefire announcement
    # Add: RBI rate cuts (+0.50), US Fed pivot (+0.60), etc.
}

KNOWN_RISK_OFF_EVENTS = {
    "2026-06-08": -0.60,   # US NFP shock + West Asia escalation
}

RECURRING_PATTERNS = {
    # Every month: RBI MPC day (high vol, direction depends on decision)
    "rbi_mpc": {"days_ahead": 0, "impact": 0.0, "conf": 0.80},
    # Every quarter: Nifty rebalancing (mild bullish for additions)
    "index_rebal": {"days_ahead": -2, "impact": +0.15, "conf": 0.60},
}

def get_event_score(date_str: str) -> tuple:
    """Returns (score, confidence) for a given date."""
    score = KNOWN_RISK_ON_EVENTS.get(date_str, 0.0)
    score += KNOWN_RISK_OFF_EVENTS.get(date_str, 0.0)
    conf  = 0.80 if abs(score) > 0.3 else 0.0
    return score, conf
