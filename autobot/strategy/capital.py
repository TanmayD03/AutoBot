class CapitalManager:
    def __init__(self, max_risk_per_trade_pct=30.0, lot_size=75):
        """
        Dynamically size lots to protect capital constraint.
        Small accounts (e.g., ₹10K) require strict percentage caps.
        """
        self.max_risk_per_trade_pct = max_risk_per_trade_pct / 100.0
        self.lot_size = lot_size

    def calculate_lots(self, capital, option_premium, is_spread=False, net_debit=0):
        """
        Calculate maximum lots based on 30% capital risk.
        If it's a spread, calculate using net debit instead of raw premium.
        """
        max_capital_to_risk = capital * self.max_risk_per_trade_pct

        cost_per_lot = (net_debit if is_spread else option_premium) * self.lot_size

        if cost_per_lot <= 0:
            return 0

        lots = int(max_capital_to_risk // cost_per_lot)
        return max(0, lots)
