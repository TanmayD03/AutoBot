class CapitalManager:
    def __init__(self, max_risk_per_trade_pct=50.0, lot_size=75):
        """
        Dynamically size lots to protect capital constraint.
        Small accounts (e.g., ₹10K) require strict percentage caps (50% per Priority 2).
        """
        self.max_risk_per_trade_pct = max_risk_per_trade_pct / 100.0
        self.lot_size = lot_size

    def calculate_lots(self, capital, option_premium, is_spread=False, net_debit=0):
        """
        Calculate maximum lots based on 50% capital risk.
        If it's a spread, calculate using net debit instead of raw premium.
        Includes a min_lots=1 guard that also checks true affordability.
        """
        max_capital_to_risk = capital * self.max_risk_per_trade_pct
        cost_per_lot = (net_debit if is_spread else option_premium) * self.lot_size

        if cost_per_lot <= 0:
            return 0

        lots = int(max_capital_to_risk // cost_per_lot)

        # Priority 2: Add min_lots=1 guard that checks affordability
        if lots == 0 and cost_per_lot <= capital:
            lots = 1

        return lots
