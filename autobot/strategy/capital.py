class CapitalManager:
    def __init__(self, max_risk_per_trade_pct=30.0, lot_size=75):
        self.max_risk_per_trade_pct = max_risk_per_trade_pct / 100.0
        self.lot_size = lot_size

    def calculate_lots(self, capital, option_premium):
        max_capital_to_risk = capital * self.max_risk_per_trade_pct
        cost_per_lot = option_premium * self.lot_size

        if cost_per_lot <= 0:
            return 0

        lots = int(max_capital_to_risk // cost_per_lot)
        return max(0, lots)
