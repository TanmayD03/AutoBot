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

    def calculate_lots_by_delta(self, capital, option_premium, delta,
                                 target_delta_exposure=0.50, lot_size=75,
                                 max_loss_per_unit=None):
        """
        Size position by capital risk first, then optionally scale by delta —
        but NEVER let the delta adjustment override the capital cap.

        max_loss_per_unit: for spreads, pass the spread's max loss per unit
        (e.g. wing_width - net_debit) instead of using option_premium as
        the risk proxy. For naked options, leave as None (option_premium
        itself is the max loss per unit, since premium can go to zero).
        """
        if option_premium <= 0:
            return 0

        # Risk-per-unit is the TRUE max loss, not the cheap net debit
        risk_per_unit = max_loss_per_unit if max_loss_per_unit is not None else option_premium

        # Hard cap: never risk more than max_risk_per_trade_pct of capital
        max_capital_to_risk = capital * self.max_risk_per_trade_pct
        max_lots = max(1, int(max_capital_to_risk // (risk_per_unit * lot_size)))

        # Delta scaling REDUCES exposure for low-delta (far OTM, low-conviction)
        # options — it should never be used to multiply lots upward.
        if abs(delta) < 0.05:
            return 1
        delta_scale = min(1.0, abs(delta) / target_delta_exposure)  # capped at 1.0, never inflates
        scaled_lots = max(1, int(max_lots * delta_scale))

        return min(scaled_lots, max_lots)   # belt-and-suspenders: scaled can never exceed the cap
