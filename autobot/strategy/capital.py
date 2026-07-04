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
                                 max_loss_per_unit=None, risk_pct_override=None):
        """
        Size position by capital risk first, then optionally scale by delta —
        but NEVER let the delta adjustment override the capital cap.

        max_loss_per_unit: for spreads, pass the spread's max loss per unit
        (e.g. wing_width - net_debit) instead of using option_premium as
        the risk proxy. For naked options, leave as None (option_premium
        itself is the max loss per unit, since premium can go to zero).

        risk_pct_override: if given (0.0-1.0), use this instead of
        self.max_risk_per_trade_pct for THIS call only. Use this to size up
        in high-confidence/trending setups and down in choppy/high-vol ones —
        it varies the % of capital risked per trade rather than scaling the
        lot count after the affordability check, so the existing cap and
        affordability guards below still apply correctly.
        """
        if option_premium <= 0:
            return 0

        risk_pct = risk_pct_override if risk_pct_override is not None else self.max_risk_per_trade_pct

        # Risk-per-unit is the TRUE max loss, not the cheap net debit
        risk_per_unit = max_loss_per_unit if max_loss_per_unit is not None else option_premium
        cost_per_lot = option_premium * lot_size  # actual cash outlay to buy 1 lot

        # Hard cap: never risk more than risk_pct of capital
        max_capital_to_risk = capital * risk_pct
        max_lots = int(max_capital_to_risk // (risk_per_unit * lot_size))

        # TRUE affordability guard: only force a minimum of 1 lot if 1 lot is
        # actually payable out of total capital. Previously this used max(1, ...)
        # unconditionally, which could return 1 lot even when the account
        # couldn't afford it, forcing an order that should never be sent.
        if max_lots == 0:
            return 1 if cost_per_lot <= capital else 0
        max_lots = max(1, max_lots)

        # Delta scaling REDUCES exposure for low-delta (far OTM, low-conviction)
        # options — it should never be used to multiply lots upward.
        if abs(delta) < 0.05:
            return 1 if cost_per_lot <= capital else 0
        delta_scale = min(1.0, abs(delta) / target_delta_exposure)  # capped at 1.0, never inflates
        scaled_lots = max(1, int(max_lots * delta_scale))

        result = min(scaled_lots, max_lots)   # belt-and-suspenders: scaled can never exceed the cap
        return result if cost_per_lot <= capital else 0

    def regime_confidence_risk_pct(self, regime, confidence, base_pct=None,
                                     conf_floor=0.70, conf_ceiling=1.0):
        """
        Map regime + signal confidence to a per-trade risk % for use as
        risk_pct_override in calculate_lots_by_delta.

        - TRENDING: this is what the naked CE/PE strategy is built for.
          Scale UP with confidence, from 1.0x base at the gate (0.70) to
          1.5x base at confidence 1.0.
        - CHOPPY: directional edge is weak in a sideways market — this
          strategy isn't really designed for chop (that's what the iron
          condor is for). Scale down hard, 0.5x base, regardless of
          confidence, so a false-positive trending signal in chop can't
          size up like it would in a real trend.
        - HIGH_VOLATILITY: bigger whipsaw risk cuts both ways — scale down
          to 0.6x base even on a high-confidence signal.
        """
        base = base_pct if base_pct is not None else self.max_risk_per_trade_pct
        conf = max(conf_floor, min(conf_ceiling, confidence))
        conf_frac = (conf - conf_floor) / max(1e-9, (conf_ceiling - conf_floor))

        if regime == "TRENDING":
            mult = 1.0 + 0.5 * conf_frac       # 1.0x -> 1.5x
        elif regime == "CHOPPY":
            mult = 0.5
        elif regime == "HIGH_VOLATILITY":
            mult = 0.6
        else:
            mult = 1.0

        # Hard ceiling at 55%: PaperBroker rejects any trade whose worst-case
        # risk exceeds 60% of capital (a safety circuit breaker independent of
        # this method). Without this cap, a high-confidence TRENDING signal
        # could compute a risk_pct above that breaker and get silently
        # rejected at the broker level after already being "sized".
        return min(0.55, base * mult)

    def calculate_lots_by_max_loss(self, capital, max_loss_per_unit, lot_size=75, risk_pct_override=None):
        """
        Generic defined-risk sizer for multi-leg trades (spreads, iron condors)
        where the true worst case is max_loss_per_unit per unit — e.g.
        wing_width - net_credit for a condor, or wing_width - net_debit for a
        debit spread. Deliberately separate from calculate_lots_by_delta:
        a credit spread's net premium is cash IN, not risk, so sizing off the
        premium the way a naked option is sized would understate real risk.
        """
        if max_loss_per_unit <= 0:
            return 0
        risk_pct = risk_pct_override if risk_pct_override is not None else self.max_risk_per_trade_pct
        max_capital_to_risk = capital * risk_pct
        cost_per_lot = max_loss_per_unit * lot_size
        lots = int(max_capital_to_risk // cost_per_lot)
        if lots == 0:
            return 1 if cost_per_lot <= capital else 0
        return lots
