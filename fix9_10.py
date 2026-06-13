import re

with open("autobot/backtest/backtester.py", "r") as f:
    content = f.read()

# Fix 9 & 10: Choppy condor and morning gap
# First, let's inject ChoppyMarketStrategy integration
choppy_integration = """        if plan is None or plan.action == "NO_TRADE":
            if regime == "CHOPPY":
                from ..strategy.intraday import ChoppyMarketStrategy
                choppy = ChoppyMarketStrategy()
                condor_action = choppy.evaluate_iron_condor(today.vix)
                if condor_action in ("SELL_IRON_CONDOR", "SELL_NARROW_CONDOR"):
                    wing = 75 if condor_action == "SELL_IRON_CONDOR" else 50
                    atm = round(spot / 50) * 50
                    span_margin = wing * lot_size * 1.5
                    condor_lots = max(1, int(broker.capital * 0.50 / span_margin))
                    sell_p = bs_price(spot, atm - wing//2, r, iv, t_exp, "P")
                    buy_p  = bs_price(spot, atm - wing,    r, iv, t_exp, "P")
                    sell_c = bs_price(spot, atm + wing//2, r, iv, t_exp, "C")
                    buy_c  = bs_price(spot, atm + wing,    r, iv, t_exp, "C")
                    credit = (sell_p - buy_p + sell_c - buy_c) * condor_lots * lot_size
                    stayed_in = abs(today.Close - spot) < wing
                    condor_pnl = credit if stayed_in else -wing * condor_lots * lot_size * 0.5
                    risk.register_pnl(condor_pnl)
                    trades.append({"date": str(today.Index.date()), "action": condor_action,
                                   "strike": atm, "entry": round(sell_p, 2),
                                   "exit": round(buy_p, 2), "pnl": round(condor_pnl, 2),
                                   "confidence": 0.60})
                    pheromone.reinforce("choppy_condor", condor_pnl)
            equity.append(broker.capital)
            continue"""

content = content.replace("""        if plan is None or plan.action == "NO_TRADE" or not risk.can_trade(len(broker.positions)):
            equity.append(broker.capital)
            continue""", choppy_integration)

with open("autobot/backtest/backtester.py", "w") as f:
    f.write(content)

with open("autobot/strategy/intraday.py", "r") as f:
    content2 = f.read()

# Fix the iron condor thresholds
vix_condor = """    def evaluate_iron_condor(self, vix):
        if vix < 17:          # practical threshold for current Indian market
            return "SELL_IRON_CONDOR"
        elif vix < 20:        # elevated vol — use tighter wings (50pt instead of 75pt)
            return "SELL_NARROW_CONDOR"
        return "NO_TRADE"     # VIX > 20: too much vol risk for condor"""

content2 = re.sub(r'    def evaluate_iron_condor\(self, vix\):.*?        return "NO_TRADE"', vix_condor, content2, flags=re.DOTALL)

with open("autobot/strategy/intraday.py", "w") as f:
    f.write(content2)
