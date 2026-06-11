from autobot.execution.paper_broker import PaperBroker
broker = PaperBroker(100000.0)
pos = broker.buy("NIFTY24000CE", 75, 100.0, 50.0, 150.0)
buy_cost = 75 * 100.0 * 1.0005
buy_charges = broker.charges(buy_cost)

sell_proceeds = 75 * 120.0 * 0.9995
sell_charges = broker.charges(sell_proceeds, True)

print(f"Buy cost (with slippage): {buy_cost}")
print(f"Buy charges: {buy_charges}")
print(f"Sell proceeds (with slippage): {sell_proceeds}")
print(f"Sell charges: {sell_charges}")

# Pnl in broker.close:
# px = price * (1 - self.SLIPPAGE_PCT) # 120 * 0.9995 = 119.94
# pnl = (px - pos.entry) * pos.qty - ch
#     = (119.94 - 100.05) * 75 - sell_charges
#     = 19.89 * 75 - sell_charges
#     = 1491.75 - sell_charges

# Actual capital change:
# proceeds - sell_charges - cost - buy_charges
# (119.94 * 75) - sell_charges - (100.05 * 75) - buy_charges
# 1491.75 - sell_charges - buy_charges

# Therefore, pnl calculated in close() doesn't include buy_charges!
