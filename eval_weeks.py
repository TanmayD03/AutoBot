import pandas as pd
from autobot.backtest.backtester import load_history, run_backtest
import autobot.backtest.backtester as bt

df = load_history(5)
original_report = bt.report

def my_report(trades, equity, capital):
    result = original_report(trades, equity, capital)
    result["all_trades"] = trades
    return result

bt.report = my_report

rep = bt.run_backtest(df.tail(250))
trades = rep.get("all_trades", [])

print(f"Total trades in period: {len(trades)}")

may_week = []
june_week = []
june_week2 = []

for t in trades:
    date_str = t["date"]
    if "2026-05-18" <= date_str <= "2026-05-22":
        may_week.append(t)
    elif "2026-06-15" <= date_str <= "2026-06-19":
        june_week.append(t)
    elif "2026-06-22" <= date_str <= "2026-06-26":
        june_week2.append(t)

def summarize_week(name, week_trades):
    print(f"\n--- {name} ---")
    if not week_trades:
        print("No trades found.")
        return
    total_pnl = sum(t["pnl"] for t in week_trades)
    wins = sum(1 for t in week_trades if t["pnl"] > 0)
    for t in week_trades:
        print(f"Date: {t['date']} | Action: {t['action']} | Strike: {t['strike']} | PnL: {t['pnl']:.2f} | Conf: {t['confidence']}")
    print(f"Total Trades: {len(week_trades)} | Wins: {wins} | Win Rate: {wins/len(week_trades)*100:.1f}% | Total PnL: {total_pnl:.2f}")

summarize_week("Week: May 18 - May 22, 2026", may_week)
summarize_week("Week: June 15 - June 19, 2026", june_week)
summarize_week("Week: June 22 - June 26, 2026", june_week2)
