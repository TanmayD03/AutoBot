"""AutoBot demo — works locally and on Google Colab (no broker credentials needed).
1. Loads up to 20 years of NIFTY + India VIX history.
2. Evolves signal weights with PSO using K-FOLD WALK-FORWARD fitness (anti-overfit).
3. Validates on the held-out last 30% — the only numbers that matter.
4. Plots the out-of-sample equity curve and renders a terminal snapshot (Colab-safe).
"""
from autobot.backtest.backtester import load_history, run_backtest, fitness_for_pso
from autobot.nature.pso import PSO

SIGNAL_NAMES = ["pivots", "vix", "gift_gap", "momentum", "macro",
                "sector", "breadth", "adr"]


def show(rep, title):
    print(f"\n=== {title} ===")
    for k, v in rep.items():
        if k in ("equity", "last_trades"):
            continue
        print(f"  {k}: {v}")
    for t in rep.get("last_trades", [])[-3:]:
        print(f"    {t}")


def main():
    print("Loading 20 years of NIFTY history...")
    df = load_history(20)
    split = int(len(df) * 0.7)
    train, test = df.iloc[:split], df.iloc[split:]
    print(f"Train {len(train)} days | Test {len(test)} days")

    print("Evolving signal weights with PSO (3-fold walk-forward fitness)...")
    pso = PSO(dim=len(SIGNAL_NAMES), n_particles=40)
    best, fit = pso.optimize(fitness_for_pso(train, SIGNAL_NAMES, folds=5), iterations=50)
    weights = dict(zip(SIGNAL_NAMES, best))
    print(f"Best weights: { {k: round(v, 2) for k, v in weights.items()} } (fitness {fit:.2f})")

    rep_tr = run_backtest(train, weights=weights)
    rep_te = run_backtest(test, weights=weights)
    show(rep_tr, "IN-SAMPLE (train)")
    show(rep_te, "OUT-OF-SAMPLE (test) — the only numbers that matter")

    # Equity curve plot (renders inline on Colab)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(rep_tr.get("equity", []))
        ax[0].set_title("Equity — train")
        ax[1].plot(rep_te.get("equity", []), color="darkorange")
        ax[1].set_title("Equity — out-of-sample test")
        for a in ax:
            a.set_xlabel("trading days")
            a.set_ylabel("capital (₹)")
            a.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"(plot skipped: {e})")

    # Terminal snapshot (Rich Live loops need a real terminal; snapshot works on Colab)
    try:
        from autobot.terminal.dashboard import Dashboard
        dash = Dashboard()
        dash.state["status"] = "BACKTEST SNAPSHOT (paper)"
        dash.state["trades"] = rep_te.get("last_trades", [])
        dash.state["day_pnl"] = rep_te.get("last_trades", [{}])[-1].get("pnl", 0.0) if rep_te.get("last_trades") else 0.0
        dash.snapshot()
    except Exception as e:
        print(f"(dashboard snapshot skipped: {e})")

    print("\nIf out-of-sample is negative, the system is telling you the truth: that\n"
          "configuration has no edge in the recent regime and it must NOT be traded live.\n"
          "Tune, retrain, and only ever act on positive OUT-OF-SAMPLE expectancy.\n"
          "Synthetic premiums via Black-Scholes + VIX proxy. Past results never guarantee profit.")


if __name__ == "__main__":
    main()
