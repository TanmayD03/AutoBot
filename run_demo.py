"""AutoBot demo — works locally and on Google Colab (no broker credentials needed).
1. Loads up to 20 years of NIFTY + India VIX history.
2. Evolves signal weights with PSO (bird-flock swarm) on the first 70% (train).
3. Walk-forward validates on the last 30% (test) — honest out-of-sample check.
4. Prints the full backtest report.
"""
from autobot.backtest.backtester import load_history, run_backtest, fitness_for_pso
from autobot.nature.pso import PSO

SIGNAL_NAMES = ["pivots", "vix", "gift_gap", "momentum"]


def main():
    print("Loading 20 years of NIFTY history...")
    df = load_history(20)
    split = int(len(df) * 0.7)
    train, test = df.iloc[:split], df.iloc[split:]
    print(f"Train {len(train)} days | Test {len(test)} days")

    print("Evolving signal weights with Particle Swarm Optimization...")
    pso = PSO(dim=len(SIGNAL_NAMES), n_particles=12)
    best, fit = pso.optimize(fitness_for_pso(train, SIGNAL_NAMES), iterations=10)
    weights = dict(zip(SIGNAL_NAMES, best))
    print(f"Best weights: { {k: round(v, 2) for k, v in weights.items()} } (fitness {fit:.2f})")

    print("\n=== IN-SAMPLE (train) ===")
    for k, v in run_backtest(train, weights=weights).items():
        print(f"  {k}: {v}")
    print("\n=== OUT-OF-SAMPLE (test) — the only numbers that matter ===")
    for k, v in run_backtest(test, weights=weights).items():
        print(f"  {k}: {v}")
    print("\nNOTE: synthetic premiums via Black-Scholes + VIX proxy. Paper-trade live\n"
          "for >= 1 month before considering live mode. Past results never guarantee profit.")


if __name__ == "__main__":
    main()
