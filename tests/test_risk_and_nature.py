from datetime import time as dtime
from autobot.strategy.risk import RiskManager
from autobot.nature.immune import ImmuneSystem
from autobot.nature.pheromone import PheromoneMemory
from autobot.nature.pso import PSO


def test_kill_switch():
    rm = RiskManager(daily_profit_target=50, daily_max_loss=20)
    rm.register_pnl(-21)
    assert rm.halted and not rm.can_trade(0)


def test_profit_lock():
    rm = RiskManager(daily_profit_target=50, daily_max_loss=20)
    rm.register_pnl(55)
    assert rm.halted  # locks the day, ends green


def test_rr_enforcement():
    rm = RiskManager(reward_risk_min=2.5)
    assert rm.validate_trade(entry=100, stop=90, target=126)
    assert not rm.validate_trade(entry=100, stop=90, target=110)


def test_squareoff_blocks_late_entries():
    rm = RiskManager()
    assert not rm.can_trade(0, now_time=dtime(15, 20))


def test_immune_flags_flash_crash():
    im = ImmuneSystem(z_threshold=4.0)
    for _ in range(100):
        im.observe(ret_pct=0.1)
    bad, reasons = im.is_anomalous(ret_pct=-8.0)
    assert bad and reasons


def test_pheromone_shifts_to_winner():
    pm = PheromoneMemory(["trend", "meanrev"])
    for _ in range(5):
        pm.reinforce("trend", 50)
        pm.reinforce("meanrev", -20)
        pm.evaporate()
    assert pm.best() == "trend"


def test_pso_converges_on_simple_bowl():
    pso = PSO(dim=2, n_particles=15, lo=-5, hi=5)
    best, fit = pso.optimize(lambda x: -(x[0] - 1) ** 2 - (x[1] + 2) ** 2, iterations=60)
    assert abs(best[0] - 1) < 0.5 and abs(best[1] + 2) < 0.5
