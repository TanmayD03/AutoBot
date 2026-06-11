"""Ant-colony pheromone memory for adaptive strategy selection.
Each strategy lays 'pheromone' proportional to its realized P&L; pheromone evaporates
every cycle, so the system continuously shifts capital toward what is working NOW and
forgets stale edges — regime tracking, like ant trails to food.
"""


class PheromoneMemory:
    def __init__(self, strategies, evaporation=0.9, floor=0.05):
        self.evaporation = evaporation
        self.floor = floor
        self.trails = {s: 1.0 for s in strategies}

    def reinforce(self, strategy, pnl):
        self.trails[strategy] = max(self.floor, self.trails.get(strategy, 1.0) + max(-0.5, pnl / 100.0))

    def evaporate(self):
        for s in self.trails:
            self.trails[s] = max(self.floor, self.trails[s] * self.evaporation)

    def weights(self):
        total = sum(self.trails.values()) or 1.0
        return {s: v / total for s, v in self.trails.items()}

    def best(self):
        return max(self.trails, key=self.trails.get)
