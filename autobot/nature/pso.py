"""Particle Swarm Optimization (bird-flocking) — evolves signal weights against backtest fitness.
Each particle is a candidate weight vector; the swarm converges on weights that maximize
backtest expectancy while penalizing drawdown (fitness = expectancy - 0.5*max_dd).
"""
import random


class PSO:
    def __init__(self, dim, n_particles=20, w=0.72, c1=1.49, c2=1.49, lo=0.0, hi=3.0):
        self.dim, self.lo, self.hi = dim, lo, hi
        self.w, self.c1, self.c2 = w, c1, c2
        self.X = [[random.uniform(lo, hi) for _ in range(dim)] for _ in range(n_particles)]
        self.V = [[0.0] * dim for _ in range(n_particles)]
        self.pbest = [x[:] for x in self.X]
        self.pbest_f = [-1e18] * n_particles
        self.gbest, self.gbest_f = self.X[0][:], -1e18

    def step(self, fitness_fn):
        for i, x in enumerate(self.X):
            f = fitness_fn(x)
            if f > self.pbest_f[i]:
                self.pbest_f[i], self.pbest[i] = f, x[:]
            if f > self.gbest_f:
                self.gbest_f, self.gbest = f, x[:]
        for i in range(len(self.X)):
            for d in range(self.dim):
                r1, r2 = random.random(), random.random()
                self.V[i][d] = (self.w * self.V[i][d]
                                + self.c1 * r1 * (self.pbest[i][d] - self.X[i][d])
                                + self.c2 * r2 * (self.gbest[d] - self.X[i][d]))
                self.X[i][d] = min(self.hi, max(self.lo, self.X[i][d] + self.V[i][d]))
        return self.gbest, self.gbest_f

    def optimize(self, fitness_fn, iterations=30):
        for _ in range(iterations):
            self.step(fitness_fn)
        return self.gbest, self.gbest_f
