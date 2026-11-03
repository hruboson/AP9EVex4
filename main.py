from __future__ import annotations

import math
import random
import statistics

import matplotlib.pyplot as plt

RUNS_NO = 10
POPULATION_SIZE = 50
EVALS_PER_DIM = 1000 # MAXFES = EVALS_PER_DIM * dimension
DIMENSIONS = (10, 30)
VMAX_RATIO = 0.2 # vmax = 0.2 * Range

SEED = 42

# objective functions (minimization), each with its search range (lower, upper)
def sphere(x: list[float]) -> float:
    return sum(v * v for v in x)

def rastrigin(x: list[float]) -> float:
    return 10 * len(x) + sum(v * v - 10 * math.cos(2 * math.pi * v) for v in x)

def rosenbrock(x: list[float]) -> float:
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))

def ackley(x: list[float]) -> float:
    d = len(x)
    s1 = sum(v * v for v in x) / d
    s2 = sum(math.cos(2 * math.pi * v) for v in x) / d
    return -20 * math.exp(-0.2 * math.sqrt(s1)) - math.exp(s2) + 20 + math.e

OBJECTIVES = {
    "Sphere": (sphere, (-5.12, 5.12)),
    "Rastrigin": (rastrigin, (-5.12, 5.12)),
    "Rosenbrock": (rosenbrock, (-5.0, 10.0)),
    "Ackley": (ackley, (-32.768, 32.768)),
}

# PSO variants
# inertia: ("linear", w_start, w_end) or ("constant", w)
# topology: "global" (star) or "ring" with the given neighborhood size
VARIANTS = {
    "Linear w 0.8->0.3, global": dict(inertia=("linear", 0.8, 0.3), c1=2.0, c2=2.0,
                                      topology="global", neighborhood=None),
    "Constant w 0.7, global": dict(inertia=("constant", 0.7), c1=1.49618, c2=1.49618,
                                   topology="global", neighborhood=None),
    "Constant w 0.7, ring (3)": dict(inertia=("constant", 0.7), c1=1.49618, c2=1.49618,
                                     topology="ring", neighborhood=3),
}

class Particle:
    def __init__(self, dimension: int, bounds: tuple[float, float], vmax: float) -> None:
        low, high = bounds
        self.position = [random.uniform(low, high) for _ in range(dimension)]
        self.velocity = [random.uniform(-vmax, vmax) for _ in range(dimension)]
        self.fitness = math.inf # fitness of the current position
        self.best_position = self.position[:] # personal best
        self.best_fitness = math.inf

    def in_bounds(self, bounds: tuple[float, float]) -> bool:
        low, high = bounds
        return all(low <= v <= high for v in self.position)

    def update_personal_best(self) -> None:
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position[:]

    def move(self, social_best: list[float], w: float, c1: float, c2: float, vmax: float) -> None:
        # v(i+1) = w*v(i) + c1*r1*(personal_best - x) + c2*r2*(neighborhood_best - x)
        for d in range(len(self.position)):
            r1, r2 = random.random(), random.random()
            v = (w * self.velocity[d]
                 + c1 * r1 * (self.best_position[d] - self.position[d])
                 + c2 * r2 * (social_best[d] - self.position[d]))
            v = max(-vmax, min(vmax, v)) # limit the step size
            self.velocity[d] = v
            self.position[d] += v # x(i+1) = x(i) + v(i+1)

class Swarm:
    def __init__(self, particles: list[Particle], topology: str = "global",
                 neighborhood: int | None = None) -> None:
        if topology == "ring" and neighborhood is None:
            raise ValueError("Ring topology needs a neighborhood size (e.g. 3)")

        self.particles = particles
        self.topology = topology
        self.neighborhood: int = neighborhood if neighborhood is not None else len(particles)
        self.iteration = 0

    def global_best(self) -> Particle:
        return min(self.particles, key=lambda p: p.best_fitness)

    def neighbors(self, index: int) -> list[Particle]:
        # ring topology
        n = len(self.particles)
        half = (self.neighborhood - 1) // 2
        return [self.particles[(index + k) % n] for k in range(-half, half + 1)]

    def social_bests(self) -> list[list[float]]:
        if self.topology == "global":
            best = self.global_best().best_position[:]
            return [best] * len(self.particles)
        result = []
        for i in range(len(self.particles)):
            leader = min(self.neighbors(i), key=lambda p: p.best_fitness)
            result.append(leader.best_position[:])
        return result

def print_stats(results: list[float], label: str = ""):
    print(f"{label:<28} best={min(results):10.4g}  worst={max(results):10.4g}  "
          f"mean={statistics.mean(results):10.4g}  median={statistics.median(results):10.4g}  "
          f"std={statistics.pstdev(results):10.4g}")

# single run
def run_pso(objective, bounds: tuple[float, float], dimension: int, max_evals: int,
            inertia: tuple, c1: float, c2: float,
            topology: str, neighborhood: int | None,
            pop_size: int = POPULATION_SIZE):
    low, high = bounds
    vmax = VMAX_RATIO * (high - low)
    history: list[float] = []
    best_so_far = math.inf

    def evaluate(particle: Particle):
        nonlocal best_so_far
        particle.fitness = objective(particle.position)
        particle.update_personal_best()
        best_so_far = min(best_so_far, particle.fitness)
        history.append(best_so_far)

    # starting population
    particles = [Particle(dimension, bounds, vmax) for _ in range(min(pop_size, max_evals))]
    for particle in particles:
        evaluate(particle)
    swarm = Swarm(particles, topology, neighborhood)

    max_iterations = 10 * max_evals
    while len(history) < max_evals and swarm.iteration < max_iterations:
        # inertia weight
        if inertia[0] == "linear":
            w = inertia[1] - (inertia[1] - inertia[2]) * len(history) / max_evals
        else:
            w = inertia[1]

        targets = swarm.social_bests()
        for particle, target in zip(swarm.particles, targets):
            particle.move(target, w, c1, c2, vmax)
            # particles that fly outside are not evaluated
            if len(history) < max_evals and particle.in_bounds(bounds):
                evaluate(particle)
        swarm.iteration += 1

    return best_so_far, history, swarm

def main():
    pass

if __name__ == "__main__":
    main()
