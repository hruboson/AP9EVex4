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

def run_experiment(objective, bounds, dimension: int, runs: int = RUNS_NO, **params):
    max_evals = EVALS_PER_DIM * dimension
    results, histories = [], []
    for _ in range(runs):
        best, history, _ = run_pso(objective, bounds, dimension, max_evals, **params)
        # pad in case the run ended early (should not happen)
        history += [history[-1]] * (max_evals - len(history))
        results.append(best)
        histories.append(history)
    average_history = [statistics.mean(h[i] for h in histories) for i in range(max_evals)]
    return results, average_history

def plot_results(curves: dict, all_results: dict):
    problems = list(curves)
    n_rows, n_cols = len(problems), len(DIMENSIONS)
    colors = ["tab:red", "tab:blue", "tab:green"]

    # convergence graphs
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 3.8 * n_rows), squeeze=False)
    for row, name in enumerate(problems):
        for col, dimension in enumerate(DIMENSIONS):
            ax = axes[row][col]
            for color, variant in zip(colors, VARIANTS):
                curve = curves[name][dimension][variant]
                ax.plot(range(1, len(curve) + 1), [max(v, 1e-12) for v in curve],
                        color=color, label=variant)
            ax.set_yscale("log")
            ax.set_title(f"{name} - {dimension}D")
            ax.set_xlabel("Evaluations")
            ax.set_ylabel(f"Mean best fitness ({RUNS_NO} runs)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"PSO - population {POPULATION_SIZE}, vmax = {VMAX_RATIO} * range",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("pso_convergence.png", dpi=150)

    # statistics table
    columns = ["Problem", "Dim", "Variant", "Best", "Worst", "Mean", "Median", "Std. dev."]
    rows = []
    for name in problems:
        for dimension in DIMENSIONS:
            for variant in VARIANTS:
                r = all_results[name][dimension][variant]
                rows.append([name, f"{dimension}D", variant,
                             f"{min(r):.3g}", f"{max(r):.3g}",
                             f"{statistics.mean(r):.3g}", f"{statistics.median(r):.3g}",
                             f"{statistics.pstdev(r):.3g}"])
    fig_t, ax_t = plt.subplots(figsize=(15, 0.3 * len(rows) + 1))
    ax_t.axis("off")
    table = ax_t.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(list(range(len(columns))))
    group = len(VARIANTS) * len(DIMENSIONS)
    for (row, col), cell in table.get_celld().items():
        if row == 0: # header
            cell.set_facecolor("#dbe5f1")
            cell.set_text_props(fontweight="bold")
        elif (row - 1) // group % 2 == 1: # shade every second problem
            cell.set_facecolor("#f4f4f4")
    fig_t.tight_layout()
    fig_t.savefig("pso_statistics.png", dpi=150)

def main():
    random.seed(SEED)
    print(f"Population: {POPULATION_SIZE}, runs: {RUNS_NO}, MAXFES: {EVALS_PER_DIM} * D\n")

    curves, all_results = {}, {}
    for name, (objective, bounds) in OBJECTIVES.items():
        print(f"=== {name} ===")
        curves[name], all_results[name] = {}, {}
        for dimension in DIMENSIONS:
            curves[name][dimension], all_results[name][dimension] = {}, {}
            print(f"-- {dimension}D ({EVALS_PER_DIM * dimension} evals)")
            for variant, params in VARIANTS.items():
                results, curve = run_experiment(objective, bounds, dimension, RUNS_NO, **params)
                all_results[name][dimension][variant] = results
                curves[name][dimension][variant] = curve
                print_stats(results, variant)
        print()

    plot_results(curves, all_results)

    if "agg" not in plt.get_backend().lower():
        plt.show()

if __name__ == "__main__":
    main()
