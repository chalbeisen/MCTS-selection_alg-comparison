import itertools
import numpy as np
from tsp import TSPEnv   # your TSPEnv
from dents_v2 import dents_search  # your DENTS implementation

# Example TSP cities
cities = [
    # Cluster A (around 0, 0)
    (0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2),
    # Cluster B (around 50, 0)
    (50, 0), (51, 0), (50, 1), (51, 1), (52, 0), (50, 2),
    # Cluster C (around 0, 50)
    (0, 50), (1, 50), (0, 51), (1, 51), (2, 50), (0, 52),
    # Cluster D (around 50, 50)
    (50, 50), (51, 50), (50, 51), (51, 51), (52, 50), (50, 52),
    # Far “distractor” cities
    (100, 25), (25, 100), (75, -20)
    ]
env = TSPEnv(cities, seed=42)

# Hyperparameter grid
base_temps = [1.0, 5.0, 10.0, 20.0, 30.0, 40.0]
decays = [0.0001, 0.001, 0.01]
epsilons = [0.1, 0.5, 1.0]
base_betas = [0.5, 1.0, 2.0]  # optional, if your DENTS code allows

# Store results
results = []

# Grid search
for base_temp, decay, epsilon, base_beta in itertools.product(base_temps, decays, epsilons, base_betas):
    # Run DENTS search
    best_path, best_reward, best_iter = dents_search(
        root_env=env,
        iterations=10000,   # can increase for more thorough testing
        base_temp=base_temp,
        decay=decay,
        epsilon=epsilon,
        seed=42  # fixed seed for reproducibility
    )
    results.append({
        "base_temp": base_temp,
        "decay": decay,
        "epsilon": epsilon,
        "base_beta": base_beta,
        "best_distance": -best_reward,  # reward is negative distance
        "best_iter": best_iter
    })
    print(f"Tested base_temp={base_temp}, decay={decay}, epsilon={epsilon}, base_beta={base_beta} => distance={-best_reward:.3f}")

# Find best params
best_config = min(results, key=lambda x: x["best_distance"])
print("\nBest config found:")
print(best_config)