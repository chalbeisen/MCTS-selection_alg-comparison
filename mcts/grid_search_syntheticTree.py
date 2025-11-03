import itertools
import numpy as np
from tsp import TSPEnv   # your TSPEnv
from dents_v2 import dents_search  # your DENTS implementation
import synthetic_tree
import random

# Example TSP cities
seed = 0
random.seed(seed)  
env = synthetic_tree.SyntheticTreeEnv(depth=4, seed=seed)

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
        seed=seed # fixed seed for reproducibility
    )
    results.append({
        "base_temp": base_temp,
        "decay": decay,
        "epsilon": epsilon,
        "base_beta": base_beta,
        "best_distance": best_reward, 
        "best_iter": best_iter
    })
    print(f"Tested base_temp={base_temp}, decay={decay}, epsilon={epsilon}, base_beta={base_beta} => best_reward={best_reward:.3f}")

# Find best params
best_config = max(results, key=lambda x: x["best_distance"])
print("\nBest config found:")
print(best_config)