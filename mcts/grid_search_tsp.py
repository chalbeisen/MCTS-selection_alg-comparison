import itertools
import numpy as np
import random
import torch
from env.tsp import TSPEnv
from adaptations.dents import dents_search  
from adaptations.ments import ments_search  
from adaptations.mmcts import mmcts_search
from adaptations.fpu import uct_search_fpu
from adaptations.beam import beam_mcts_search

import json

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

dir = "./grid_search_results"

# Hyperparameter grid
# MENTS/DENTS
base_temps = [0.1, 0.2, 0.5, 1.0]
decays = [0.000001, 0.00001, 0.0001]
epsilons = [0.01, 0.05, 0.1]
# FPU
fpu_constants = [0.1, 0.5, 1, 2, 5, 10]
# BEAM MCTS SEARCH
beam_widths = [10, 20, 30, 40, 50]
sim_limits = [10, 20, 50, 100, 200, 500]
# MMCTS
uct_inf_softenings = [2, 3, 5, 10, 20, 50]
p_maxs = [1.2, 1.5, 1.7, 2.0, 2.5, 3.0]
results_ments = []
results_dents = []
results_mmcts = []
results_fpu = []
results_beam = []
seed = 0
iterations = 1000

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 

# Grid search

# MENTS, DENTS
for base_temp, decay, epsilon in itertools.product(base_temps, decays, epsilons):
    root_ments, best_path_ments, best_reward_ments, best_iteration_ments = ments_search(
        root_env=env, 
        iterations=iterations, 
        seed=seed, 
        base_temp=base_temp, 
        decay=decay, 
        epsilon=epsilon
    )

    root_dents, best_path_dents, best_reward_dents, best_iteration_dents = dents_search(
        root_env=env,
        iterations=iterations,   
        base_temp=base_temp,
        decay=decay,
        epsilon=epsilon,
        seed=seed,
    )

    results_ments.append({
        "base_temp": base_temp,
        "decay": decay,
        "epsilon": epsilon,
        "best_distance": best_reward_ments, 
        "best_iter": best_iteration_ments
    })
    results_dents.append({
        "base_temp": base_temp,
        "decay": decay,
        "epsilon": epsilon,
        "best_distance": best_reward_dents, 
        "best_iter": best_iteration_dents
    })
    print(f"MENTS base_temp={base_temp}, decay={decay}, epsilon={epsilon} => distance={best_reward_ments:.3f}")
    print(f"DENTS base_temp={base_temp}, decay={decay}, epsilon={epsilon} => distance={best_reward_dents:.3f}")

# FPU
for fpu in fpu_constants:
    root_fpu, best_path_fpu, best_reward_fpu, best_iteration_fpu = uct_search_fpu(
            env,
            iterations=iterations,
            seed=seed, 
            fpu=fpu
    )
    results_fpu.append({
        "fpu": fpu,
        "best_distance": best_reward_fpu, 
        "best_iter": best_iteration_fpu
    })
    print(f"FPU fpu={fpu} => distance={best_reward_fpu:.3f}")

# BEAM MCTS
for (beam_width, sim_limit) in zip(beam_widths, sim_limits):
    root_beam, best_path_beam, best_reward_beam, best_iteration_beam = beam_mcts_search(
            env,
            iterations=iterations,
            seed=seed, 
            beam_width=beam_width,
            sim_limit=sim_limit
    )
    results_beam.append({
        "beam_width": beam_width,
        "sim_limit": sim_limit,
        "best_distance": best_reward_beam, 
        "best_iter": best_iteration_beam
    })
    print(f"BEAM MCTS beam_width={beam_width}, sim_limit={sim_limit} => distance={best_reward_beam:.3f}")

# MMCTS
for base_temp, decay, uct_inf_softening, p_max in itertools.product(base_temps, decays, uct_inf_softenings, p_maxs):
    root_mmcts, best_path_mmcts, best_reward_mmcts, best_iteration_mmcts = mmcts_search(
        root_env=env,
        iterations=iterations,   
        base_temp=base_temp,
        decay=decay,
        uct_inf_softening=uct_inf_softening,
        p_max=p_max
    )
    results_mmcts.append({
        "base_temp": base_temp,
        "decay": decay,
        "epsilon": epsilon,
        "base_beta": base_beta,
        "uct_inf_softening": uct_inf_softening,
        "p_max": p_max,
        "best_distance": best_reward_mmcts, 
        "best_iter": best_iteration_mmcts
    })
    print(f"MMCTS base_temp={base_temp}, decay={decay}, epsilon={epsilon}, uct_inf_softening={uct_inf_softening}, p_max={p_max} => distance={best_reward_mmcts:.3f}")

# Find best params
best_results = {}
best_results["ments"] = min(results_ments, key=lambda x: x["best_distance"])
print("\n MENTS best config found:")
print(best_results["ments"])

best_results["dents"] = min(results_dents, key=lambda x: x["best_distance"])
print("\n DENTS best config found:")
print(best_results["dents"])

best_results["fpu"] = min(results_fpu, key=lambda x: x["best_distance"])
print("\n FPU best config found:")
print(best_results["fpu"])

best_results["beam"] = min(results_beam, key=lambda x: x["best_distance"])
print("\n BEAM MCTS best config found:")
print(best_results["beam"])

best_results["mmcts"] = min(results_mmcts, key=lambda x: x["best_distance"])
print("\n MMCTS best config found:")
print(best_results["mmcts"])

import os
os.makedirs(dir, exist_ok=True)
with open("grid_search_results/best_parameters_tsp.json", "w") as f:
    json.dump(best_results, f)