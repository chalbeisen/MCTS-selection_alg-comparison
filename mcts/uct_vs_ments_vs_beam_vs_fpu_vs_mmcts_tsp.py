import env.tsp as tsp

from mcts.mcts import MCTS_Search
from adaptations.mmcts import MMCTS_Search
from adaptations.dents import DENTS_Search
from adaptations.ments import MENTS_Search
from adaptations.beam import BEAM_Search
from adaptations.fpu import FPU_Search

import random
import numpy as np
import matplotlib.pyplot as plt

from eval.utils import *

colors_box_plot = [ "#FFFFFF00", "#09a93e3e", "#09a93e8c", "#09a93ea7", "#09a93ed3", "#09a93e",]
colors_hist_plot = [ "#FFFFFF00", "#09a93e3e", "#09a93e8c", "#09a93ea7", "#09a93ed3", "#09a93e",]
edgecolor = ["#3838ff", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF"]
patterns = [ "", "/" , "\\" , "-" , "x" , "" ]
output_dir = "results/plots"

if __name__ == "__main__":
    nr_of_runs = 10
    cities = [
    # Cluster A (around 0, 0)
    (0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2),
    # Cluster B (around 50, 0)
    (50, 0), (51, 0), (50, 1), (51, 1), (52, 0), (50, 2),
    # Cluster C (around 0, 50)
    (0, 50), (1, 50), (0, 51), (1, 51), (2, 50), (0, 52),
    # Cluster D (around 50, 50)
    (50, 50), (51, 50), (50, 51), (51, 51), (52, 50), (50, 52),
    # Far distractor cities
    (100, 25), (25, 100), (75, -20)
    ]

    iterations = 100
    base_temp = 1
    decay = 0.0
    epsilon = 2.0
    env = tsp.TSPEnv(cities)

    search_algorithms = {
        "MCTS": {
            "instance": MCTS_Search(env),                        
            "search_fn": "mcts_search",            
            "params": {}                    
        },

        "MENTS": {
            "instance": MENTS_Search(env),
            "search_fn": "ments_search",
            "params": {'base_temp': 0.1, 'decay': 0.0001, 'epsilon': 0.1}
        },

        "DENTS": {
            "instance": DENTS_Search(env),
            "search_fn": "dents_search",
            "params": {'base_temp': 0.1, 'decay': 0.0001, 'epsilon': 0.1}
        },

        "FPU": {
            "instance": FPU_Search(env),
            "search_fn": "fpu_search",
            "params": {'fpu': 0.1}
        },

        "BEAM": {
            "instance": BEAM_Search(env),
            "search_fn": "beam_mcts_search",
            "params": {'beam_width': 30, 'sim_limit': 50}
        },

        "MMCTS": {
            "instance": MMCTS_Search(env),
            "search_fn": "mmcts_search",
            "params": {'base_temp': 1.0, 'decay': 0.0001, 'uct_inf_softening': 5, 'p_max': 1.5}
        },
    }

    labels = list(search_algorithms.keys())
    results = {name: {"best_path": [], "best_reward": [], "best_iter": []} for name in search_algorithms}
    means = []
    errors = []
    best_rewards = []
    best_iters = []
    for i in range(nr_of_runs):
        random.seed(i)
        random.shuffle(cities)
        tsp.cities = cities

        for name, cfg in search_algorithms.items():
            # Get search_fn
            search_fn = getattr(cfg["instance"], cfg["search_fn"]) 
            # Run algorithm
            root, best_path, best_reward, best_iter, _  = search_fn(iterations=iterations, **cfg["params"])

            # Append results
            results[name]["best_path"].append(best_path)
            results[name]["best_reward"].append(best_reward)
            results[name]["best_iter"].append(best_iter/iterations * 100)

            #draw_tree(root, env, filename=f"tree_{name}_run{i}", max_depth=len(cities), is_print_action = False)

    for name in search_algorithms.keys():
        means.append(np.mean(results[name]["best_reward"]))
        errors.append(np.std(results[name]["best_reward"]))
        best_iters.append(results[name]["best_iter"])

    box_plot(
        means, errors,
        ylabel='Mean Max Reward', labels=labels, colors=colors_box_plot,
        save_path=f"{output_dir}/mean_best_reward.png", patterns = patterns, edgecolor = edgecolor
    )
    hist_plot(
        best_iters, colors_hist_plot, xlabel="Iterations", ylabel="Count",
        labels=labels, save_path=f"{output_dir}/iterations.png", bins=10, patterns = patterns, edgecolor = edgecolor
    )
    print("test")
