import mcts.mcts as mcts
import adaptations.ments as ments
import env.tsp as tsp
import adaptations.mmcts as mmcts
import adaptations.dents as dents
import adaptations.beam as beam
import adaptations.fpu as fpu

import random
import numpy as np
import matplotlib.pyplot as plt

from eval.utils import *

colors_box_plot = ["#09a93e3e", "#09a93e8c", "#09a93ea7", "#09a93ed3", "#09a93e", "#FFFFFF00"]
colors_hist_plot = ["#09a93e3e", "#09a93e8c", "#09a93ea7", "#09a93ed3", "#09a93e", "#FFFFFF00"]
edgecolor = ["#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#3838ff"]
patterns = [ "/" , "\\" , "-" , "x" , "" , "" ]
output_dir = "results/plots"

if __name__ == "__main__":
    nr_of_runs = 5
    cities = [
    # Cluster A (around 0, 0)
    (0, 0),
    # Cluster B (around 50, 0)
    (50, 0), (51, 0),
    # Cluster C (around 50, 50)
    (50, 50),
    # Far distractor cities
    (100, 25), (25, 100), (75, -20)
    ]

    iterations = 50
    base_temp = 1
    decay = 0.0
    epsilon = 2.0

    search_algorithms = {
        "UCT": (mcts.uct_search, {}),
        "MENTS": (ments.ments_search, {'base_temp': 0.1, 'decay': 0.0001, 'epsilon': 0.1}),
        "DENTS": (dents.dents_search, {'base_temp': 0.1, 'decay': 0.0001, 'epsilon': 0.1}),
        "FPU": (fpu.uct_search_fpu, {'fpu': 0.1}),
        "BEAM": (beam.beam_mcts_search, {'beam_width': 30, 'sim_limit': 50}),
        "MMCTS": (mmcts.mmcts_search, {'base_temp': 1.0, 'decay': 0.0001, 'uct_inf_softening': 5, 'p_max': 1.5}),
    }
    labels = search_algorithms.keys()
    results = {name: {"best_path": [], "best_reward": [], "best_iter": [], "root": []} for name in search_algorithms}
    means = []
    errors = []
    best_rewards = []
    best_iters = []
    best_paths = []

    seed = 0
    random.seed(seed)
    tsp_env = tsp.TSPEnv(cities, seed=seed)
    for i in range(nr_of_runs): 

        for name, (search_fn, params) in search_algorithms.items():
            root, best_path, best_reward, best_iter, _ = search_fn(
                tsp_env, iterations=iterations, seed=i, **params
            )

            # Append results
            results[name]["best_path"].append(best_path)
            results[name]["best_reward"].append(best_reward)
            results[name]["best_iter"].append(best_iter/iterations * 100)
            results[name]["root"].append(root)


    for name in search_algorithms.keys():
        best_path_index, _ = min(enumerate(results[name]["best_reward"]), key=lambda x: x[1])
        best_path = [None]+results[name]["best_path"][best_path_index][:-1]
        means.append(np.mean(results[name]["best_reward"]))
        errors.append(np.std(results[name]["best_reward"]))
        best_iters.append(results[name]["best_iter"])
        draw_tree(results[name]["root"][best_path_index], tsp_env, filename=f"tree_{name}", max_depth=len(cities), best_path=best_path, is_print_action=False)

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
