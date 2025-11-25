import mcts.mcts as mcts
import adaptations.ments as ments
from env.go import GoEnv
import adaptations.mmcts as mmcts
import adaptations.dents as dents
import adaptations.beam as beam
import adaptations.fpu as fpu

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
    N = 5

    nr_of_runs = 50
    iterations = 1000
    base_temp = 1
    decay = 0.0
    epsilon = 2.0

    search_algorithms = {
        "MCTS": (mcts.uct_search, {}),}
    """ 
        "MENTS": (ments.ments_search, {'base_temp': 0.1, 'decay': 0.0001, 'epsilon': 0.1}),
        "DENTS": (dents.dents_search, {'base_temp': 0.1, 'decay': 0.0001, 'epsilon': 0.1}),
        "FPU": (fpu.uct_search_fpu, {'fpu': 0.1}),
        "BEAM": (beam.beam_mcts_search, {'beam_width': 30, 'sim_limit': 50}),
        "MMCTS": (mmcts.mmcts_search, {'base_temp': 1.0, 'decay': 0.0001, 'uct_inf_softening': 5, 'p_max': 1.5}),
        
    }"""
    labels = search_algorithms.keys()
    results = {name: {"best_path": [], "best_reward": [], "best_iter": []} for name in search_algorithms}
    means = []
    errors = []
    best_rewards = []
    best_iters = []
    for i in range(nr_of_runs):
        env = GoEnv(N)

        for name, (search_fn, params) in search_algorithms.items():
            root, best_path, best_reward, best_iter = search_fn(
                env, iterations=iterations, seed=i, **params
            )

            # Append results
            results[name]["best_path"].append(best_path)
            results[name]["best_reward"].append(best_reward)
            results[name]["best_iter"].append(best_iter/iterations * 100)

            #draw_tree(root, filename=f"tree_{name}_run{i}", max_depth=len(cities))

    for name, (search_fn, params) in search_algorithms.items():
        means.append(np.mean(results[name]["best_reward"]))
        errors.append(np.std(results[name]["best_reward"]))
        best_iters.append(results[name]["best_iter"])

    box_plot(
        means, errors,
        ylabel='Mean Max Reward', labels=labels, colors=colors_box_plot,
        save_path=f"{output_dir}/mean_best_reward_go.png", patterns = patterns, edgecolor = edgecolor
    )
    hist_plot(
        best_iters, colors_hist_plot, xlabel="Iterations", ylabel="Count",
        labels=labels, save_path=f"{output_dir}/iterations_go.png", bins=10, patterns = patterns, edgecolor = edgecolor
    )
    print("test")
