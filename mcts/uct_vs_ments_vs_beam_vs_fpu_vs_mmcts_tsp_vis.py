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
from matplotlib.animation import FuncAnimation
from eval.utils import *

colors_box_plot = [ "#FFFFFF00", "#09a93e3e", "#09a93e8c", "#09a93ea7", "#09a93ed3", "#09a93e",]
colors_hist_plot = [ "#FFFFFF00", "#09a93e3e", "#09a93e8c", "#09a93ea7", "#09a93ed3", "#09a93e",]
edgecolor = ["#3838ff", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF"]
patterns = [ "", "/" , "\\" , "-" , "x" , "" ]
output_dir = "results/plots"

def filter_unique_changes(paths):
    unique = [paths[0]]
    for p in paths[1:]:
        if p != unique[-1]:
            unique.append(p)
    return unique

def animate_tsp(cities, best_path, title, fontsize):
    def animate(i):
        plt.cla()
        plt.plot([c[0] for c in cities], [c[1] for c in cities], "*", color="red", label="cities",  markersize=fontsize)
        plt.plot([c[0] for c in best_path[i]], [c[1] for c in best_path[i]], color="blue", label="path")
        plt.title(title, fontsize = fontsize)
        plt.xlabel("x", fontsize = fontsize)
        plt.ylabel("y", fontsize = fontsize)
    return animate
    

def save_best_tsp_path(cities, best_path, title, save_path, figsize=(7,7), fontsize=16):
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot([c[0] for c in cities], [c[1] for c in cities], "*", color="red", markersize=fontsize, label="cities")
    ax.plot([c[0] for c in best_path[-1]], [c[1] for c in best_path[-1]], color="blue", linewidth=2, label="path")

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("x", fontsize=fontsize)
    ax.set_ylabel("y", fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def draw_tsp_animation(cities, path_over_iter, title, save_path = None, figsize=(7,7), fontsize=16):
    cities = cities
    fig = plt.figure(figsize=figsize)
    path_over_iter = filter_unique_changes(path_over_iter)
    ani = FuncAnimation(fig, animate_tsp(cities, path_over_iter, title, fontsize=fontsize), frames=len(path_over_iter), interval=700)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ani.save(save_path, writer="ffmpeg", fps=2)
    plt.close(fig)

if __name__ == "__main__":
    nr_of_runs = 10

    cities = [
    # Cluster A (around 0, 0)
    (0, 0), (4, 0), (0, 6), (8, 8),
    # Cluster B (around 50, 0)
    (50, 2), (53, 3), (54, 3), (51, 4),
    # Cluster C (around 0, 50)
    (0, 58), (3, 50), (0, 57), (5, 53),
    # Far distractor cities
    (100, 25), (25, 100), (75, -20)
    ]

    iterations = 50
    base_temp = 1
    decay = 0.0
    epsilon = 2.0

    search_algorithms = {
        "MCTS": (mcts.uct_search, {}),
        "MENTS": (ments.ments_search, {'base_temp': 0.1, 'decay': 0.0001, 'epsilon': 0.1}),
        "DENTS": (dents.dents_search, {'base_temp': 0.1, 'decay': 0.0001, 'epsilon': 0.1}),
        "FPU": (fpu.uct_search_fpu, {'fpu': 0.1}),
        "BEAM": (beam.beam_mcts_search, {'beam_width': 30, 'sim_limit': 50}),
        "MMCTS": (mmcts.mmcts_search, {'base_temp': 1.0, 'decay': 0.0001, 'uct_inf_softening': 5, 'p_max': 1.5}),
    }
    labels = search_algorithms.keys()
    results = {name: {"best_path": [], "best_reward": [], "best_iter": [], "path_over_iter": []} for name in search_algorithms}
    means = []
    errors = []
    best_rewards = []
    best_iters = []
    for i in range(nr_of_runs):
        random.seed(i)
        random.shuffle(cities)
        tsp_env = tsp.TSPEnv(cities, seed=i)
        
        for name, (search_fn, params) in search_algorithms.items():
            root, best_path, best_reward, best_iter, path_over_iter = search_fn(
                tsp_env, iterations=iterations, seed=i, **params
            )

            # Append results
            results[name]["best_path"].append(best_path)
            results[name]["best_reward"].append(best_reward)
            results[name]["best_iter"].append(best_iter/iterations * 100)
            results[name]["path_over_iter"].append(path_over_iter)

    for name, (search_fn, params) in search_algorithms.items():
        means.append(np.mean(results[name]["best_reward"]))
        errors.append(np.std(results[name]["best_reward"]))
        best_iters.append(results[name]["best_iter"])
        title = f"{name} TSP"
        save_path = f"{output_dir}/{name}"
        fontsize = 20
        draw_tsp_animation(cities, results[name]["path_over_iter"][0], figsize = (7,7), title = title, save_path=f"{save_path}.mp4", fontsize=fontsize)
        save_best_tsp_path(cities, results[name]["path_over_iter"][0], title=title, save_path=f"{save_path}.png", figsize=(7,7), fontsize=fontsize)

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
