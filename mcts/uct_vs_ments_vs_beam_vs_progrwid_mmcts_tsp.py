import uct
import ments
import tsp
import mmcts_v2
import dents_v2
import random

import numpy as np

import matplotlib.pyplot as plt

colors = ["#8453F7", "#5BE0D1", "#EDC951", "#ED5192"]
labels = ['MCTS with UCT', 'MENTS', 'DENTS', 'MMCTS']


def box_plot(data, yerrors, title, ylabel, labels, colors, figsize = (8,6)):
    plt.figure(figsize=figsize)
    plt.bar(labels, data, yerr=yerrors, capsize=10, color=colors)

    plt.ylabel(ylabel, fontsize=18)
    plt.title(title, fontsize=24)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)
    plt.grid(axis='y')
    plt.tight_layout()

    plt.show()

def hist_plot(data, colors, xlabel, ylabel, labels, title = None, bins = 10, figsize = (8,6)):
    plt.figure(figsize=figsize)
    for d, l, c in zip(data, labels, colors):
        plt.hist(d, bins=bins, color=c, label=l)

    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title, fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.legend()
    plt.show()




if __name__ == "__main__":
    nr_of_runs = 50
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
    iterations = 10000
    base_temp = 1
    decay = 0.0
    epsilon = 2.0
    uct_res = {"best_path": [], "max_reward": [], "best_iter": []}
    ments_res = {"best_path": [], "max_reward": [], "best_iter": []}
    dents_res = {"best_path": [], "max_reward": [], "best_iter": []}
    mmcts_res = {"best_path": [], "max_reward": [], "best_iter": []}
    for i in range(nr_of_runs):
        random.seed(i)
        random.shuffle(cities)
        tsp_env = tsp.TSPEnv(cities, seed=i)
        best_path_uct, max_reward_uct, best_iteration_uct = uct.uct_search(tsp_env, iterations=iterations, seed=i)
        best_path_ments, max_reward_ments, best_iteration_ments = ments.ments_search(tsp_env, iterations=iterations, seed=i, base_temp=base_temp, decay=decay, epsilon=epsilon)
        best_path_dents, max_reward_dents, best_iteration_dents = dents_v2.dents_search(tsp_env, iterations=iterations, seed=i, base_temp=base_temp, decay=decay, epsilon=epsilon)
        best_path_mmcts, max_reward_mmcts, best_iteration_mmcts = mmcts_v2.mmcts_search(tsp_env, iterations=iterations, seed=i, base_temp=base_temp, decay=decay)

        for res, path, reward, it in [
            (uct_res, best_path_uct, max_reward_uct, best_iteration_uct),
            (ments_res, best_path_ments, max_reward_ments, best_iteration_ments),
            (dents_res, best_path_dents, max_reward_dents, best_iteration_dents),
            (mmcts_res, best_path_mmcts, max_reward_mmcts, best_iteration_mmcts),
        ]:
            res["best_path"] += [path]
            res["max_reward"] += [reward]
            res["best_iter"] += [it]


    means = [
            np.mean(uct_res["max_reward"]), 
            np.mean(ments_res["max_reward"]),
            np.mean(dents_res["max_reward"]),
            np.mean(mmcts_res["max_reward"])
            ]
    errors = [
            np.std(uct_res["max_reward"]), 
            np.std(ments_res["max_reward"]), 
            np.std(dents_res["max_reward"]), 
            np.std(mmcts_res["max_reward"])
            ]

    box_plot(means, errors, title = 'Mean Max Reward with Std Dev', ylabel= 'Mean Max Reward', labels=labels, colors=colors)
    data = [uct_res["max_reward"], ments_res["max_reward"], dents_res["max_reward"], mmcts_res["max_reward"]]
    hist_plot(data, colors, xlabel="Minimum cost", ylabel="Count", labels=labels)
    data = [uct_res["best_iter"], ments_res["best_iter"], dents_res["best_iter"], mmcts_res["best_iter"]]
    hist_plot(data, colors, xlabel="Iterations", ylabel="Count", labels=labels)