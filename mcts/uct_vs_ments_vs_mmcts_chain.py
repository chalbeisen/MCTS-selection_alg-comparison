import uct
import ments
import tsp
import mmcts_v2
import dents_v2
import random

import numpy as np
import chain

import matplotlib.pyplot as plt

colors = ["#8453F7", "#5BE0D1", "#EDC951", "#ED5192"]
labels = ['MCTS with UCT', 'MENTS', 'DENTS', 'MMCTS']

ylabel = 'Mean Max Reward'
title = 'Mean Max Reward with Std Dev'

base_temp = 50.0
decay = 0.0


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


if __name__ == "__main__":
    nr_of_runs = 10
    iterations = 10000
    uct_res = {"best_path": [], "max_reward": [], "best_iter": []}
    ments_res = {"best_path": [], "max_reward": [], "best_iter": []}
    dents_res = {"best_path": [], "max_reward": [], "best_iter": []}
    mmcts_res = {"best_path": [], "max_reward": [], "best_iter": []}
    for i in range(nr_of_runs):
        chain_env = chain.ChainEnv(D=10, Rf=1.0)

        best_path_uct, max_reward_uct, best_iteration_uct = uct.uct_search(chain_env, iterations=iterations, seed=i)
        best_path_ments, max_reward_ments, best_iteration_ments = ments.ments_search(chain_env, iterations=iterations, seed=i, base_temp=base_temp, decay=decay)
        best_path_dents, max_reward_dents, best_iteration_dents = dents_v2.dents_search(chain_env, iterations=iterations, seed=i, base_temp=base_temp, decay=decay)
        best_path_mmcts, max_reward_mmcts, best_iteration_mmcts = mmcts_v2.mmcts_search(chain_env, iterations=iterations, seed=i, base_temp=base_temp, decay=decay)

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

    box_plot(means, errors, title, ylabel, labels, colors)
    print("test")
