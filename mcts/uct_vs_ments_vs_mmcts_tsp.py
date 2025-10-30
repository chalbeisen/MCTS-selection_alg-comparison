import uct
import ments
import tsp
import mmcts_v2
import random

import numpy as np

import matplotlib.pyplot as plt

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
    # Far “distractor” cities
    (100, 25), (25, 100), (75, -20)
    ]
    iterations = 100
    uct_res = {"best_path": [], "max_reward": [], "best_iter": []}
    ments_res = {"best_path": [], "max_reward": [], "best_iter": []}
    mmcts_res = {"best_path": [], "max_reward": [], "best_iter": []}
    for i in range(nr_of_runs):
        random.seed(i)
        random.shuffle(cities)
        tsp_env = tsp.TSPEnv(cities)
        best_path_uct, max_reward_uct, best_iteration_uct = uct.uct_search(tsp_env, iterations=iterations, seed=i)
        best_path_ments, max_reward_ments, best_iteration_ments = ments.ments_search(tsp_env, iterations=iterations, seed=i)
        best_path_mmcts, max_reward_mmcts, best_iteration_mmcts = mmcts_v2.mmcts_search(tsp_env, iterations=iterations, seed=i)

        for res, path, reward, it in [
            (uct_res, best_path_uct, max_reward_uct, best_iteration_uct),
            (ments_res, best_path_ments, max_reward_ments, best_iteration_ments),
            (mmcts_res, best_path_mmcts, max_reward_mmcts, best_iteration_mmcts),
        ]:
            res["best_path"] += [path]
            res["max_reward"] += [reward]
            res["best_iter"] += [it]



    plt.hist(uct_res["max_reward"], bins=10, color='r', label='mcts')
    plt.hist(ments_res["max_reward"], bins=10, color='y', label='ments')
    plt.hist(mmcts_res["max_reward"], bins=10, color='g', label='mmcts')
    plt.xlabel("Minimum cost")
    plt.ylabel("Count")
    plt.legend()
    plt.show(block=True)

    plt.figure()  # start a fresh figure

    plt.hist(uct_res["best_iter"], bins=10, color='r', label='mcts_bestiter')
    plt.hist(ments_res["best_iter"], bins=10, color='y', label='ments_bestiter')
    plt.hist(mmcts_res["best_iter"], bins=10, color='g', label='mmcts_bestiter')
    plt.xlabel("Iterations")
    plt.ylabel("Count")
    plt.legend()
    plt.show(block=True)