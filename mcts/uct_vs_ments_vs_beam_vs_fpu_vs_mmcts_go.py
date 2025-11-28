from mcts.mcts import MCTS_Search
from adaptations.mmcts import MMCTS_Search
import adaptations.ments as ments
from env.go import GoEnv
import adaptations.mmcts as mmcts
import adaptations.dents as dents
import adaptations.beam as beam
import adaptations.fpu as fpu

import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from eval.utils import *

colors_box_plot = [ "#FFFFFF00", "#09a93e3e", "#09a93e8c", "#09a93ea7", "#09a93ed3", "#09a93e",]
colors_hist_plot = [ "#FFFFFF00", "#09a93e3e", "#09a93e8c", "#09a93ea7", "#09a93ed3", "#09a93e",]
edgecolor = ["#3838ff", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF"]
patterns = [ "", "/" , "\\" , "-" , "x" , "" ]
output_dir = "results/plots"
seed = 1


if __name__ == "__main__":
    N = 5

    nr_of_runs = 50
    iterations = 1000
    base_temp = 1
    decay = 0.0
    epsilon = 2.0

    env = GoEnv()
    search_algorithms = [
        {"alg_name": "MCTS", "params": {}, "search_fn": MCTS_Search(env).uct_search_turn_based},
        {"alg_name": "MMCTS", "params": {}, "search_fn": MMCTS_Search(env).mmcts_search_turn_based}
    ]

    labels = [d["alg_name"] for d in search_algorithms]
    results = [{"wins": 0} for i in range(len(search_algorithms))]
    means = [] 
    errors = []
    best_rewards = []
    best_iters = []
    current_player = 0
    for i in range(nr_of_runs):
        random.seed(i)
        torch.manual_seed(i)
        np.random.seed(i)
        # switch up first player:
        # search_algorithms[0], search_algorithms[1] = search_algorithms[1], search_algorithms[0]
        env.set_initial_state([])
        env.reset_to_initial_state()
        while not env.is_terminal():
            current_player = env.get_current_player()
            action = search_algorithms[current_player]["search_fn"](
                iterations=iterations, **search_algorithms[current_player]["params"]
            )
            print(env.state)
        print(env.get_state())
        
        winner = env.get_winner()
        print(winner)
        if winner >= 0:
            results[winner]["wins"]+=1

    print(results)
