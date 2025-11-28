from mcts.mcts import MCTS_Search

from env.tic_tac_toe import TicTacToeEnv
from mcts.mcts import MCTS_Search
from adaptations.mmcts import MMCTS_Search


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

    env = TicTacToeEnv()
    search_algorithms = {
        "MCTS": {
            "instance": MCTS_Search(env),                        
            "search_fn": "mcts_search_turn_based",            
            "params": {}                    
        },

        "MMCTS": {
            "instance": MMCTS_Search(env),
            "search_fn": "mmcts_search_turn_based",
            "params": {'base_temp': 1.0, 'decay': 0.0001, 'uct_inf_softening': 5, 'p_max': 1.5}
        },
    }

    labels = list(search_algorithms.keys())
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
        env.reset_to_initial_state([])
        while not env.is_terminal():
            current_player = env.get_current_player()
            # Get search_fn
            current_player_alg = labels[current_player]
            cfg = search_algorithms[current_player_alg]
            search_fn = getattr(cfg["instance"], cfg["search_fn"]) 
            # Run algorithm
            action = search_fn(iterations=iterations, **cfg["params"])
        print(env.get_state())
        winner = env.get_winner()
        print(winner)
        if winner >= 0:
            results[winner]["wins"]+=1

    print(results)
