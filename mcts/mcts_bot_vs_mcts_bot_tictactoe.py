from mcts.mcts import MCTS_Search

from env.tic_tac_toe import TicTacToeEnv
from mcts.mcts import MCTS_Search
from adaptations.mmcts import MMCTS_Search


import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import math

from eval.utils import *
from mcts.mcts_open_spiel import MCTSBot, SearchNode, RandomRolloutEvaluator

colors_box_plot = [ "#FFFFFF00", "#09a93e3e", "#09a93e8c", "#09a93ea7", "#09a93ed3", "#09a93e",]
colors_hist_plot = [ "#FFFFFF00", "#09a93e3e", "#09a93e8c", "#09a93ea7", "#09a93ed3", "#09a93e",]
edgecolor = ["#3838ff", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF"]
patterns = [ "", "/" , "\\" , "-" , "x" , "" ]
output_dir = "results/plots"
seed = 1


if __name__ == "__main__":
    N = 5

    nr_of_runs = 1
    iterations = 2000
    base_temp = 1
    decay = 0.0
    epsilon = 2.0

    env = TicTacToeEnv()

    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    evaluator = RandomRolloutEvaluator(1, rng)
    search_algorithms = {
        "MCTS_3": {
            "instance": MCTSBot(game = env.game, 
                                     uct_c=math.sqrt(2),
                                     max_simulations=iterations,
                                     evaluator=evaluator,
                                     random_state=rng,
                                     solve=False,
                                     dirichlet_noise=None, 
                                     child_selection_fn=SearchNode.uct_value),                        
            "search_fn": "step",                        
        },
        "MCTS_4": {
            "instance": MCTSBot(game = env.game, 
                                     uct_c=math.sqrt(2),
                                     max_simulations=iterations,
                                     evaluator=evaluator,
                                     random_state=rng,
                                     solve=False,
                                     dirichlet_noise=None, 
                                     child_selection_fn=SearchNode.uct_value),                        
            "search_fn": "step",                        
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
        # switch up first player:
        # search_algorithms[0], search_algorithms[1] = search_algorithms[1], search_algorithms[0]
        env.reset_to_initial_state([])
        j = 0
        while not env.is_terminal():
            current_player = env.get_current_player()
            # Get search_fn
            current_player_alg = labels[current_player]
            cfg = search_algorithms[current_player_alg]
            search_fn = getattr(cfg["instance"], cfg["search_fn"]) 

            action, root = search_fn(env.state)
            env.state.apply_action(action)
            env.set_initial_state(env.get_state())
            #draw_tree_bot(root, env, filename=f"tree_{current_player_alg}_tictactoe_run{i}_{j}", max_depth=9, is_print_action = False, add_terminal_state = True)
            if j % 2 != 0:
                print("\nj   Action  Visit_Count  Value")
                print("###############################")
                result = sorted([(c.action, c.explore_count, c.explore_count and c.total_reward / c.explore_count) for c in root.children], key=lambda x: x[0])
                for (action, visit_count, value) in result:
                    print(f"{j}      {action}  {visit_count}  {round(value, 3)}")
            j+=1
                
        print(env.get_state())
        winner = env.get_winner()
        print(winner)
        if winner >= 0:
            results[winner]["wins"]+=1
        print(env.state)

    print(results)
