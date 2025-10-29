import uct
import ments
import knapsack
import mmcts

if __name__ == "__main__":
    knapsack_env = knapsack.KnapsackEnv([(3, 4.0), (2, 3.0), (1, 2.0), (2, 4.0), (5, 2.1)], capacity=5)
    #best_path_uct = uct.uct_search(knapsack_env, iterations=100, seed=42)
    #best_path_ments = ments.ments_search(knapsack_env, iterations=100, seed=42)
    best_path_mmcts = mmcts.mmcts_search(knapsack_env, iterations=100, seed=42)