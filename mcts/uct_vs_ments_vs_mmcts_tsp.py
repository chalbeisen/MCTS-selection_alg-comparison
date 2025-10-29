import uct
import ments
import tsp
import mmcts_v2

if __name__ == "__main__":
    tsp_env = tsp.TSPEnv(cities=[(0, 0), (1, 0), (0, 1), (1, 1)])
    #best_path_uct = uct.uct_search(knapsack_env, iterations=100, seed=42)
    #best_path_ments = ments.ments_search(knapsack_env, iterations=100, seed=42)
    best_path_mmcts = mmcts_v2.mmcts_search(tsp_env, iterations=100, seed=42)