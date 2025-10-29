import uct
import ments
import job_scheduling
import mmcts

if __name__ == "__main__":
    jobscheduling_env = job_scheduling.JobSchedulingEnv(jobs=[(3, 2.0), (2, 1.0), (1, 3.0)])
    #best_path_uct = uct.uct_search(knapsack_env, iterations=100, seed=42)
    #best_path_ments = ments.ments_search(knapsack_env, iterations=100, seed=42)
    best_path_mmcts = mmcts.mmcts_search(jobscheduling_env, iterations=100, seed=42)