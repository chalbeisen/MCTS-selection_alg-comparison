import uct
import ments
import tsp
import mmcts_v2
import dents_v2
import node
import beam
import fpu

import random
import os
import numpy as np

import matplotlib.pyplot as plt

from graphviz import Digraph


colors = ["#8453F7", "#5BE0D1", "#EDC951","#0B46E7", "#014F3A" "#ED5192"]
labels = ['MCTS with UCT', 'MENTS', 'DENTS', 'BEAM', 'FPU', 'MMCTS']
output_dir = "results/plots"


def draw_tree(root: node._Node, filename="tree", max_depth: int = 3):
    """
    Draw a UCT tree up to a maximum depth.
    
    root: UCT_Node
    filename: output file name without extension
    max_depth: limit depth to avoid huge trees
    """
    dot = Digraph(comment='Tree')
    
    def add_node(node: node._Node, parent_id: str = None, depth: int = 0):
        if depth > max_depth:
            return
        node_id = str(id(node))
        label = f"Action: {node.action}\nVisits: {node.visits}\nValue: {node.edge_reward:.2f}"
        dot.node(node_id, label)
        if parent_id is not None:
            dot.edge(parent_id, node_id)
        for child in node.children:
            add_node(child, node_id, depth + 1)
    
    add_node(root)
    dot.render(filename, format="png", cleanup=True)
    print(f"Tree saved as {filename}.png")

def box_plot(data, yerrors, title, ylabel, labels, colors, figsize = (8,6), save_path = None):
    plt.figure(figsize=figsize)
    plt.bar(labels, data, yerr=yerrors, capsize=10, color=colors)

    plt.ylabel(ylabel, fontsize=18)
    plt.title(title, fontsize=24)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)
    plt.grid(axis='y')
    plt.tight_layout()


    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved box plot to {save_path}")
    plt.show()

def hist_plot(data, colors, xlabel, ylabel, labels, title = None, bins = 10, figsize = (8,6), save_path = f"{output_dir}/mean_max_reward.png"):
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

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved box plot to {save_path}")
    plt.show()



if __name__ == "__main__":
    nr_of_runs = 5
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

    iterations = 1000
    base_temp = 1
    decay = 0.0
    epsilon = 2.0

    # Result dictionaries
    uct_res = {"best_path": [], "max_reward": [], "best_iter": []}
    ments_res = {"best_path": [], "max_reward": [], "best_iter": []}
    dents_res = {"best_path": [], "max_reward": [], "best_iter": []}
    mmcts_res = {"best_path": [], "max_reward": [], "best_iter": []}
    fpu_res = {"best_path": [], "max_reward": [], "best_iter": []}
    beam_res = {"best_path": [], "max_reward": [], "best_iter": []}

    for i in range(nr_of_runs):
        random.seed(i)
        random.shuffle(cities)
        tsp_env = tsp.TSPEnv(cities, seed=i)

        # Run each algorithm
        root_uct, best_path_uct, max_reward_uct, best_iteration_uct = uct.uct_search(
            tsp_env, iterations=iterations, seed=i
        )
        root_ments, best_path_ments, max_reward_ments, best_iteration_ments = ments.ments_search(
            tsp_env, iterations=iterations, seed=i, base_temp=base_temp, decay=decay, epsilon=epsilon
        )
        root_dents, best_path_dents, max_reward_dents, best_iteration_dents = dents_v2.dents_search(
            tsp_env, iterations=iterations, seed=i, base_temp=base_temp, decay=decay, epsilon=epsilon
        )
        root_mmcts, best_path_mmcts, max_reward_mmcts, best_iteration_mmcts = mmcts_v2.mmcts_search(
            tsp_env, iterations=iterations, seed=i, base_temp=base_temp, decay=decay
        )
        root_fpu, best_path_fpu, max_reward_fpu, best_iteration_fpu = fpu.uct_search_fpu(
            tsp_env, iterations=iterations, seed=i, 
        )
        root_beam, best_path_beam, max_reward_beam, best_iteration_beam = beam.beam_mcts_search(
            tsp_env, iterations=iterations, seed=i, beam_width=5
        )

        # Draw trees
        """draw_tree(root_uct, filename=f"tree_uct_run{i}", max_depth=len(cities))
        draw_tree(root_ments, filename=f"tree_ments_run{i}", max_depth=len(cities))
        draw_tree(root_dents, filename=f"tree_dents_run{i}", max_depth=len(cities))
        draw_tree(root_mmcts, filename=f"tree_mmcts_run{i}", max_depth=len(cities))
        draw_tree(root_fpu, filename=f"tree_fpu_run{i}", max_depth=len(cities))
        draw_tree(root_beam, filename=f"tree_beam_run{i}", max_depth=len(cities))"""

        # Append results
        for res, path, reward, it in [
            (uct_res, best_path_uct, max_reward_uct, best_iteration_uct),
            (ments_res, best_path_ments, max_reward_ments, best_iteration_ments),
            (dents_res, best_path_dents, max_reward_dents, best_iteration_dents),
            (mmcts_res, best_path_mmcts, max_reward_mmcts, best_iteration_mmcts),
            (fpu_res, best_path_fpu, max_reward_fpu, best_iteration_fpu),
            (beam_res, best_path_beam, max_reward_beam, best_iteration_beam)
        ]:
            res["best_path"].append(path)
            res["max_reward"].append(reward)
            res["best_iter"].append(it)

    # Update labels and colors
    labels = ['MCTS with UCT', 'MENTS', 'DENTS', 'FPU', 'BEAM', 'MMCTS']
    colors = ["#8453F7", "#5BE0D1", "#EDC951", "#ED5192", "#FF7F50", "#1E90FF"]

    # Compute means and errors
    means = [
        np.mean(uct_res["max_reward"]),
        np.mean(ments_res["max_reward"]),
        np.mean(dents_res["max_reward"]),
        np.mean(fpu_res["max_reward"]),
        np.mean(beam_res["max_reward"]),
        np.mean(mmcts_res["max_reward"]),
    ]
    errors = [
        np.std(uct_res["max_reward"]),
        np.std(ments_res["max_reward"]),
        np.std(dents_res["max_reward"]),
        np.std(fpu_res["max_reward"]),
        np.std(beam_res["max_reward"]),
        np.std(mmcts_res["max_reward"]),
    ]

    # Box plot
    box_plot(
        means, errors, title='Mean Max Reward with Std Dev',
        ylabel='Mean Max Reward', labels=labels, colors=colors,
        save_path=f"{output_dir}/mean_max_reward.png"
    )

    # Histogram for rewards
    data_rewards = [
        uct_res["max_reward"], ments_res["max_reward"], dents_res["max_reward"],
        fpu_res["max_reward"], beam_res["max_reward"], mmcts_res["max_reward"],
    ]
    hist_plot(
        data_rewards, colors, xlabel="Minimum cost", ylabel="Count",
        labels=labels, save_path=f"{output_dir}/minimum_cost.png", bins=10
    )

    # Histogram for best iteration
    data_iters = [
        uct_res["best_iter"], ments_res["best_iter"], dents_res["best_iter"],
        fpu_res["best_iter"], beam_res["best_iter"], mmcts_res["best_iter"]
    ]
    hist_plot(
        data_iters, colors, xlabel="Iterations", ylabel="Count",
        labels=labels, save_path=f"{output_dir}/iterations.png", bins=10
    )
    print("test")
