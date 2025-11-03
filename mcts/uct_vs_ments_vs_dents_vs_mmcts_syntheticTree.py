import uct
import ments
import synthetic_tree
import mmcts_v2
import dents_v2
import node

import random
import os
import numpy as np

import matplotlib.pyplot as plt

from graphviz import Digraph


colors = ["#8453F7", "#5BE0D1", "#EDC951", "#ED5192"]
labels = ['MCTS with UCT', 'MENTS', 'DENTS', 'MMCTS']
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
        label = f"Action: {node.action}\nVisits: {node.visits}\nValue: {node.value:.2f}"
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
    nr_of_runs = 50
    iterations = 1000
    base_temp = 1
    decay = 0.0
    epsilon = 2.0
    uct_res = {"best_path": [], "max_reward": [], "best_iter": []}
    ments_res = {"best_path": [], "max_reward": [], "best_iter": []}
    dents_res = {"best_path": [], "max_reward": [], "best_iter": []}
    mmcts_res = {"best_path": [], "max_reward": [], "best_iter": []}
    for i in range(nr_of_runs):
        random.seed(i)
        env = synthetic_tree.SyntheticTreeEnv(depth=4, seed=i)
        root_uct, best_path_uct, max_reward_uct, best_iteration_uct = uct.uct_search(env, iterations=iterations, seed=i)
        root_ments, best_path_ments, max_reward_ments, best_iteration_ments = ments.ments_search(env, iterations=iterations, seed=i, base_temp=base_temp, decay=decay, epsilon=epsilon)
        root_dents, best_path_dents, max_reward_dents, best_iteration_dents = dents_v2.dents_search(env, iterations=iterations, seed=i, base_temp=base_temp, decay=decay, epsilon=epsilon)
        root_mmcts, best_path_mmcts, max_reward_mmcts, best_iteration_mmcts = mmcts_v2.mmcts_search(env, iterations=iterations, seed=i, base_temp=base_temp, decay=decay)

        draw_tree(root_uct, filename="synthetic_tree_uct_tree", max_depth=4)
        draw_tree(root_ments, filename="synthetic_tree_ments_tree", max_depth=4)
        draw_tree(root_dents, filename="synthetic_tree_dents_tree", max_depth=4)
        draw_tree(root_mmcts, filename="synthetic_tree_mmcts_tree", max_depth=4)

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

    box_plot(means, errors, title = 'Mean Max Reward with Std Dev', ylabel= 'Mean Max Reward', labels=labels, colors=colors, save_path=f"{output_dir}/mean_max_reward.png")
    data = [uct_res["max_reward"], ments_res["max_reward"], dents_res["max_reward"], mmcts_res["max_reward"]]
    hist_plot(data, colors, xlabel="Minimum cost", ylabel="Count", labels=labels, save_path=f"{output_dir}/minimum_cost.png")
    data = [uct_res["best_iter"], ments_res["best_iter"], dents_res["best_iter"], mmcts_res["best_iter"]]
    hist_plot(data, colors, xlabel="Iterations", ylabel="Count", labels=labels, save_path=f"{output_dir}/iterations.png")