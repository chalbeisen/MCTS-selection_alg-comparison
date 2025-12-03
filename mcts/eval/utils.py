import mcts.node as node
import os
import numpy as np

import matplotlib.pyplot as plt

from graphviz import Digraph
from env.env import Env
from typing import List, Tuple

def draw_tree(root: node._Node, env: Env, filename="tree", max_depth: int = 3, best_path: List[Tuple[int, int]] = None, is_print_action: bool = True, add_terminal_state: bool = False):
    """
    Draw a UCT tree up to a maximum depth.
    
    root: UCT_Node
    filename: output file name without extension
    max_depth: limit depth to avoid huge trees
    """
    dot = Digraph(comment='Tree')
    dot.attr(dpi='200')
    dot.attr(rankdir='TB')       # makes the tree tall instead of wide
    dot.node_attr.update({
        'fontsize': '20',
        'shape': 'box',
        'style': 'filled',
        'fillcolor': 'white',
    })
    env_clone = env.clone()
    state = env_clone.get_state()[:-1]
    env_clone.set_initial_state(state)

    # draw action history before the root
    prev_node_id = None
    history_node_ids = []
    for i, action in enumerate(state):
        hist_id = f"hist_{i}"
        hist_label = f"{action}"
        dot.node(hist_id, hist_label, style="filled", fillcolor="lightgrey", fontsize="18")
        if prev_node_id is not None:
            dot.edge(prev_node_id, hist_id)
        prev_node_id = hist_id
        history_node_ids.append(hist_id)
    # connect history to root later

    def add_node(node: node._Node, parent_id: str = None, depth: int = 0., best_path = None, is_print_action: bool = True, add_terminal_state: bool = False):
        
        if depth > max_depth:
            return
        node_id = str(id(node))
        if is_print_action:
            action = env_clone.get_action(node.action) if node.action is not None else None
        else:
            action = node.action
        label = f"Action: {action}\nPlayer: {node.player}\nVisits: {node.visits}\nValue: {node.value:.2f}"
        
        is_along_best_path = best_path is not None and action == best_path[0]
        color = "red" if is_along_best_path else "black"
        dot.node(node_id, label, style="filled", penwidth="2", color = color, fillcolor="white")
        if parent_id is not None:
            dot.edge(parent_id, node_id)
        if node.children:
            for child in node.children:
                next_path = best_path[1:] if is_along_best_path else None
                add_node(child, node_id, depth + 1, best_path=next_path, add_terminal_state=add_terminal_state)
        elif add_terminal_state:
            env_clone.update(node.state)
            label = env_clone.state.to_string()
            terminal_id = f"{node_id}_terminal"
            dot.node(terminal_id, label, style="filled", fillcolor="lightgrey")
            dot.edge(node_id, terminal_id)
            
    add_node(root, best_path=best_path, is_print_action=is_print_action, add_terminal_state=add_terminal_state)
    
    # Connect last history action with root
    if history_node_ids:
        dot.edge(history_node_ids[-1], str(id(root)))
    
    dot.render(filename, format="pdf", cleanup=True)
    print(f"Tree saved as {filename}.pdf")

def draw_tree_bot(root: node._Node, env: Env, filename="tree", max_depth: int = 3, best_path: List[Tuple[int, int]] = None, is_print_action: bool = True, add_terminal_state: bool = False):
    """
    Draw a UCT tree up to a maximum depth.
    
    root: UCT_Node
    filename: output file name without extension
    max_depth: limit depth to avoid huge trees
    """
    dot = Digraph(comment='Tree')
    dot.attr(dpi='200')
    dot.attr(rankdir='TB')       # makes the tree tall instead of wide
    dot.node_attr.update({
        'fontsize': '20',
        'shape': 'box',
        'style': 'filled',
        'fillcolor': 'white',
    })
    env_clone = env.clone()
    state = env_clone.get_state()[:-1]
    env_clone.set_initial_state(state)

    # draw action history before the root
    prev_node_id = None
    history_node_ids = []
    for i, action in enumerate(state):
        hist_id = f"hist_{i}"
        hist_label = f"{action}"
        dot.node(hist_id, hist_label, style="filled", fillcolor="lightgrey", fontsize="18")
        if prev_node_id is not None:
            dot.edge(prev_node_id, hist_id)
        prev_node_id = hist_id
        history_node_ids.append(hist_id)
    # connect history to root later

    def add_node(node: node._Node, parent_id: str = None, depth: int = 0., best_path = None, is_print_action: bool = True, add_terminal_state: bool = False):
        
        if depth > max_depth:
            return
        node_id = str(id(node))
        if is_print_action:
            action = env_clone.get_action(node.action) if node.action is not None else None
        else:
            action = node.action
        value = node.explore_count and node.total_reward / node.explore_count
        label = f"Action: {action}\nPlayer: {node.player}\nVisits: {node.explore_count}\nValue: {node.total_reward:.2f}"
        
        is_along_best_path = best_path is not None and action == best_path[0]
        color = "red" if is_along_best_path else "black"
        dot.node(node_id, label, style="filled", penwidth="2", color = color, fillcolor="white")
        if parent_id is not None:
            dot.edge(parent_id, node_id)
        if node.children:
            for child in node.children:
                next_path = best_path[1:] if is_along_best_path else None
                add_node(child, node_id, depth + 1, best_path=next_path, add_terminal_state=add_terminal_state)
        elif add_terminal_state:
            env_clone.update(node.state)
            label = env_clone.state.to_string()
            terminal_id = f"{node_id}_terminal"
            dot.node(terminal_id, label, style="filled", fillcolor="lightgrey")
            dot.edge(node_id, terminal_id)
            
    add_node(root, best_path=best_path, is_print_action=is_print_action, add_terminal_state=add_terminal_state)
    
    # Connect last history action with root
    if history_node_ids:
        dot.edge(history_node_ids[-1], str(id(root)))

    dot.render(filename, format="pdf", cleanup=True)
    print(f"Tree saved as {filename}.pdf")


def box_plot(data, yerrors, ylabel, labels, colors, edgecolor, figsize = (8,6), save_path = None, patterns = None):
    plt.figure(figsize=figsize)
    for i, (d, yerr, l, c, e) in enumerate(zip(data, yerrors, labels, colors, edgecolor)):
        if patterns is not None:
            plt.bar(x = l, height = d, label = l, yerr=yerr, capsize=10, color=c, edgecolor = e, lw=2., hatch = patterns[i], zorder = 0)
        else:
            plt.bar(l, d, yerr=yerr, capsize=10, color='none', label=l)

    plt.ylabel(ylabel, fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)
    plt.tight_layout()


    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved box plot to {save_path}")
    plt.show()

def hist_plot(data, colors, xlabel, ylabel, labels, edgecolor, title = None, bins = 10, figsize = (8,6), save_path = None, bin_width = 20.0, patterns = None):
    plt.figure(figsize=figsize)

    all_data = np.concatenate(data)
    if bin_width is not None:
        min_val, max_val = np.min(all_data), np.max(all_data)
        bins = np.arange(min_val, max_val + bin_width, bin_width)
    else:
        _, bins = np.histogram(all_data, bins=bins)

    plt.hist(x = data, bins=bins, color = colors, label=labels, edgecolor = edgecolor, lw=2., hatch = patterns, zorder = 0)

    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved histogram to {save_path}")

    plt.show()