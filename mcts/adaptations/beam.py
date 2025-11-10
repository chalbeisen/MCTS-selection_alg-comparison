import math
import numpy as np
import random
from typing import List, Optional, Dict

from env.env import SimpleEnv   # or your TSPEnv
from mcts.node import _Node


class BeamNode(_Node):
    """BMCTS node identical to _Node but tracks depth."""
    def __init__(self, parent: Optional["_Node"], action: Optional[int], untried_actions: List[int]):
        super().__init__(parent, action, untried_actions)
        self.depth = 0 if parent is None else parent.depth + 1

    def _best_child(self) -> "_Node":
        child = max(self.children, key=lambda c: c.uct_score())
        return child

    def _select_best_child(self, env: SimpleEnv) -> tuple["_Node", float]:
        child = self._best_child()
        reward = env.step(child.action)
        return child, reward
    
    def _create_new_child(self, action: int, env: SimpleEnv) -> "_Node":
        untried_actions = self.update_untried_actions(action, env)
        child = BeamNode(parent=self, action=action, untried_actions=untried_actions)
        self.children.append(child)
        child.state = child.parent.state + [action] if child.parent else [action]
        return child


def beam_mcts_search(root_env: SimpleEnv, iterations: int = 1000, beam_width: int = 20, sim_limit: int = 100, seed: Optional[int] = None) -> tuple[list[int], float, int]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    root = BeamNode(parent=None, action=None, untried_actions=list(root_env.legal_actions))
    depth_counter: Dict[int, int] = {}
    best_path: list[int] = []
    best_reward = -np.inf
    best_iteration = 0

    for i in range(iterations):
        node = root
        env = root_env.clone()
        path: list[int] = []

        while not env.is_terminal():
            # count how many rollouts reached this depth
            depth_counter[node.depth] = depth_counter.get(node.depth, 0) + 1

            if node.untried_actions == [] and node.children:
                # Selection by UCT
                node, reward = node._select_best_child(env)
            else:
                # Expansion
                node, reward = node._expand(env)

            path.append(node.action)
        depth_counter[node.depth] = depth_counter.get(node.depth, 0) + 1

        # Beam pruning when limit reached
        if depth_counter[node.depth] >= sim_limit:
            prune_tree(root, node.depth, beam_width)
            depth_counter[node.depth] = 0


        # Backpropagation
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

        if reward > best_reward:
            best_reward = reward
            best_path = path.copy()
            best_iteration = i

    return root, env.get_items_in_path(best_path), abs(best_reward), best_iteration


def prune_tree(root: BeamNode, depth: int, beam_width: int):
    """
    Prune the search tree at a given depth:
    Keep only top 'beam_width' nodes by visit count + their ancestors.
    """
    # Gather all nodes at this depth
    level_nodes: list[BeamNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.depth == depth:
            level_nodes.append(node)
        stack.extend(node.children)

    if not level_nodes:
        return

    # Select top W nodes by visit count
    survivors = sorted(level_nodes, key=lambda n: n.visits, reverse=True)[:beam_width]
    survivor_set = set(survivors)

    # Include their ancestors so the tree stays connected
    for n in survivors:
        while n is not None:
            survivor_set.add(n)
            n = n.parent

    # Recursively prune children not in survivor set
    def prune_children(n: BeamNode):
        n.children = [c for c in n.children if c in survivor_set]
        for c in n.children:
            prune_children(c)

    prune_children(root)
