"""A simple UCT implementation used as the first algorithm port.

This is a minimal, well-commented implementation intended as a starting
point to port other algorithms from the C++ codebase.
"""

import math
import numpy as np
import random
from typing import List, Optional

from env import SimpleEnv
from node import _Node


class UCT_Node(_Node):
    def __init__(self, parent: Optional["_Node"], action: Optional[int], untried_actions: List[int]):
        super().__init__(parent, action, untried_actions)

    def _best_child(self) -> "_Node":
        child = max(self.children, key=lambda c: c.uct_score())
        return child

    def _select_best_child(self, env: SimpleEnv) -> tuple["_Node", float]:
        child = self._best_child()
        reward = env.step(child.action)
        return child, reward

    def _create_new_child(self, action: int, env: SimpleEnv) -> "_Node":
        untried_actions = self.update_untried_actions(action, env)
        child = UCT_Node(parent=self, action=action, untried_actions=untried_actions)
        self.children.append(child)
        child.state = child.parent.state + [action] if child.parent else [action]
        return child


def uct_search(root_env: SimpleEnv, iterations: int = 1000, seed: Optional[int] = None) -> int:
    root = UCT_Node(parent=None, action=None, untried_actions=list(root_env.legal_actions))
    max_reward = -np.inf
    best_path = []
    best_iteration = 0

    for i in range(iterations):
        node = root
        env = root_env.clone()
        path = []

        while not env.is_terminal():
            if node.untried_actions == [] and node.children:
                # Selection
                node, reward = node._select_best_child(env)
            else:
                # Expansion
                node, reward = node._expand(env)
            path.append(node.action)

        # Backpropagation
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
        
        if reward > max_reward:
            max_reward = reward
            best_path = path
            best_iteration = i

    return env.get_items_in_path(best_path), np.abs(max_reward), best_iteration