"""
MCTS + RAVE implementation keeping the same structure as the UCT code.
"""

import math
import numpy as np
import random
from typing import List, Optional

from env.env import SimpleEnv
from mcts.node import _Node


class RAVE_Node(_Node):
    def __init__(self, parent: Optional["_Node"], action: Optional[int], untried_actions: List[int]):
        super().__init__(parent, action, untried_actions)
        self.rave_counts = {}  # count of each action in rollouts
        self.rave_values = {}  # value estimates from RAVE

    def _best_child(self, alpha: float = 0.5) -> "_Node":
        """
        Compute combined score: standard UCT + RAVE
        alpha: weight of RAVE vs UCT (0=only UCT, 1=only RAVE)
        """
        def combined_score(child: "_Node") -> float:
            # UCT score
            uct = child.uct_score()
            # RAVE estimate
            rave_val = self.rave_values.get(child.action, 0.0)
            return (1 - alpha) * uct + alpha * rave_val

        return max(self.children, key=combined_score)

    def _select_best_child(self, env: SimpleEnv, alpha: float = 0.5) -> tuple["_Node", float]:
        child = self._best_child(alpha)
        reward = env.step(child.action)
        return child, reward

    def _create_new_child(self, action: int, env: SimpleEnv) -> "_Node":
        untried_actions = self.update_untried_actions(action, env)
        child = RAVE_Node(parent=self, action=action, untried_actions=untried_actions)
        self.children.append(child)
        child.state = self.state + [action] if self.state else [action]
        return child

    def _expand(self, env: SimpleEnv) -> tuple["_Node", float]:
        action = self.untried_actions[0]
        reward = env.step(action)
        child = self._create_new_child(action, env)
        self.untried_actions.remove(action)
        return child, reward

    def _backpropagate(self, reward: float, path_actions: List[int], alpha: float = 0.5):
        """
        Backpropagate reward along the tree and update RAVE statistics.
        path_actions: all actions taken in this rollout
        """
        node = self
        while node is not None:
            node.visits += 1
            node.value += reward

            # Update RAVE statistics
            for a in path_actions:
                node.rave_counts[a] = node.rave_counts.get(a, 0) + 1
                prev_val = node.rave_values.get(a, 0.0)
                node.rave_values[a] = prev_val + (reward - prev_val) / node.rave_counts[a]

            node = node.parent


def rave_search(root_env: SimpleEnv, iterations: int = 500, alpha: float = 0.5, seed: Optional[int] = None):
    root = RAVE_Node(parent=None, action=None, untried_actions=list(root_env.legal_actions))
    max_reward = -np.inf
    best_path = []
    best_iteration = 0

    for i in range(iterations):
        node = root
        env = root_env.clone()
        path = []

        # Selection & Expansion
        while not env.is_terminal():
            if node.untried_actions == [] and node.children:
                node, reward = node._select_best_child(env, alpha)
            else:
                node, reward = node._expand(env)
            path.append(node.action)

        # Backpropagation
        node._backpropagate(reward, path, alpha)

        if reward > max_reward:
            max_reward = reward
            best_path = path
