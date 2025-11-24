"""
UCT with First-Play Urgency (FPU)

Extends basic UCT by assigning a finite FPU value instead of ∞ for unexplored moves.
"""

import math
import random
import numpy as np
from typing import List, Optional

from env.env import Env
from mcts.node import _Node


class UCTNodeFPU(_Node):
    def __init__(
        self,
        parent: Optional["_Node"],
        action: Optional[int],
        untried_actions: List[int],
        fpu: float = 0.5,
    ):
        super().__init__(parent, action, untried_actions)
        self.fpu = fpu  # First-play urgency

    def uct_score(self, explore_const: float = math.sqrt(2.0)) -> float:
        if self.visits == 0:
            return self.fpu  # Use FPU instead of infinity
        parent_visits = self.parent.visits if self.parent else 1
        return self.value + explore_const * math.sqrt(math.log(parent_visits + 1) / self.visits)

    def _best_child(self, explore_const: float = math.sqrt(2.0)) -> "_Node":
        return max(self.children, key=lambda c: c.uct_score(explore_const))

    def _select_best_child(self, env: Env) -> tuple["_Node", float]:
        child = self._best_child()
        reward = env.step(child.action)
        return child, reward

    def _create_new_child(self, action: int, env: Env) -> "_Node":
        untried_actions = self.update_untried_actions(action, env)
        child = UCTNodeFPU(self, action, untried_actions, fpu=self.fpu)
        self.children.append(child)
        child.state = child.parent.state + [action] if child.parent else [action]
        return child

    def _backpropagate(self, reward: float) -> None:
        self.visits += 1
        self.edge_reward = reward
        if self.value is None:
            self.value = reward
        else:
            self.value = (self.value * (self.visits - 1) + reward) / self.visits
        if self.parent:
            self.parent._backpropagate(reward)
        return


def uct_search_fpu(
    root_env: Env,
    iterations: int = 1000,
    fpu: float = 0.5,
    seed: Optional[int] = None,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    root = UCTNodeFPU(parent=None, action=None, untried_actions=root_env.get_legal_actions(), fpu=fpu)

    max_reward = -np.inf
    best_path = []
    best_iteration = 0
    path_over_iter = []

    for i in range(iterations):
        node = root
        env = root_env.clone()
        path = []

        while not env.is_terminal():
            if node.untried_actions == [] and node.children:
                node, reward = node._select_best_child(env)
            else:
                node, reward = node._expand(env)
            path.append(node.action)

        node._backpropagate(reward)

        if reward > max_reward:
            max_reward = reward
            best_path = path.copy()
            best_iteration = i
        path_over_iter.append(env.get_items_in_path(best_path))

    return root, env.get_items_in_path(best_path), abs(max_reward), best_iteration, path_over_iter
