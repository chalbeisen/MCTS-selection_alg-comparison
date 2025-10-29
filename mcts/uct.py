"""A simple UCT implementation used as the first algorithm port.

This is a minimal, well-commented implementation intended as a starting
point to port other algorithms from the C++ codebase.
"""

import math
import random
from typing import List, Optional

from env import SimpleEnv


class _Node:
    def __init__(self, parent: Optional["_Node"], action: Optional[int], env: SimpleEnv):
        self.parent = parent
        self.action = action
        self.env = env.clone()
        self.children: List[_Node] = []
        self.untried_actions = list(env.legal_actions)
        self.visits = 0
        self.value = 0.0
        self.state = []

    def uct_score(self, explore_const: float = math.sqrt(2.0)) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else 1
        return (self.value / self.visits) + explore_const * math.sqrt(math.log(parent_visits) / self.visits)

    def _best_child(self) -> "_Node":
        child = max(self.children, key=lambda c: c.uct_score())
        return child

    def _select_best_child(self, env: SimpleEnv) -> tuple["_Node", float]:
        child = self._best_child()
        reward = env.step(child.action)
        return child, reward

    def _create_new_child(self, action: int, env: SimpleEnv) -> "_Node":
        child = _Node(parent=self, action=action, env=env)
        self.children.append(child)
        self.untried_actions.remove(action)
        self.state = self.parent.state + [action]
        return child
    
    def _expand(self, env: SimpleEnv, rng: random.Random) -> tuple["_Node", float]:
        action = self.untried_actions.pop(0)
        reward = env.step(action)
        child = self._create_new_child(action, env)
        return child, reward


def uct_search(root_env: SimpleEnv, iterations: int = 1000, seed: Optional[int] = None) -> int:
    rng = random.Random(seed)
    root = _Node(parent=None, action=None, env=root_env)
    max_reward = 0
    best_path = []

    for _ in range(iterations):
        node = root
        env = root_env.clone()
        path = []

        while not env.is_terminal():
            if node.untried_actions == [] and node.children:
                # Selection
                node, reward = node._select_best_child(env)
            else:
                # Expansion
                node, reward = node._expand(env, rng)
            path.append(node.action)

        # Backpropagation
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
        
        if reward > max_reward:
            max_reward = reward
            best_path = path

    # return the most visited child action
    if not root.children:
        # no expansion happened; pick a legal action
        return root_env.legal_actions[0]
    return env.get_path(best_path)