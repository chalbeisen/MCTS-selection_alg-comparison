"""A simple DENTS-like MCTS: deterministic greedy selection with a
decaying temperature term that encourages exploration early and becomes
greedy later.

This minimal implementation is intended as a companion to `ments.py` for
testing and comparison.
"""

import random
from typing import List, Optional

from env import SimpleEnv


class _Node:
    def __init__(self, parent: Optional["_Node"], action: Optional[int], env: SimpleEnv):
        self.parent = parent
        self.action = action
        self.env = env.clone()
        self.children: List[_Node] = []
        self.untried_actions = env.legal_actions()
        self.visits = 0
        self.value = 0.0

    def average(self) -> float:
        return self.value / self.visits if self.visits > 0 else 0.0
    
    def _average_best_child(self, temp, env) -> "_Node":
        # pick child maximizing average + bonus/temp
        best = None
        best_score = -float("inf")
        for c in self.children:
            bonus = temp / (1.0 + c.visits)
            score = c.average() + bonus
            if score > best_score:
                best_score = score
                best = c
        return best
    
    def _select_average_best_child(self, temp: float, env: SimpleEnv) -> tuple["_Node", float]:
        child = self._average_best_child(temp, env)
        reward = env.step(child.action)
        return child, reward
    
    def _expand_random(self, env: SimpleEnv, rng: random.Random) -> tuple["_Node", float]:
        a = rng.choice(self.untried_actions)
        reward = env.step(a)
        self.untried_actions.remove(a)
        child = _Node(parent=self, action=a, env=env)
        self.children.append(child)
        return child, reward


def dents_search(root_env: SimpleEnv, iterations: int = 1000, start_temp: float = 1.0, decay: float = 0.001, seed: Optional[int] = None) -> int:
    rng = random.Random(seed)
    root = _Node(parent=None, action=None, env=root_env)
    max_reward = 0
    best_path = []

    for it in range(iterations):
        node = root
        env = root_env.clone()
        path = []

        while not env.is_terminal():
            if node.untried_actions == [] and node.children:
                # Selection
                temp = start_temp / (1.0 + decay * it)
                node._select_average_best_child(temp, env)
            else:
                # Expansion
                node, reward = node._expand_random(env, rng)
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
