"""A simple MENTS-like MCTS: uses Boltzmann (softmax) selection over child
average-values with a decaying temperature.

This is a minimal, illustrative implementation to act as a starting point for
porting the C++ `ments` algorithm.
"""

import math
import random
import numpy as np
from typing import List, Optional

from env import SimpleEnv
from node import _Node


def _softmax(weights, rng: random.Random):
    total = sum(weights)
    if total <= 0:
        # fallback to uniform
        return rng.randrange(len(weights))
    r = rng.random() * total
    upto = 0.0
    for i, w in enumerate(weights):
        upto += w
        if r <= upto:
            return i
    return len(weights) - 1

def f_tau(r, tau):
    if len(r) == 0:
        return 0.0
    r = np.array(r, dtype=float)
    return np.exp((r - F_tau(r, tau)) / tau)

def F_tau(r, tau) -> float: 
    if len(r) == 0: 
        return 0.0 
    r = np.array(r, dtype=float)
    return float(tau * np.log(sum(np.exp(r / tau))))

class Ments_Node(_Node):
    def __init__(self, parent: Optional["_Node"], action: Optional[int], untried_actions: List[int]):
        super().__init__(parent, action, untried_actions)
        self.Qsft = 0
    
    def _softmax_policy(self, temp: float, epsilon: float) -> tuple[List[float], List[int]]:
        # Soft indmax (Boltzmann) policy f_tau(Q)
        Qsft_children = [child.Qsft for child in self.children] + len(self.untried_actions) * [0]
        actions = [child.action for child in self.children] + self.untried_actions
        soft_probs = f_tau(Qsft_children, temp)

        # Add uniform mixing for exploration
        visit_count = sum([child.visits for child in self.children])
        lamb = epsilon * len(Qsft_children) / np.log(visit_count + 2)
        lamb = min(max(lamb, 0.0), 1.0)  # clamp to [0,1]
        uniform = np.ones_like(soft_probs) / len(Qsft_children)

        # Mixed exploration policy π_t(a|s)
        mixed_policy = (1 - lamb) * soft_probs + lamb * uniform
        return mixed_policy, actions
    
    def _create_new_child(self, action: int, env: SimpleEnv) -> "_Node":
        untried_actions = self.update_untried_actions(action, env)
        child = Ments_Node(parent=self, action=action, untried_actions=untried_actions)
        self.children.append(child)
        child.state = child.parent.state + [action] if child.parent else [action]
        return child
    
    def _select_action(self, temp: float, epsilon: float) -> int:
        probs, actions = self._softmax_policy(temp, epsilon)
        return int(np.random.choice(actions, p=probs))

    def _select_child(self, env: SimpleEnv, temp:float, epsilon: float) -> tuple["_Node", float]:
        action = self._select_action(temp, epsilon)
        reward = env.step(action)
        selected_child = self._get_node_by_action(action)
        if selected_child is None:
            selected_child = self._create_new_child(action, env)
        return selected_child, reward

    def _backpropagate(self, temp: float, reward: float, node: "_Node"):
        self.Qsft = reward
        node = node.parent
        while node.parent:
            child_values = [child.Qsft for child in node.children]
            node.Qsft = F_tau(child_values, temp)
            node = node.parent
                

def ments_search(root_env: SimpleEnv, iterations: int = 1000, base_temp: float = 1.0, decay: float = 0.01, epsilon: float = 0.2, seed: Optional[int] = None) -> int:
    np.random.seed(seed)
    root = Ments_Node(parent=None, action=None, untried_actions=list(root_env.legal_actions))
    max_reward = -np.inf
    best_path = []

    for it in range(iterations):
        node = root
        env = root_env.clone()
        path = []

        while not env.is_terminal():
            # Selection
            tempscale = base_temp / (1.0 + decay * node.visits)
            node, reward = node._select_child(env, tempscale, epsilon)
            path.append(node.action)

        # Backpropagation
        node._backpropagate(tempscale, reward, node)
        
        if reward > max_reward:
            max_reward = reward
            best_path = path

    # return the most visited child action
    if not root.children:
        # no expansion happened; pick a legal action
        return root_env.legal_actions[0]
    return env.get_items_in_path(best_path)
