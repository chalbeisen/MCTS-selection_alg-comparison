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

def entropy(probs: List[float]):
    probs = np.array(probs, dtype=float)
    probs = probs / (probs.sum() + 1e-8)
    return -np.sum(probs * np.log(probs + 1e-8))

def beta(N: int, base_beta: float = 1.0, decay: float = 0.05):
    return base_beta / (1.0 + decay * N)

def f_tau(r, tau):
    r = np.array(r, dtype=float)
    r_max = r.max()
    exp_r = np.exp((r - r_max) / tau)
    return exp_r / exp_r.sum()


def F_tau(r, tau) -> float: 
    if len(r) == 0: 
        return 0.0 
    r = np.array(r, dtype=float)
    return float(tau * np.log(sum(np.exp(r / tau))))

class Dents_Node(_Node):
    def __init__(self, parent: Optional["_Node"], action: Optional[int], untried_actions: List[int]):
        super().__init__(parent, action, untried_actions)
        self.Qsft = 0
        self.HQ = 0
    
    def get_entropy_values(self, N):
        return 0.0

    def _boltzmann_policy(self, temp: float, epsilon: float) -> tuple[List[float], List[int]]:
        # Soft indmax (Boltzmann) policy f_tau(Q)
        Qsft_children = [child.Qsft for child in self.children] + len(self.untried_actions) * [0]
        if len(Qsft_children) == 0:  # no actions available
            return [], []
        if (-np.inf) in Qsft_children:
            print("test")
        actions = [child.action for child in self.children] + self.untried_actions
        # effective Q with entropy bonus
        H_bonus = [child.HQ for child in self.children] + len(self.untried_actions) * [0]

        visit_count = sum([child.visits for child in self.children])
        b = beta(visit_count)
        Q_eff = [q + b * h for q, h in zip(Qsft_children, H_bonus)]
        probs = f_tau(Q_eff, temp)

        lamb = epsilon * len(Qsft_children) / np.log(visit_count + 2)
        lamb = min(max(lamb, 0.0), 1.0)  # clamp to [0,1]
        uniform = np.ones_like(probs) / len(Qsft_children)

        # Mixed exploration policy π_t(a|s)
        mixed_policy = (1 - lamb) * probs + lamb * uniform
        return mixed_policy, actions
    
    def _create_new_child(self, action: int, env: SimpleEnv) -> "_Node":
        untried_actions = self.update_untried_actions(action, env)
        child = Dents_Node(parent=self, action=action, untried_actions=untried_actions)
        self.children.append(child)
        child.state = child.parent.state + [action] if child.parent else [action]
        return child
    
    def _select_action(self, temp: float, epsilon: float) -> int:
        probs, actions = self._boltzmann_policy(temp, epsilon)
        return int(np.random.choice(actions, p=probs))

    def _select_child(self, env: SimpleEnv, temp:float, epsilon: float) -> tuple["_Node", float]:
        action = self._select_action(temp, epsilon)
        reward = env.step(action)
        selected_child = self._get_node_by_action(action)
        if selected_child is None:
            selected_child = self._create_new_child(action, env)
        return selected_child, reward

    def _backpropagate(self, temp: float, reward: float, node: "_Node", epsilon: float):
        self.Qsft = reward
        node = node.parent
        while node.parent:
            child_values = [child.Qsft for child in node.children]
            node.Qsft = max(child_values)
            pi, _ = self._boltzmann_policy(temp, epsilon)
            node.HQ = entropy(pi)
            node = node.parent
                

def dents_search(root_env: SimpleEnv, iterations: int = 1000, base_temp: float = 100, decay: float = 0.05, epsilon: float = 0.2, seed: Optional[int] = None) -> int:
    root = Dents_Node(parent=None, action=None, untried_actions=list(root_env.legal_actions))
    max_reward = -np.inf
    best_path = []
    best_iteration = 0

    for i in range(iterations):
        node = root
        env = root_env.clone()
        path = []

        while not env.is_terminal():
            # Selection
            tempscale = base_temp / (1.0 + decay * node.visits)
            node, reward = node._select_child(env, tempscale, epsilon)
            path.append(node.action)

        # Backpropagation
        node._backpropagate(tempscale, reward, node, epsilon)
        
        if reward > max_reward:
            max_reward = reward
            best_path = path
            best_iteration = i

    return env.get_items_in_path(best_path), np.abs(max_reward), best_iteration