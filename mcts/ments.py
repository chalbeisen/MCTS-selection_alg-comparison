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

class _Node:
    def __init__(self, parent: Optional["_Node"], action: Optional[int], env: SimpleEnv):
        self.parent = parent
        self.action = action
        self.env = env.clone()
        self.children: List[_Node] = []
        self.untried_actions = list(env.legal_actions)
        self.valid_actions = list(env.legal_actions)
        self.visits = 0
        self.Qsft = 0
        self.state = []
    
    def _exp_average_best_child(self, tempscale: float, env: SimpleEnv, rng:int) -> "_Node":
        weights = []
        for c in self.children:
            avg = c.average()
            # exponentiate scaled average (shift to positive)
            weights.append(math.exp(avg / max(1e-6, tempscale)))
        idx = _softmax(weights, rng)
        return self.children[idx]
    
    def _select_exp_average_best_child(self, tempscale: float, env: SimpleEnv, rng:int) -> tuple["_Node", float]:
        child = self._exp_average_best_child(tempscale, env, rng)
        reward = env.step(child.action)
        return child, reward

    def _expand_random(self, env: SimpleEnv, rng: random.Random) -> tuple["_Node", float]:
        a = rng.choice(self.untried_actions)
        reward = env.step(a)
        self.untried_actions.remove(a)
        child = _Node(parent=self, action=a, env=env)
        self.children.append(child)
        return child, reward
    
    def _softmax_policy(self, temp: float, epsilon: float) -> tuple[List[float], List[int]]:
        # Soft indmax (Boltzmann) policy f_tau(Q)
        Qsft_children = [child.Qsft for child in self.children] + (len(self.valid_actions) - len(self.children)) * [0]
        soft_probs = f_tau(Qsft_children, temp)

        # Add uniform mixing for exploration
        visit_count = sum([child.visits for child in self.children])
        lamb = epsilon * len(Qsft_children) / np.log(visit_count + 2)
        lamb = min(max(lamb, 0.0), 1.0)  # clamp to [0,1]
        uniform = np.ones_like(soft_probs) / len(Qsft_children)

        # Mixed exploration policy π_t(a|s)
        mixed_policy = (1 - lamb) * soft_probs + lamb * uniform
        return mixed_policy
    
    def _select_action(self, temp: float, epsilon: float) -> int:
        probs = self._softmax_policy(temp, epsilon)
        return int(np.random.choice(self.valid_actions, p=probs))
    
    def _create_new_child(self, action: int, env: SimpleEnv) -> "_Node":
        child = _Node(parent=self, action=action, env=env)
        self.children.append(child)
        self.untried_actions.remove(action)
        self.state = self.parent.state + [action]
        return child
    
    def _get_node_by_action(self, action: int) -> Optional["_Node"]: 
        return next((child for child in self.children if child.action == action), None)

    def _select_child(self, env: SimpleEnv, temp:float, epsilon: float) -> tuple["_Node", float]:
        action = self._select_action(temp, epsilon)
        reward = env.step(action)
        selected_child = self._get_node_by_action(action)
        if selected_child is None:
            selected_child = self._create_new_child(action, env)
        return selected_child, reward

    def _softmax_backup(self, temp: float, reward: float, node: "_Node"):
        self.Qsft = reward
        node = node.parent
        while node.parent:
            child_values = [child.Qsft for child in node.children]
            node.Qsft = F_tau(child_values, temp)
            node = node.parent
                

def ments_search(root_env: SimpleEnv, iterations: int = 1000, base_temp: float = 1.0, decay: float = 0.01, epsilon: float = 0.2, seed: Optional[int] = None) -> int:
    np.random.seed(seed)
    root = _Node(parent=None, action=None, env=root_env)
    max_reward = 0
    best_path = []

    for it in range(iterations):
        node = root
        env = root_env.clone()
        path = []

        while not env.is_terminal():
            # Selection
            tempscale = base_temp / (1.0 + decay * node.visits)
            node, reward = node._select_child(env, tempscale, epsilon)
            path.append(node)

        # Backpropagation
        node._softmax_backup(tempscale, reward, node)
        
        if reward > max_reward:
            max_reward = reward
            best_path = path

    # return the most visited child action
    if not root.children:
        # no expansion happened; pick a legal action
        return root_env.legal_actions[0]
    return env.get_path(best_path)
