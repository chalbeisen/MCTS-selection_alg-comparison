"""A simple MENTS-like MCTS: uses Boltzmann (softmax) selection over child
average-values with a decaying temperature.

This is a minimal, illustrative implementation to act as a starting point for
porting the C++ `ments` algorithm.
"""

import math
import numpy as np
import random
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

def f_tau(values, tau):
    """
    Soft indmax (Boltzmann distribution).
    f_tau(r)(a) = exp(r(a)/tau) / sum_b exp(r(b)/tau)
    Returns a probability distribution over actions.
    """
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return np.array([])
    exp_vals = np.exp(values / tau)
    return exp_vals / np.sum(exp_vals)

def F_tau(values, tau): 
    """Softmax value: Fτ(r) = τ * log(Σ exp(r/τ)).""" 
    values = np.array(values, dtype=float) 
    if len(values) == 0: 
        return 0.0 
    return tau * np.log(np.sum(np.exp(values / tau)))

class _Node:
    def __init__(self, parent: Optional["_Node"], action: Optional[int], env: SimpleEnv):
        self.parent = parent
        self.action = action
        self.env = env.clone()
        self.children: List[_Node] = []
        self.untried_actions = list(env.legal_actions)
        self.visits = 0
        self.value = 0.0
        self.Qsft = dict.fromkeys(self.untried_actions, 0)

    def average(self) -> float:
        return self.value / self.visits if self.visits > 0 else 0.0
    
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
    
    def softmax_policy(self, tau, epsilon):
        actions = list(self.Qsft.keys())
        q_values = np.array([self.Qsft[a] for a in actions])

        # Soft indmax (Boltzmann) policy f_tau(Q)
        soft_probs = f_tau(q_values, tau)

        # Add uniform mixing for exploration
        lamb = epsilon * len(actions) / math.log(sum(self.visits) + 2)
        lamb = min(max(lamb, 0.0), 1.0)  # clamp to [0,1]
        uniform = np.ones_like(soft_probs) / len(actions)

        # Mixed exploration policy π_t(a|s)
        mixed_policy = (1 - lamb) * soft_probs + lamb * uniform
        return mixed_policy, actions
    
    def select_action(self, tau, epsilon):
        probs, actions = self.softmax_policy(tau, epsilon)
        return np.random.choice(actions, p=probs)
    

    def softmax_backup(self, reward):
        child = self
        while child.parent is not None:
            node = child.parent
            if child.Qsft:
                soft_value = F_tau(list(child.Qsft.values()), self.tau)
            else:
                soft_value = 0.0

            target = reward + soft_value
            node.visits += 1
            node.Qsft[node.action] += (target - node.Qsft[child.action]) / node.visits
            child = node
            

def ments_search(root_env: SimpleEnv, iterations: int = 1000, base_temp: float = 1.0, decay: float = 0.01, seed: Optional[int] = None) -> int:
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
                tempscale = base_temp / (1.0 + decay * node.visits)
                node, reward = node._select_exp_average_best_child(tempscale, env, rng)
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
