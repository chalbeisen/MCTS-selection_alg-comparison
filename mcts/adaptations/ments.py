import math
import random
import numpy as np
from typing import List, Optional

from env.env import Env
from mcts.node import _Node

def f_tau(r, tau):
    r = np.array(r, dtype=float)
    if len(r) == 0:
        return np.array([])
    r_max = np.max(r)
    exp_r = np.exp((r - r_max) / (tau + 1e-8))
    return exp_r / (np.sum(exp_r) + 1e-8)

def F_tau(r, tau) -> float:
    if len(r) == 0:
        return 0.0
    r = np.array(r, dtype=float)
    r_max = np.max(r)
    return float(r_max + tau * np.log(np.sum(np.exp((r - r_max) / (tau + 1e-8)))))

class Ments_Node(_Node):
    def __init__(self, parent: Optional["_Node"], action: Optional[int], untried_actions: List[int]):
        super().__init__(parent, action, untried_actions)
        self.Qsft = 0
    
    def _softmax_policy(self, temp: float, epsilon: float) -> tuple[List[float], List[int]]:
        # Soft indmax (Boltzmann) policy f_tau(Q)
        Qsft_children = [child.Qsft for child in self.children] + len(self.untried_actions) * [0]
        if len(Qsft_children) == 0:  # no actions available
            return [], []
        actions = [child.action for child in self.children] + self.untried_actions
        soft_probs = f_tau(Qsft_children, temp)

        # Add uniform mixing for exploration
        visit_count = sum([child.visits for child in self.children])
        lamb = epsilon * len(Qsft_children) / np.log(visit_count + 2)
        lamb = min(max(lamb, 0.0), 1.0)  # clamp to [0,1]
        #lamb = min(1.0, epsilon / max(1.0, np.log(np.e + self.visits)))
        uniform = np.ones_like(soft_probs) / len(Qsft_children)

        # Mixed exploration policy π_t(a|s)
        mixed_policy = (1 - lamb) * soft_probs + lamb * uniform
        return mixed_policy, actions
    
    def _create_new_child(self, action: int, env: Env) -> "_Node":
        untried_actions = self.update_untried_actions(action, env)
        child = Ments_Node(parent=self, action=action, untried_actions=untried_actions)
        self.children.append(child)
        child.state = child.parent.state + [action] if child.parent else [action]
        return child
    
    def _select_action(self, temp: float, epsilon: float) -> int:
        probs, actions = self._softmax_policy(temp, epsilon)
        return int(np.random.choice(actions, p=probs))

    def _select_child(self, env: Env, temp:float, epsilon: float) -> tuple["_Node", float]:
        action = self._select_action(temp, epsilon)
        reward = env.step(action)
        selected_child = self._get_node_by_action(action)
        if selected_child is None:
            selected_child = self._create_new_child(action, env)
        return selected_child, reward

    def _backpropagate_v2(self, reward: float, node: "_Node"):
        """Hard (max) Bellman backup for sparse-reward environments."""
        cur = node
        while cur is not None:
            if cur.children:
                # Hard Bellman value update
                child_qs = [child.q_value for child in cur.children]
                cur.value = max(child_qs)
            else:
                # Leaf: its value is the (possibly nonzero) terminal reward
                cur.value = reward  # or rollout result

            # Sparse reward: most of the time cur.reward == 0
            cur.q_value = reward + cur.value
            cur.visits += 1
            cur.edge_reward = reward
            cur = cur.parent

    def _backpropagate(self, temp: float, reward: float, node: "_Node"):
        node.Qsft = reward
        cur = node.parent
        while cur is not None:
            child_values = [child.Qsft for child in cur.children]
            cur.Qsft = F_tau(child_values, temp)
            cur.visits += 1
            cur.edge_reward = reward
            cur = cur.parent

    def _backpropagate_stochastic(self, temp: float, reward: float, node: "_Node"):
        """
        Backpropagate reward in a stochastic environment for MENTS.
        node.Qsft stores the mean Q-value estimate for each node.
        """
        # Update this node's mean Q-value
        node.visits += 1
        node.Qsft = (node.Qsft * (node.visits - 1) + reward) / node.visits

        cur = node.parent
        while cur is not None:
            cur.visits += 1  # increment parent visit

            # Compute soft-max (F_tau) over child Q-values
            child_values = [child.Qsft for child in cur.children]
            cur.Qsft = F_tau(child_values, temp) if child_values else 0.0
            cur.edge_reward = reward
            cur = cur.parent

                

def ments_search(root_env: Env, iterations: int = 1000, base_temp: float = 1000, decay: float = 0.05, epsilon: float = 1.0, seed: Optional[int] = None) -> int:
    root = Ments_Node(parent=None, action=None, untried_actions=root_env.get_legal_actions(None))
    max_reward = -np.inf
    best_path = []
    best_iteration = 0
    path_over_iter = []

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
        node._backpropagate_v2(reward, node)
        
        if reward > max_reward:
            max_reward = reward
            best_path = path.copy()
            best_iteration = i
        path_over_iter.append(env.get_items_in_path(best_path))

    return root, env.get_items_in_path(best_path), abs(max_reward), best_iteration, path_over_iter