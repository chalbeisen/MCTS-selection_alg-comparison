import math
import random
import torch
import numpy as np
from typing import List, Optional
from adaptations.sampling import uct_distribution, cap_distribution
from mcts.node import _Node

from env.env import Env
from copy import copy


class MMCTS_Node(_Node):
    def __init__(self, parent: Optional["_Node"], action: Optional[int], untried_actions: List[int]):
        super().__init__(parent, action, untried_actions)

    def _create_new_child(self, action: int, env: Env) -> "_Node":
        untried_actions = self.update_untried_actions(action, env)
        child = MMCTS_Node(parent=self, action=action, untried_actions=untried_actions)
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

def mmcts_search(root_env: Env, iterations: int = 1000, uct_inf_softening: float = 2, base_temp: float = 1000, decay: float = 0.05, p_max: float = 1.5, seed: Optional[int] = None) -> int:
    root = MMCTS_Node(parent=None, action=None, untried_actions=root_env.get_legal_actions(None))
    best_path = []
    best_iteration = 0
    path_over_iter = []

    node = root
    env = root_env.clone()
    while not env.is_terminal():
        node, state_reward = node._expand(env)
    current_state = node.state
    original_path = copy(node.state)
    max_reward = state_reward
    best_path = original_path
    node._backpropagate(state_reward)
    path_over_iter.append(env.get_items_in_path(best_path))
    
    for i in range(iterations):
        node = root
        env = root_env.clone()

        for j in range(len(original_path)):
            legal_mutation_actions = node.untried_actions
            node_uct_values = torch.tensor([child.uct_score() for child in node.children] + [float("inf")] * len(legal_mutation_actions))
            candidates = [child.action for child in node.children] + legal_mutation_actions
            distribution = uct_distribution(node_uct_values, uct_inf_softening)
            distribution = cap_distribution(distribution, (1+p_max)/len(candidates))
            candidate_index = torch.multinomial(distribution, 1).item()
            child_action = candidates[candidate_index]

            proposed_path, reward = swap(current_state, j, child_action, env)
            tempscale = base_temp / (1.0 + decay * node.visits)

            node = go_to_state(proposed_path, root, env)
            # Backpropagation
            node._backpropagate(reward)
            ratio = _exp_ratio(state_reward, reward, temperature=tempscale)

            accept_prob = min(1, ratio)
            if torch.rand(1) < accept_prob:
                state_reward = reward
                current_state = proposed_path
            node = go_to_state(current_state[:j+1], root, env)
            env.update(node.state)
        
            if reward > max_reward:
                max_reward = reward
                best_path = proposed_path
                best_iteration = i

        path_over_iter.append(env.get_items_in_path(best_path))

    return root, env.get_items_in_path(best_path), abs(max_reward), best_iteration, path_over_iter

def mutate(original_path: List[int], mutate_index: int, mutate_value: float, env: Env):
    path = copy(original_path)
    path[mutate_index] = mutate_value
    reward = env.update(path)
    return path, reward

def swap(original_path: List[int], new_index: int, new_value: float, env: Env):
    path = copy(original_path)
    if len(original_path) > 1: 
        old_index = next(i for i in range(len(original_path)) if original_path[i] == new_value)
        path[old_index], path[new_index] = path[new_index], path[old_index]
    else:
        path[new_index] = new_value
    reward = env.update(path)
    return path, reward

def _exp_ratio(state_reward: float, proposal_reward: float, temperature: float = 1) -> float:
    exponent = -(state_reward - proposal_reward) / temperature
    exponent = max(min(exponent, 700), -700)  # clamp to safe range for exp
    return math.exp(exponent)

def go_to_state(state: List[int], root: "_Node", env: Env):
    node = root
    for action in state:
        if action not in [n.action for n in node.children]:
            node = node._create_new_child(action, env)
        else:
            node = node._get_node_by_action(action)
    return node