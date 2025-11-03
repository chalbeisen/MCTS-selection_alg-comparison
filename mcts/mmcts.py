"""A simple UCT implementation used as the first algorithm port.

This is a minimal, well-commented implementation intended as a starting
point to port other algorithms from the C++ codebase.
"""

import math
import random
import torch
from typing import List, Optional
from sampling import uct_distribution, cap_distribution

from env import SimpleEnv
from copy import copy

def mutate(original_path, mutate_index, mutate_value, env):
    path = copy(original_path)
    path[mutate_index] = mutate_value
    reward = env.update(path)
    return path, reward

def swap(original_path, new_index, new_value, env):
    path = copy(original_path)
    old_index = next(i for i in range(len(original_path)) if original_path[i] == new_value)
    path[old_index], path[new_index] = path[new_index], path[old_index]
    reward = env.update(path)
    return path, reward
    

def _exp_ratio(state_reward: float, proposal_reward: float, temperature: float = 1) -> float:
    return math.exp(-(state_reward - proposal_reward) / temperature)

class _Node:
    def __init__(self, parent: Optional["_Node"], action: Optional[int], untried_actions):
        self.parent = parent
        self.action = action
        self.children: List[_Node] = []
        self.untried_actions = untried_actions
        self.visits = 0
        self.value = None
        self.state = []
        self._min_value = None
        self._max_value = None

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
        untried_actions = [a for a in env.legal_actions if a not in self.state]
        child = _Node(parent=self, action=action, untried_actions=untried_actions)
        self.children.append(child)
        child.state = child.parent.state + [action] if child.parent else [action]
        return child
    
    def _expand(self, env: SimpleEnv, rng: random.Random) -> tuple["_Node", float]:
        action = self.untried_actions.pop(0)
        reward = env.step(action)
        child = self._create_new_child(action, env)
        return child, reward
    
    def _get_node_by_action(self, action: int) -> Optional["_Node"]: 
        return next((child for child in self.children if child.action == action), None)
    
    def go_to_state(self, state, root, env):
        node = root
        for action in state:
            if action not in [n.action for n in node.children]:
                node = node._create_new_child(action, env)
            else:
                node = node._get_node_by_action(action)
        return node
    
    def _backpropagate(self, reward: float) -> None:
        """
        Updates the node's statistics.
        """
        self.visits += 1
        # self._results[result] += 1
        self._min_value = min(self._min_value, reward) if self._min_value is not None else reward
        self._max_value = max(self._max_value, reward) if self._max_value is not None else reward
        self.edge_reward = reward
        if self.value is None:
            self.value = reward
        else:
            self.value = (self.value * (self.visits - 1) + reward) / self.visits
        if self.parent:
            self.parent._backpropagate(reward)
        return


def mmcts_search(root_env: SimpleEnv, iterations: int = 1000, uct_inf_softening: float = 2, base_temp: float = 5, decay: float = 0.05, p_max: float = 0.25, seed: Optional[int] = None) -> int:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    root = _Node(parent=None, action=None, untried_actions=root_env.legal_actions)
    best_path = []

    node = root
    env = root_env.clone()
    while not env.is_terminal():
        node, state_reward = node._expand(env, rng)
    current_state = node.state
    original_path = copy(node.state)
    max_reward = state_reward
    best_path = original_path
    node._backpropagate(state_reward)
    
    for _ in range(iterations):
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

            node = node.go_to_state(proposed_path, root, env)
            # Backpropagation
            node._backpropagate(reward)
            ratio = _exp_ratio(state_reward, reward, temperature=tempscale)

            accept_prob = min(1, ratio)
            if torch.rand(1) < accept_prob:
                state_reward = reward
                current_state = proposed_path
            node = node.go_to_state(current_state[:j+1], root, env)
            env.update(node.state)
        
            if reward > max_reward:
                max_reward = reward
                best_path = proposed_path


    # return the most visited child action
    if not root.children:
        # no expansion happened; pick a legal action
        return root_env.legal_actions[0]
    return env.get_path(best_path)