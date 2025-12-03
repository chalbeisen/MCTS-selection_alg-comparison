import math
import random
import torch
import numpy as np
from typing import List, Optional
from adaptations.sampling import uct_distribution
from adaptations.sampling.cap_distribution import cap_distribution, cap_distribution_wo_redistribution
from mcts.node import _Node, _NodeTurnBased

from env.env import Env
from copy import copy  

class MMCTS_Node(_Node):
    def __init__(self, parent: Optional["_Node"],
                 action: Optional[int],
                 untried_actions: List[int],
                 state: List[int]):
        super().__init__(parent, action, untried_actions, state)
    
    def _best_child_uct(self) -> "_Node":
        child = max(self.children, key=lambda c: c.uct_score())
        return child
    
    def _best_child_N(self) -> "_Node":
        child = max(self.children, key=lambda c: c.visits)
        return child

    def _best_action_capped_distr(self, uct_inf_softening: float, p_max: float) -> "_Node":
        """
        Sample new action based on capped UCT distribution. 
        """
        legal_mutation_actions = self.untried_actions
        node_uct_values = torch.tensor([child.uct_score() for child in self.children] + [float("inf")] * len(legal_mutation_actions), dtype=torch.float32)
        candidates = [child.action for child in self.children] + legal_mutation_actions
        distribution = uct_distribution(node_uct_values, uct_inf_softening)
        distribution = cap_distribution_wo_redistribution(distribution, (1+p_max)/len(candidates))
        candidate_index = torch.multinomial(distribution, 1).item()
        child_action = candidates[candidate_index]
        return child_action
    
    def _create_new_child(self, action: int, env: Env) -> "_Node":
        untried_actions = self.update_untried_actions(action, env)
        state = self.state + [action] if self.state else [action]
        child = MMCTS_Node(parent=self, action=action, untried_actions=untried_actions, state = state)
        self.children.append(child)
        return child
    
    def _backpropagate(self, reward: float) -> None:
        self.visits += 1
        self.edge_reward = reward
        if self.parent is None:
            self.value = reward
        else:
            self.value += reward
            self.parent._backpropagate(reward)
        return

class MMCTS_Search():
    def __init__(self, root_env: Env):
        self.root_env = root_env

    def _exp_ratio(self, state_reward: float, proposal_reward: float, temperature: float = 1, objective: str = "min") -> float:
        """
        Sample new action based on capped UCT distribution. Uses a softmax function and caps the UCT values so that 
        they form a distribution with densities below a certain threshold.
        """
        if objective == "min":
            exponent = -(state_reward - proposal_reward) / temperature
        else:
            exponent = (state_reward - proposal_reward) / temperature
        exponent = max(min(exponent, 700), -700)  # clamp to safe range for exp
        return math.exp(exponent)

    
    def go_to_state(self, state: List[int], root: "_Node", env: Env):
        """
        From the root node, find path along the tree that represents the given state. If an action in the sequence is not represented
        in the given tree, create new root with the corresponding action. 
        """
        node = root
        for i, action in enumerate(state):
            new_child = node._get_child_by_action(action)
            if new_child is None:
                new_child = node._create_new_child(action, env)
            new_child.state = state[:i+1]

            node = new_child
        return node
    
    def swap(self, original_path: List[int], new_index: int, new_value: float, env: Env):
        """
        Swap the action in original_path at new_index with action in original path which has the value
        of new_value.
        For example: abcdefg -> abfdecg
        """
        path = copy(original_path)
        if len(original_path) > 1: 
            old_index = next(i for i in range(len(original_path)) if original_path[i] == new_value)
            path[old_index], path[new_index] = path[new_index], path[old_index]
        else:
            path[new_index] = new_value
        reward = env.update(path)
        return path, reward
    
    def mutate(self, node, original_path: List[int],  mutate_index: int, mutate_value: float, env: Env):
        new_path = original_path[:mutate_index] + [mutate_value]
        
        new_child = node._get_node_by_action(mutate_value)
        if new_child is None:
            new_child = node._create_new_child(mutate_value, env)

        reward = env.update(new_path)
        return new_path, reward

    def mmcts_search(self, 
                     iterations: int,
                     uct_inf_softening: float, 
                     p_max: float, 
                     base_temp: float, 
                     decay: float) -> int:
        """
        Performs Metropolis Monte Carlo Search, adapting the standard MCTS with capped UCT sampling
        and Metropolis acceptance/rejection scheme. 

        Args:
            iterations (int): Number of search iterations.
            uct_inf_softening (float): Replacement value for infinite UCT values.
            p_max (float): Maximum allowed probability for any entry.
            base_temp (float): Initial temperature for acceptance probability.
            decay (float): Temperature decay per node visit for acceptance probability.
        """
        best_path = []
        best_iteration = 0
        path_over_iter = []
    
        root = MMCTS_Node(parent=None, action=None, untried_actions=list(self.root_env.get_legal_actions()), state = [])
        node = root
        env = self.root_env.clone()
        # Sample initial random path 
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
            env = self.root_env.clone()

            for j in range(len(original_path)):
                # Sample proposal action for index j along original path
                child_action = node._best_action_capped_distr(uct_inf_softening, p_max)

                # Swap action at index j along original path with action of value child_action
                proposed_path, reward = self.swap(current_state, j, child_action, env)
                tempscale = base_temp / (1.0 + decay * node.visits)
                
                # Find path in tree that represents proposed state
                node = self.go_to_state(proposed_path, root, env)

                # Backpropagation
                node._backpropagate(reward)

                # Calculate acceptance probability for proposal state (Metropolis inspired)
                ratio = self._exp_ratio(state_reward, reward, temperature=tempscale)

                # Accept or reject new proposal
                accept_prob = min(1, ratio)
                if torch.rand(1) < accept_prob:
                    state_reward = reward
                    current_state = proposed_path
                node = self.go_to_state(current_state[:j+1], root, env)
                env.update(node.state)
            
                if reward > max_reward:
                    max_reward = reward
                    best_path = proposed_path
                    best_iteration = i

            path_over_iter.append(env.get_items_in_path(best_path))

        return root, env.get_items_in_path(best_path), abs(max_reward), best_iteration, path_over_iter
    
    