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
    def __init__(self, parent: Optional["_Node"], action: Optional[int], untried_actions: List[int], state: List[int]):
        super().__init__(parent, action, untried_actions, state)

    def get_player(self):
        return self.player 
    
    def _best_child_uct(self) -> "_Node":
        child = max(self.children, key=lambda c: c.uct_score())
        return child
    
    def _best_child_N(self) -> "_Node":
        child = max(self.children, key=lambda c: c.visits)
        return child
    
    
    def _best_action_capped_distr(self, uct_inf_softening, p_max) -> "_Node":
        legal_mutation_actions = self.untried_actions
        node_uct_values = torch.tensor([child.uct_score() for child in self.children] + [float("inf")] * len(legal_mutation_actions))
        uct_values = compute_uct_scores(
            values=[c.value for c in self.children],
            visits=[c.visits for c in self.children],
            parent_visits=max(self.visits, 1)
        )

        # Append +inf for untried actions
        node_uct_values = torch.cat([uct_values, torch.full((len(legal_mutation_actions),), float("inf"), dtype=torch.float32)])

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
        if self.value is None:
            self.value = reward
        else:
            self.value = (self.value * (self.visits - 1) + reward) / self.visits
        if self.parent:
            self.parent._backpropagate(reward)
        return

class MMCTS_Node_TurnBased(MMCTS_Node, _NodeTurnBased):
    def __init__(self, parent, action, untried_actions, state, player=None):
        super().__init__(parent, action, untried_actions, state)
        self.player = player
    
    
    def _create_new_child(self, action: int, env: Env) -> "_Node":
        untried_actions = self.update_untried_actions(action, env)
        next_player = env.get_current_player()
        if next_player >=0 :
            player = 1 - env.get_current_player()
        else:
            player = 1 - self.player
        state = self.state + [action] if self.state else [action]
        child = MMCTS_Node_TurnBased(parent=self, action=action, untried_actions=untried_actions, state=state, player=player)
        self.children.append(child)

        return child

    
    def _backpropagate(self, reward: float, env: Env):
        self.visits += 1

        # Detect root node
        if self.action is not None:
            curr_reward = self._determine_reward(reward, env)
            self.value = (self.value * (self.visits - 1) + curr_reward) / self.visits

        if self.parent is not None:
            self.parent._backpropagate(reward, env)


def mmcts_search(root_env: Env, iterations: int = 1000, uct_inf_softening: float = 2, base_temp: float = 1000, decay: float = 0.05, p_max: float = 1.5, seed: Optional[int] = None) -> int:
    root = MMCTS_Node(parent=None, action=None, untried_actions=root_env.get_legal_actions())
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



def mutate(node, original_path: List[int],  mutate_index: int, mutate_value: float, env: Env):
    new_path = original_path[:mutate_index] + [mutate_value]
    reward = env.update(new_path)

    new_child = node._get_node_by_action(mutate_value)
    if new_child is None:
        new_child = node._create_new_child(mutate_value, env)

    return new_path, reward


def swap(original_path: List[int], new_index: int, new_value: float, env: Env):
    path = copy(original_path)
    if len(original_path) > 1: 
        old_index = next(i for i in range(len(original_path)) if original_path[i] == new_value)
        path[old_index], path[new_index] = path[new_index], path[old_index]
    else:
        path[new_index] = new_value
    reward = env.update(path)
    return path, reward

def compute_uct_scores(values: List[float], visits: List[int], parent_visits: int, explore_const: float = math.sqrt(2)) -> torch.Tensor:
    values_tensor = torch.tensor(values, dtype=torch.float32)
    visits_tensor = torch.tensor(visits, dtype=torch.float32)
    
    # Avoid division by zero
    safe_visits = torch.clamp(visits_tensor, min=1.0)
    uct_scores = (values_tensor / safe_visits) + explore_const * torch.sqrt(torch.log(torch.tensor(parent_visits, dtype=torch.float32)) / safe_visits)

    uct_scores[visits_tensor == 0] = float("inf")

    return uct_scores


def _exp_ratio(state_reward: float, proposal_reward: float, temperature: float = 1, objective: str = "min") -> float:
    if objective == "min":
        exponent = -(state_reward - proposal_reward) / temperature
    else:
        exponent = (state_reward - proposal_reward) / temperature
    exponent = max(min(exponent, 700), -700)  # clamp to safe range for exp
    return math.exp(exponent)


def go_to_state(state: List[int], root: "_Node", env: Env):
    node = root
    for i, action in enumerate(state):

        new_child = node._get_node_by_action(action)
        if new_child is None:
            new_child = node._create_new_child(action, env)
        new_child.state = state[:i+1]

        node = new_child
    return node


def complete_path_until_terminal(proposed_path: List[int], node: "_Node", env: Env, uct_inf_softening: float, p_max: float):          
    while not env.is_terminal():
        child_action = node._best_action_capped_distr(uct_inf_softening, p_max)
        proposed_path = proposed_path + [child_action]
        reward = env.step(child_action)

        new_child = node._get_node_by_action(child_action)
        if new_child is None:
            new_child = node._create_new_child(child_action, env)
        new_child.state = proposed_path

        node = new_child
    return proposed_path, reward, node

class MMCTS_Search():
    def __init__(self, root_env):
        self.max_reward = -np.inf
        self.best_path = []
        self.best_iteration = 0
        self.path_over_iter = []
        self.root_env = root_env
        self.uct_inf_softening = 2
        self.p_max = 1.5
        self.base_temp = 1000
        self.decay = 0.05

    
    def mmcts_search_turn_based(self, iterations: int) -> int:
        if self.root_env.get_state() == []:
            player = None
        else:
            player = self.root_env.get_current_player()

        root = MMCTS_Node_TurnBased(parent=None, action=None, untried_actions=self.root_env.get_legal_actions(), state = [], player = player)

        node = root
        # always start search from current state of global env
        initial_state = self.root_env.get_state()
        self.root_env.set_initial_state(initial_state)
        env = self.root_env.clone()
    
        while not env.is_terminal():
            node, state_reward = node._expand_random(env)
        #current_state = [s for s in node.state if s not in initial_state]
        current_state = copy(node.state)
        node._backpropagate(state_reward, env)
        
        for i in range(iterations):
            node = root
            env = self.root_env.clone()
            j = 0
            player = 0
            while not env.is_terminal():
                child_action = node._best_action_capped_distr(self.uct_inf_softening, self.p_max)

                # if the old path is the same as the new path continue with old path
                if current_state[j] == child_action:
                    node = node._get_node_by_action(child_action)
                    env.step(child_action)
                    player = 1 - player
                    j+=1
                    continue

                proposed_path, reward = mutate(node, current_state, j, child_action, env)
                tempscale = self.base_temp / (1.0 + self.decay * node.visits)
                
                node = go_to_state(proposed_path, root, env)
                if not env.is_terminal():
                    proposed_path, reward, node = complete_path_until_terminal(proposed_path, node, env, self.uct_inf_softening, self.p_max)

                # Backpropagation
                node._backpropagate(reward, env)
                ratio = _exp_ratio(state_reward[player], reward[player], temperature=tempscale)

                accept_prob = min(1, ratio)
                if torch.rand(1) < accept_prob:
                    state_reward = reward
                    current_state = proposed_path
                node = go_to_state(current_state[:j+1], root, env)
                env.update(node.state)

                player = 1 - player
                j+=1

        best_child_action = root._best_child_N().action
        self.root_env.step(best_child_action)

        return best_child_action