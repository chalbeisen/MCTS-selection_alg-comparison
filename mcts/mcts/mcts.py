"""A simple UCT implementation used as the first algorithm port.

This is a minimal, well-commented implementation intended as a starting
point to port other algorithms from the C++ codebase.
"""

import numpy as np
from typing import List, Optional

from env.env import Env
from mcts.node import _Node, _NodeTurnBased


class UCT_Node(_Node):
    def __init__(self, parent: Optional["_Node"], action: Optional[int], untried_actions: List[int]):
        super().__init__(parent, action, untried_actions)

    def _best_child_uct(self) -> "_Node":
        child = max(self.children, key=lambda c: c.uct_score())
        return child
    
    def _best_child_N(self) -> "_Node":
        child = max(self.children, key=lambda c: c.visits)
        return child

    def _select_best_child(self, env: Env) -> tuple["_Node", float]:
        child = self._best_child_uct()
        reward = env.step(child.action)
        return child, reward

    def _create_new_child(self, action: int, env: Env) -> "_Node":
        untried_actions = self.update_untried_actions(action, env)
        child = UCT_Node(parent=self, action=action, untried_actions=untried_actions)
        self.children.append(child)
        child.state = env.get_state()
        return child
    
    def _backpropagate(self, reward: float, env: Env) -> None:
        self.visits += 1
        self.edge_reward = reward
        if self.value is None:
            self.value = reward
        else:
            self.value = (self.value * (self.visits - 1) + reward) / self.visits
        if self.parent:
            self.parent._backpropagate(reward, env)
        return

class UCT_Node_TurnBased(UCT_Node, _NodeTurnBased):
    def __init__(self, parent, action, untried_actions, player=None):
        super().__init__(parent, action, untried_actions)
        self.player = player

    def get_player(self):
        return self.player 
    
    def _create_new_child(self, action: int, env: Env) -> "_Node":
        untried_actions = self.update_untried_actions(action, env)
        next_player = env.get_current_player()
        if next_player >=0 :
            player = 1 - env.get_current_player()
        else:
            player = 1 - self.player
        child = UCT_Node_TurnBased(parent=self, action=action, untried_actions=untried_actions, player=player)
        self.children.append(child)
        child.state = env.get_state()

        return child

    def _backpropagate(self, reward: float, env: Env):
        self.visits += 1

        # Detect root node
        if self.action is not None:
            curr_reward = self._determine_reward(reward, env)
            self.value = (self.value * (self.visits - 1) + curr_reward) / self.visits

        if self.parent is not None:
            self.parent._backpropagate(reward, env)


def uct_search(root_env: Env, iterations: int = 1000, seed: Optional[int] = None) -> int:
    root = UCT_Node(parent=None, action=None, untried_actions=root_env.get_legal_actions())

    max_reward = -np.inf
    best_path = []
    best_iteration = 0
    path_over_iter = []

    for i in range(iterations):
        node = root
        env = root_env.clone()
        path = []
        while not env.is_terminal():
            if node.untried_actions == [] and node.children:
                # Selection
                node, reward = node._select_best_child(env)
            else:
                # Expansion
                node, reward = node._expand(env)
            path.append(node.action)

        # Backpropagation
        node._backpropagate(reward, env)
        
        if reward > max_reward:
            max_reward = reward
            best_path = path.copy()
            best_iteration = i
        path_over_iter.append(env.get_items_in_path(best_path))

    return root, env.get_items_in_path(best_path), np.abs(max_reward), best_iteration, path_over_iter

class MCTS_Search():
    def __init__(self, root_env):
        self.max_reward = -np.inf
        self.best_path = []
        self.best_iteration = 0
        self.path_over_iter = []
        self.root_env = root_env

    def uct_search_turn_based(self, iterations: int) -> int:
        if self.root_env.get_state() == []:
            player = None
        else:
            player = self.root_env.get_current_player()

        root = UCT_Node_TurnBased(parent=None, action=None, untried_actions=self.root_env.get_legal_actions(), player = player)

        for i in range(iterations):
            node = root
            env = self.root_env.clone()
            path = []
            while not env.is_terminal():
                if node.untried_actions == [] and node.children:
                    # Selection
                    node, reward = node._select_best_child(env)
                else:
                    # Expansion
                    node, reward = node._expand_ramdom(env)
                path.append(node.action)

            # Backpropagation
            node._backpropagate(reward, env)
        best_child_action = root._best_child_N().action
        self.root_env.step(best_child_action)

        return best_child_action