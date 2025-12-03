"""A simple UCT implementation used as the first algorithm port.

This is a minimal, well-commented implementation intended as a starting
point to port other algorithms from the C++ codebase.
"""

import numpy as np
from typing import List, Optional

from env.env import Env
from mcts.node import _Node, _NodeTurnBased


class MCTS_Node(_Node):
    def __init__(self, parent: Optional["_Node"], action: Optional[int], untried_actions: List[int], state: List[int]):
        super().__init__(parent, action, untried_actions, state)

    def best_child_uct(self) -> "_Node":
        child = max(self.children, key=lambda c: c.uct_score())
        return child
    
    def best_child_N(self) -> "_Node":
        child = max(self.children, key=lambda c: c.visits)
        return child

    def select_best_child(self, env: Env) -> tuple["_Node", float]:
        child = self.best_child_uct()
        reward = env.step(child.action)
        return child, reward

    def create_new_child(self, action: int, env: Env) -> "_Node":
        untried_actions = self.update_untried_actions(action, env)
        state = self.state + [action] if self.state else [action]
        child = MCTS_Node(parent=self, action=action, untried_actions=untried_actions, state = state)
        self.children.append(child)
        return child
    
    def backpropagate(self, reward: float, env: Env) -> None:
        self.visits += 1
        self.edge_reward = reward
        if self.value is None:
            self.value = reward
        else:
            self.value += reward
        if self.parent:
            self.parent.backpropagate(reward, env)
        return

class MCTS_Search():
    def __init__(self, root_env:Env):
        self.max_reward = -np.inf
        self.root_env = root_env
    
    def mcts_search(self, iterations: int) -> int:
        """
        Performs standard Monte Carlo Tree Search. 

        Args:
            iterations (int): Number of search iterations.
        """
        root = MCTS_Node(parent=None, action=None, untried_actions=list(self.root_env.get_legal_actions()), state = [])

        max_reward = -np.inf
        best_path = []
        best_iteration = 0
        path_over_iter = []

        for i in range(iterations):
            node = root
            env = self.root_env.clone()
            path = []
            while not env.is_terminal():
                if node.untried_actions == [] and node.children:
                    # Selection
                    node, reward = node.select_best_child(env)
                else:
                    # Expansion
                    node, reward = node.expand_random(env)
                path.append(node.action)

            # Backpropagation
            node.backpropagate(reward, env)
            
            if reward > max_reward:
                max_reward = reward
                best_path = path.copy()
                best_iteration = i
            path_over_iter.append(env.get_items_in_path(best_path))

        return root, env.get_items_in_path(best_path), np.abs(max_reward), best_iteration, path_over_iter