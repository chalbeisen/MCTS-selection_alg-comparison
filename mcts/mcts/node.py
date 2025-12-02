
from typing import List, Optional
from env.env import Env
import math
import random
import torch

class _Node:
    def __init__(self, parent: Optional["_Node"], action: Optional[int], untried_actions: List[int], state: List[int]):
        self.parent = parent
        self.action = action
        self.children: List[_Node] = []
        self.untried_actions = untried_actions
        self.visits = 0
        self.value = 0
        self.edge_reward = 0
        self.state = state

    def uct_score(self, explore_const: float = math.sqrt(2)) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else 1
        return (self.value / self.visits) + explore_const * math.sqrt(math.log(parent_visits) / self.visits)

    def update_untried_actions(self, action: int, env: Env) -> List[int]:
        untried_actions = env.get_legal_actions()
        self.untried_actions.remove(action)
        untried_actions.remove(action)
        return untried_actions
    
    # working only for tsp
    """
    def update_untried_actions(self, action: int, env: Env) -> List[int]:
        untried_actions = [a for a in env.get_legal_actions() if a not in self.state and a!=action]
        self.untried_actions.remove(action)
        return untried_actions
    """
    
    
    def _expand(self, env: Env) -> tuple["_Node", float]:
        action = self.untried_actions[0]
        child = self._create_new_child(action, env)
        reward = env.step(action)
        return child, reward            
    
    def _expand_random(self, env: Env) -> tuple["_Node", float]:
        action = random.choice(self.untried_actions)
        child = self._create_new_child(action, env)
        reward = env.step(action)
        return child, reward  
    
    def _expand_all_choose_random(self, env: Env) -> tuple["_Node", float]:
        untried_actions = self.untried_actions.copy()
        for action in untried_actions:
            self._create_new_child(action, env)
        action = random.choice(untried_actions)
        reward = env.step(action)
        child = self._get_node_by_action(action)
        return child, reward  
    
    def _get_state(self) -> List[int]:
        raise NotImplementedError("this method should be implemented by subclass")

    def _get_node_by_action(self, action: int) -> Optional["_Node"]: 
        return next((child for child in self.children if child.action == action), None)
    
    def _backpropagate(self, reward: float):
        return None
    
class _NodeTurnBased(_Node):
    def __init__(self, parent: Optional["_NodeTurnBased"], action: Optional[int], untried_actions: List[int], state: List[int], player: int = None):
        super().__init__(parent, action, untried_actions, state)
        self.player = player

    def get_player():
        raise NotImplementedError("this method should be implemented by subclass")

    def _determine_reward(self, reward: float, env: Env):
        return reward[self.player]