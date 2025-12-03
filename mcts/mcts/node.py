
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

    def uct_score(self, explore_const=math.sqrt(2)):
        if self.visits == 0:
            return float("inf")
        return (self.value / self.visits) + explore_const * math.sqrt(math.log(self.parent.visits) / self.visits)

    def update_untried_actions(self, action: int, env: Env) -> List[int]:
        """
        Update untried actions of current node, when new child node is created. 
        """
        untried_actions = [a for a in env.get_legal_actions() if a not in self.state and a!=action]
        self.untried_actions.remove(action)
        return untried_actions
    
    def expand(self, env: Env) -> tuple["_Node", float]:
        """
        Create new child node by sampling the first action from untried actions of current node. 
        """
        action = self.untried_actions[0]
        child = self._create_new_child(action, env)
        reward = env.step(action)
        return child, reward            
    
    def expand_random(self, env: Env) -> tuple["_Node", float]:
        """
        Create new child node by sampling random action from untried actions of current node. 
        """
        action = random.choice(self.untried_actions)
        child = self._create_new_child(action, env)
        reward = env.step(action)
        return child, reward  
    
    def expand_all_choose_random(self, env: Env) -> tuple["_Node", float]:
        """
        For each action in untried actions of current node create a new child node. 
        """
        untried_actions = self.untried_actions.copy()
        for action in untried_actions:
            self._create_new_child(action, env)
        action = random.choice(untried_actions)
        reward = env.step(action)
        child = self.get_child_by_action(action)
        return child, reward  
    
    def get_state(self) -> List[int]:
        raise NotImplementedError("this method should be implemented by subclass")

    def get_child_by_action(self, action: int) -> Optional["_Node"]: 
        return next((child for child in self.children if child.action == action), None)
    
    def backpropagate(self, reward: float):
        return None