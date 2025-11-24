
from typing import List, Optional
from env.env import Env
import math

class _Node:
    def __init__(self, parent: Optional["_Node"], action: Optional[int], untried_actions: List[int], player = 0):
        self.parent = parent
        self.action = action
        self.children: List[_Node] = []
        self.untried_actions = untried_actions
        self.visits = 0
        self.value = 0
        self.state = []
        self.edge_reward = 0
        self.player = player

    def uct_score(self, explore_const: float = math.sqrt(2.0)) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else 1
        return (self.value / self.visits) + explore_const * math.sqrt(math.log(parent_visits) / self.visits)

    def update_untried_actions(self, action: int, env: Env) -> "_Node":
        untried_actions = env.get_legal_actions()
        self.untried_actions.remove(action)
        return untried_actions
    
    def _expand(self, env: Env) -> tuple["_Node", float]:
        action = self.untried_actions[0]
        reward = env.step(action)
        child = self._create_new_child(action, env)
        return child, reward
    
    def _determine_reward(self, reward: float, env: Env):
        if not env.is_turn_based():
            return reward
        parent_player = not self.player if self.parent_player is None else self.parent.player
        if reward > 0:
            return reward if parent_player == 0 else -reward
        else:
            return reward if parent_player == 1 else -reward
                
    
    def _get_node_by_action(self, action: int) -> Optional["_Node"]: 
        return next((child for child in self.children if child.action == action), None)
    
    def _backpropagate(self, reward: float):
        return None