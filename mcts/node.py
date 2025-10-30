
from typing import List, Optional
from env import SimpleEnv
import math

class _Node:
    def __init__(self, parent: Optional["_Node"], action: Optional[int], untried_actions: List[int]):
        self.parent = parent
        self.action = action
        self.children: List[_Node] = []
        self.untried_actions = untried_actions
        self.visits = 0
        self.value = None
        self.state = []

    def uct_score(self, explore_const: float = math.sqrt(2.0)) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else 1
        return (self.value / self.visits) + explore_const * math.sqrt(math.log(parent_visits) / self.visits)

    def _create_new_child(self, action: int, env: SimpleEnv) -> "_Node":
        if not self.state: 
            untried_actions = list(env.legal_actions)
        else:
            untried_actions = [a for a in env.legal_actions if a not in self.state]
        self.untried_actions.remove(action)
        untried_actions.remove(action)
        child = _Node(parent=self, action=action, untried_actions=untried_actions)
        self.children.append(child)
        child.state = child.parent.state + [action] if child.parent else [action]
        return child
    
    def _expand(self, env: SimpleEnv) -> tuple["_Node", float]:
        action = self.untried_actions[0]
        reward = env.step(action)
        child = self._create_new_child(action, env)
        return child, reward
    
    def _get_node_by_action(self, action: int) -> Optional["_Node"]: 
        return next((child for child in self.children if child.action == action), None)
    
    def _backpropagate(self, reward: float):
        return None