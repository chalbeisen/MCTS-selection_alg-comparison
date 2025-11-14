"""A simple sequential knapsack environment.

State is the current index into `items` and the remaining capacity. Each
decision chooses whether to take (1) or skip (0) the current item. Actions
are provided as integers. Rewards are returned only at terminal states
(after the last item), consistent with the other scaffolded envs.
"""

from typing import List, Tuple
import copy


class KnapsackEnv:
    def __init__(self, items: List[Tuple[int, float]], capacity: int):
        """Create a knapsack env.

        items: list of (weight:int, value:float)
        capacity: int maximum total weight
        """
        self.items = list(items)
        self.capacity = int(capacity)
        self.remaining_capacity = int(capacity)
        self.taken: List[int] = []  # indices of items taken
        self.legal_actions: List[int] = []  # indices of legal items 
        self.index = 0

        self.update_legal_actions()

    def clone(self) -> "KnapsackEnv":
        return copy.deepcopy(self)
    
    def get_action(self, i) -> int:
        return self.items[i]
    
    def get_path(self, nodes) -> List[Tuple[int, float]]:
        return [self.items[n.action] for n in nodes]
    
    def legal_mutation_actions(self, original_path, mutate_index):
        used = sum(self.items[i][0] for i in original_path)
        available = used - self.items[original_path[mutate_index]][0]
        legal_actions = []
        for action, item in enumerate(self.items):
            if item[0] + available < self.capacity:
                legal_actions.append(action)
        return legal_actions

    def update_remaining_capacity(self) -> int:
        used = sum(self.items[i][0] for i in self.taken)
        self.remaining_capacity = self.capacity - used

    def update_legal_actions(self) -> List[int]:
        self.legal_actions = []
        for i, (w, _) in enumerate(self.items):
            if w <= self.remaining_capacity and i not in self.taken:
                self.legal_actions.append(i)

    def step(self, action_index: int) -> float:
        """Apply action (0=skip, 1=take). Return reward only at terminal.
        Raises ValueError on invalid actions.
        """
        w, _ = self.items[action_index]
        if w > self.remaining_capacity:
            raise ValueError("cannot take item; weight exceeds remaining capacity")
        self.taken.append(action_index)
        self.update_remaining_capacity()
        self.update_legal_actions()
        self.index += 1
        if self.is_terminal():
            self.index = 0
            return float(self.total_value())
        return 0.0
    
    def mutate(self, new_path) -> float:
        self.taken = new_path
        self.update_remaining_capacity()
        self.update_legal_actions()
        return float(self.total_value())
        

    def is_terminal(self) -> bool:
        return self.index >= len(self.items) or len(self.legal_actions)==0
            
    def total_value(self) -> float:
        return sum(self.items[i][1] for i in self.taken)

    def __repr__(self) -> str:
        return f"KnapsackEnv(index={self.index}, taken={self.taken}, remaining={self.remaining_capacity()})"
