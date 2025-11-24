"""A tiny environment interface used by the scaffolded UCT algorithm.

This is a deliberately simple environment to enable tests and examples.
It implements the minimum methods used by the UCT implementation:

- clone()
- legal_actions()
- step(action) -> reward
- is_terminal()

"""
import copy
import random
from typing import List, Tuple


class Env:

    def clone(self) -> "Env":
        return copy.deepcopy(self)
    
    def is_turn_based(self) -> bool:
        raise NotImplementedError("this method should be implemented by subclass")

    def get_legal_actions(self, state):
        raise NotImplementedError("this method should be implemented by subclass")

    def get_action(self, i: int):
        raise NotImplementedError("this method should be implemented by subclass")
    
    def get_items_in_path(self, path: List[int]) -> List[Tuple[float,float]]:
        raise NotImplementedError("this method should be implemented by subclass")

    def step(self, action: int) -> float:
        raise NotImplementedError("this method should be implemented by subclass")
    
    def update(self, new_path: List[int]):
        raise NotImplementedError("this method should be implemented by subclass")

    def is_terminal(self) -> bool:
        raise NotImplementedError("this method should be implemented by subclass")