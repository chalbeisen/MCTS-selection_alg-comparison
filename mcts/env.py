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


class SimpleEnv:
    """A depth-limited tree environment with fixed branching.

    The environment state is just an integer depth. Actions are integers
    in range(branching). The game ends when depth >= max_depth. A
    terminal reward is sampled randomly (0.0 or 1.0) to allow stochastic
    rollouts.
    """
    def __init__(self, max_depth: int = 5, branching: int = 2, seed: int | None = None):
        self.max_depth = max_depth
        self.branching = branching
        self.depth = 0
        self._rand = random.Random(seed)

    def clone(self) -> "SimpleEnv":
        return copy.deepcopy(self)
    
    def get_items_in_path(self, path: List[int]) -> List[Tuple[float,float]]:
        raise NotImplementedError("this method should be implemented by sublcasse")

    def step(self, action: int) -> float:
        """Apply action, advance depth, and return reward (if terminal) or 0.0.

        Reward is only provided at terminal states and is sampled as 0.0 or 1.0
        uniformly to keep things simple.
        """
        if action < 0 or action >= self.branching:
            raise ValueError("invalid action")
        self.depth += 1
        if self.is_terminal():
            return float(self._rand.choice([0.0, 1.0]))
        return 0.0

    def is_terminal(self) -> bool:
        return self.depth >= self.max_depth
    
    def update(self, new_path: List[int]):
        raise NotImplementedError("this method should be implemented by sublcasse")