import copy
import numpy as np
from typing import List

class SyntheticTreeEnv:
    def __init__(self, depth: int, seed: int = 0):
        self.k = depth  # branching factor
        self.d = depth  # tree depth equals branching factor
        self.rng = np.random.default_rng(seed)
        self.current_path: List[int] = []
        
        # Generate random edge values for the tree
        self.edge_values = {}
        self.leaf_means = {}
        self._generate_tree([], 0, 0.0)
        
        # Normalize leaf means to [0,1]
        means = list(self.leaf_means.values())
        min_val, max_val = min(means), max(means)
        for leaf in self.leaf_means:
            self.leaf_means[leaf] = (self.leaf_means[leaf] - min_val) / (max_val - min_val + 1e-8)

        self.legal_actions = list(range(depth))

    def _generate_tree(self, path: List[int], depth: int, acc: float):
        if depth == self.d:
            self.leaf_means[tuple(path)] = acc
            return
        for a in range(self.k):
            self.edge_values[tuple(path + [a])] = self.rng.random()
            self._generate_tree(path + [a], depth + 1, acc + self.edge_values[tuple(path + [a])])

    def clone(self) -> "SyntheticTreeEnv":
        return copy.deepcopy(self)

    def step(self, action: int) -> float:
        """Take an action and return reward (only at leaves)."""
        self.current_path.append(action)
        if self.is_terminal():
            return self._sample_reward()
        return 0.0

    def _sample_reward(self) -> float:
        mean = self.leaf_means[tuple(self.current_path)]
        return self.rng.normal(loc=mean, scale=1.0)

    def is_terminal(self) -> bool:
        return len(self.current_path) == self.d

    def update(self, new_path: List[int]) -> float:
        """Replace current path with a new one (for search algorithms)."""
        self.current_path = new_path
        if self.is_terminal():
            return self._sample_reward()
        return 0.0

    def get_items_in_path(self, path: List[int]) -> List[int]:
        return path
