"""Minimal distribution utilities used by algorithms in the scaffold.

Currently provides a tiny categorical distribution wrapper.
"""

import random
from typing import Sequence


class CategoricalDistribution:
    def __init__(self, weights: Sequence[float], seed: int | None = None):
        if len(weights) == 0:
            raise ValueError("weights must be non-empty")
        self.weights = list(weights)
        self._rand = random.Random(seed)

    def sample(self) -> int:
        total = sum(self.weights)
        r = self._rand.random() * total
        upto = 0.0
        for i, w in enumerate(self.weights):
            upto += w
            if r <= upto:
                return i
        return len(self.weights) - 1
