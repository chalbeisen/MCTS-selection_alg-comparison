"""Lightweight MCTS package (scaffold)

Expose core modules: env, distributions, uct
"""

from .env import SimpleEnv
from .distributions import CategoricalDistribution
from .uct import uct_search
from .knapsack import KnapsackEnv

__all__ = ["SimpleEnv", "CategoricalDistribution", "uct_search", "KnapsackEnv"]
