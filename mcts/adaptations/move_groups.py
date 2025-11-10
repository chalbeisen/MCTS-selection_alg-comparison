"""
UCT with Move Groups

Extends basic UCT by introducing an additional decision layer:
1. Select a move group (UCT applied at the group level)
2. Select a move within that group
"""

import math
import random
import numpy as np
from typing import List, Optional, Dict

from env.env import SimpleEnv
from mcts.node import _Node


class UCTMoveGroupNode(_Node):
    def __init__(
        self,
        parent: Optional["_Node"],
        action: Optional[int],
        untried_actions: List[int],
        group_size: int = 3,
    ):
        super().__init__(parent, action, untried_actions)
        self.group_size = group_size
        self.move_groups = self._create_groups(untried_actions, group_size)

    def _create_groups(self, actions: List[int], group_size: int) -> Dict[int, List[int]]:
        """Split actions into fixed-size random groups."""
        random.shuffle(actions)
        groups = {}
        group_id = 0
        for i in range(0, len(actions), group_size):
            groups[group_id] = actions[i:i + group_size]
            group_id += 1
        return groups

    def _select_group(self, explore_const: float = math.sqrt(2.0)) -> int:
        """UCT selection among move groups."""
        best_score, best_group = -float("inf"), None
        for gid, actions in self.move_groups.items():
            group_children = [c for c in self.children if c.action in actions]
            group_visits = sum(c.visits for c in group_children) or 1e-6
            group_value = sum(c.value for c in group_children)
            avg_value = group_value / group_visits if group_visits > 0 else 0.0

            score = avg_value + explore_const * math.sqrt(math.log(self.visits + 1) / group_visits)
            if score > best_score:
                best_score, best_group = score, gid
        return best_group

    def _select_move_in_group(self, group_id: int, explore_const: float = math.sqrt(2.0)) -> int:
        """UCT selection among moves within a chosen group."""
        group_actions = self.move_groups[group_id]
        best_score, best_action = -float("inf"), None
        for a in group_actions:
            child = self._get_node_by_action(a)
            if child is None or child.visits == 0:
                return a  # Try unvisited move first
            score = child.value + explore_const * math.sqrt(math.log(self.visits + 1) / child.visits)
            if score > best_score:
                best_score, best_action = score, a
        return best_action

    def _select_best_child(self, env: SimpleEnv) -> tuple["_Node", float]:
        """Two-level UCT: choose group, then move."""
        group_id = self._select_group()
        move = self._select_move_in_group(group_id)
        reward = env.step(move)
        child = self._get_node_by_action(move)
        if child is None:
            child = self._create_new_child(move, env)
        return child, reward

    def _create_new_child(self, action: int, env: SimpleEnv) -> "_Node":
        """Create and attach a new child node."""
        untried_actions = self.update_untried_actions(action, env)
        child = UCTMoveGroupNode(self, action, untried_actions, self.group_size)
        self.children.append(child)
        child.state = child.parent.state + [action] if child.parent else [action]
        return child

    def _backpropagate(self, reward: float) -> None:
        self.visits += 1
        self.edge_reward = reward
        if self.value is None:
            self.value = reward
        else:
            self.value = (self.value * (self.visits - 1) + reward) / self.visits
        if self.parent:
            self.parent._backpropagate(reward)
        return


def uct_search_movegroups(
    root_env: SimpleEnv,
    iterations: int = 1000,
    group_size: int = 3,
    seed: Optional[int] = None,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    root = UCTMoveGroupNode(parent=None, action=None, untried_actions=list(root_env.legal_actions), group_size=group_size)

    max_reward = -np.inf
    best_path = []
    best_iteration = 0

    for i in range(iterations):
        node = root
        env = root_env.clone()
        path = []

        while not env.is_terminal():
            if node.untried_actions == [] and node.children:
                node, reward = node._select_best_child(env)
            else:
                node, reward = node._expand(env)
            path.append(node.action)

        node._backpropagate(reward)

        if reward > max_reward:
            max_reward = reward
            best_path = path.copy()
            best_iteration = i

    return root, env.get_items_in_path(best_path), np.abs(max_reward), best_iteration
