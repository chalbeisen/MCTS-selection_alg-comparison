import math
import numpy as np
import random
from typing import List, Optional, Dict

from env.env import Env   # or your TSPEnv
from mcts.node import _Node


class BeamNode(_Node):
    def __init__(self, parent: Optional["_Node"], 
                 action: Optional[int], 
                 untried_actions: List[int], 
                 state: List[int]):
        super().__init__(parent, action, untried_actions, state)
        self.depth = 0 if parent is None else parent.depth + 1

    def best_child(self) -> "_Node":
        child = max(self.children, key=lambda c: c.uct_score())
        return child

    def select_best_child(self, env: Env) -> tuple["_Node", float]:
        child = self.best_child()
        reward = env.step(child.action)
        return child, reward
    
    def create_new_child(self, action: int, env: Env) -> "_Node":
        untried_actions = self.update_untried_actions(action, env)
        state = self.state + [action] if self.state else [action]
        child = BeamNode(parent=self, action=action, untried_actions=untried_actions, state=state)
        self.children.append(child)
        child.state = child.parent.state + [action] if child.parent else [action]
        return child

class BEAM_Search:
    def __init__(self, root_env:Env):
        self.root_env = root_env
        
    def beam_mcts_search(self, 
                         iterations: int, 
                         beam_width: int = 20,
                         sim_limit: int = 100)-> tuple[list[int], float, int]:
        """
        MCTS search with beam pruning. Run MCTS search until the amount of rollouts reaches the depth limit and 
        prunes the tree at a given depth, keeping the top 

        Args:
            action (int): Action used to create the new child.
            env (Env): Environment used to update untried actions.
        """
        root = BeamNode(parent=None, action=None, untried_actions=list(self.root_env.get_legal_actions()), state = [])
        depth_counter: Dict[int, int] = {}
        best_path: list[int] = []
        best_reward = -np.inf
        best_iteration = 0
        path_over_iter = []

        for i in range(iterations):
            node = root
            env = self.root_env.clone()
            path: list[int] = []

            while not env.is_terminal():
                # count how many rollouts reached this depth
                depth_counter[node.depth] = depth_counter.get(node.depth, 0) + 1

                if node.untried_actions == [] and node.children:
                    # Selection by UCT
                    node, reward = node.select_best_child(env)
                else:
                    # Expansion
                    node, reward = node.expand_random(env)

                path.append(node.action)
            depth_counter[node.depth] = depth_counter.get(node.depth, 0) + 1

            # Beam pruning when limit reached
            if depth_counter[node.depth] >= sim_limit:
                self.prune_tree(root, node.depth, beam_width)
                depth_counter[node.depth] = 0


            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent

            if reward > best_reward:
                best_reward = reward
                best_path = path.copy()
                best_iteration = i
            path_over_iter.append(env.get_items_in_path(best_path))

        return root, env.get_items_in_path(best_path), abs(best_reward), best_iteration, path_over_iter


    def prune_tree(self, root: BeamNode, depth: int, beam_width: int):
        """
        Prune the search tree at a given depth:
        Keep only top 'beam_width' nodes by visit count + their ancestors.
        """
        # Gather all nodes at this depth
        level_nodes: list[BeamNode] = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.depth == depth:
                level_nodes.append(node)
            stack.extend(node.children)

        if not level_nodes:
            return

        # Select top nodes by visit count
        survivors = sorted(level_nodes, key=lambda n: n.visits, reverse=True)[:beam_width]
        survivor_set = set(survivors)

        # Include their ancestors so the tree stays connected
        for n in survivors:
            while n is not None:
                survivor_set.add(n)
                n = n.parent

        # Recursively prune children not in survivor set
        def prune_children(n: BeamNode):
            n.children = [c for c in n.children if c in survivor_set]
            for c in n.children:
                prune_children(c)

        prune_children(root)
