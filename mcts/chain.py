"""
A simple deterministic D-chain environment (for testing tree search algorithms).

State: current position d in {1, ..., D}.
Actions:
    0 = a_L (Left): go to absorbing terminal state, reward = (D - d)/D
    1 = a_R (Right): move to next state (d + 1), unless d == D, where reward = Rf.

Episode terminates after either action (absorbing state).
"""

from typing import List
import copy


class ChainEnv:
    def __init__(self, D: int = 10, Rf: float = 1.0):
        """Create a D-chain environment.

        Args:
            D: chain length (number of states).
            Rf: final reward for reaching the end via a_R from state D.
        """
        self.D = D
        self.Rf = Rf
        self.legal_actions = [0, 1]
        self.done = False
        self.state = 1 
        self.visited = []  

    def reset(self):
        """Reset to the starting state."""
        self.state = 1          # start at state 1
        self.done = False
        self.visited = []       # list of taken action indices
        # actions have fixed indices: 0 = a_L, 1 = a_R
        self.legal_actions: List[int] = [0, 1]
        return self.state

    def clone(self) -> "ChainEnv":
        """Return a deep copy of the environment (for MCTS branching)."""
        return copy.deepcopy(self)

    def step(self, action_index: int) -> float:
        """Perform one environment step.

        Args:
            action_index: integer index of the action (0 = a_L, 1 = a_R)
        Returns:
            reward (float): immediate reward.
        """
        if self.done:
            raise ValueError("Episode already finished.")
        if action_index not in [0, 1]:
            raise ValueError("Invalid action index; must be 0 (a_L) or 1 (a_R).")

        self.visited.append(action_index)
        d = self.state

        # Apply deterministic transitions
        if action_index == 0:  # a_L
            reward = (self.D - d) / self.D
            self.done = True

        elif action_index == 1:  # a_R
            if d == self.D:
                reward = self.Rf
                self.done = True
            else:
                self.state += 1
                reward = 0.0
        return reward

    def is_terminal(self) -> bool:
        """Return True if current state is terminal."""
        return self.done

    def get_items_in_path(self, path: List[int]) -> List[str]:
        """Return a readable sequence of actions for a given path."""
        name_map = {0: "a_L", 1: "a_R"}
        return [name_map[a] for a in path]

    def update(self, new_path: List[int]) -> float:
        """Replace current trajectory with a new one and compute cumulative reward.

        Args:
            new_path: list of action indices (0 or 1).

        Returns:
            Total cumulative reward after following the given path.
        """
        self.reset()
        total_reward = 0.0

        for action in new_path:
            if self.done:
                break
            reward = self.step(action)
            total_reward += reward

        return total_reward
    
    def __repr__(self) -> str:
        if self.done:
            return f"ChainEnv(state=terminal)"
        return f"ChainEnv(state={self.state}, legal_actions={self.legal_actions})"
