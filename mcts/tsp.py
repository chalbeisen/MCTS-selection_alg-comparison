"""A simple sequential Traveling Salesman environment.

State: current city and set of visited cities.
Each action chooses the next city to visit (integer index).
All cities must be visited exactly once.
Reward is returned only at terminal state (after returning to start).
"""

from typing import List, Tuple
import math
import copy


class TSPEnv:
    def __init__(self, cities: List[Tuple[float, float]]):
        """Create a TSP environment.

        cities: list of (x, y) coordinates for each city.
        """
        self.cities = list(cities)
        self.n_cities = len(cities)
        self.current_city = None
        self.visited = []
        self.legal_actions = [i for i in range(len(cities))]

    def clone(self) -> "TSPEnv":
        return copy.deepcopy(self)

    def distance(self, a: int, b: int) -> float:
        """Euclidean distance between cities a and b."""
        x1, y1 = self.cities[a]
        x2, y2 = self.cities[b]
        return math.hypot(x2 - x1, y2 - y1)

    def total_distance(self) -> float:
        """Compute total distance of the full tour (returning to start)."""
        if len(self.visited) < 2:
            return 0.0
        dist = sum(self.distance(self.visited[i], self.visited[i + 1])
                   for i in range(len(self.visited) - 1))
        # return to start
        dist += self.distance(self.visited[-1], self.visited[0])
        return dist

    def step(self, action_index: int) -> float:
        self.visited.append(action_index)
        self.current_city = action_index

        if self.is_terminal():
            # tour complete — reward = -total distance (since we minimize)
            return -self.total_distance()
        return 0.0

    def update(self, new_path: List[int]) -> float:
        """Replace current path with a new one (for search algorithms)."""
        self.visited = new_path
        self.current_city = new_path[-1]
        return -self.total_distance()

    def is_terminal(self) -> bool:
        """All cities have been visited."""
        return len(self.visited) == self.n_cities

    def __repr__(self) -> str:
        return (f"TSPEnv(current_city={self.current_city}, "
                f"visited={self.visited}, "
                f"remaining={self.legal_actions})")
