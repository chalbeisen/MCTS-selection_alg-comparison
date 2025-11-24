from typing import List, Tuple
import math
import copy
import numpy as np
from env.env import Env


class TSPEnv(Env):
    def __init__(self, cities: List[Tuple[float, float]], seed = 0):
        self.cities = list(cities)
        self.n_cities = len(cities)
        self.current_city = None
        self.visited = []
        self.legal_actions = [i for i in range(len(cities))]
        self.seed = seed
        np.random.seed(seed)

    def is_turn_based(self):
        return False 
    
    def reset_state(self):
        self.visited = []
        
    def get_legal_actions(self):
        if self.visited == []: 
            untried_actions = list(self.legal_actions)
        else:
            untried_actions = [a for a in self.legal_actions if a not in self.visited]
        return untried_actions
    
    def get_action(self, i) -> Tuple[int,int]:
        return self.cities[i]
    
    def get_items_in_path(self, path: List[int]) -> List[Tuple[float,float]]:
        return [self.cities[i] for i in path] + [self.cities[path[0]]]
    
    def step(self, action_index: int) -> float:
        self.visited.append(action_index)
        self.current_city = action_index

        if self.is_terminal():
            # tour complete — reward = -total distance (since we minimize)
            return -self.total_distance()
        return 0.0
    
    def is_terminal(self) -> bool:
        return len(self.visited) == self.n_cities

    def distance(self, a: int, b: int) -> float:
        """Euclidean distance between cities a and b."""
        x1, y1 = self.cities[a]
        x2, y2 = self.cities[b]
        distance = math.hypot(x2 - x1, y2 - y1)
        noise = np.random.normal(0, distance*0.5)
        #return max(0.1, distance + noise)
        return distance

    def total_distance(self) -> float:
        """Compute total distance of the full tour (returning to start)."""
        if len(self.visited) < 2:
            return 0.0
        dist = sum(self.distance(self.visited[i], self.visited[i + 1])
                   for i in range(len(self.visited) - 1))
        # return to start
        dist += self.distance(self.visited[-1], self.visited[0])
        return dist

    def update(self, new_path: List[int]) -> float:
        self.visited = new_path
        self.current_city = new_path[-1]
        return -self.total_distance()

    def __repr__(self) -> str:
        return (f"TSPEnv(current_city={self.current_city}, "
                f"visited={self.visited}, "
                f"remaining={self.legal_actions})")
