import copy
import random
import mcts.mcts as mcts
from env.env import Env_TurnBased
from typing import List, Tuple
import pyspiel

# http://moderndescartes.com/essays/implementing_go/
# https://www.britgo.org/intro/intro2.html

class GoEnv(Env_TurnBased):
    def __init__(self, N: int = 6):
        game = pyspiel.load_game(f"go(board_size={N})")
        self.state = game.new_initial_state()

    def clone(self) -> "GoEnv":
        return copy.deepcopy(self)
    
    def get_action(self, i):
        return i
    
    def get_state(self):
        return self.state
    
    def get_items_in_path(self, path: List[int]) -> List[Tuple[float,float]]:
        return path

    def step(self, action: int) -> float:
        self.state.apply_action(action)
        if self.is_terminal():
            return self.state.returns()
        return 0.0

    def is_terminal(self) -> bool:
        return self.state.is_terminal()
    
    def update(self, new_path: List[int]):
        raise NotImplementedError("this method should be implemented by sublcasse")
    
