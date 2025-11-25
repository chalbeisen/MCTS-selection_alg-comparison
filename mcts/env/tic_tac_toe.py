import copy
import random
import mcts.mcts as mcts
from env.env import Env_TurnBased
from typing import List, Tuple
import pyspiel

#https://github.com/AlexMGitHub/Checkers-MCTS/tree/2e6bc7e9dfe1570fd2e1c17c230848daf01a4f37

class TicTacToeEnv(Env_TurnBased):
    def __init__(self):
        self.game = pyspiel.load_game(f"tic_tac_toe")

    def set_initial_state(self):
        self.state = self.game.new_initial_state()
        self.legal_actions = self.state.legal_actions()

    def is_turn_based(self):
        return True 
    
    def get_winner(self):
        # return index of winner or -1 for draw
        reward = self.state.returns()
        max_reward = max(reward)
        if max_reward > 0: 
            return reward.index(max_reward)
        else:
            return -1
    
    def get_state(self):
        return self.state.history()
    
    def get_current_player(self):
        return self.state.current_player()
    
    def get_legal_actions(self):
        return self.state.legal_actions()
    
    def get_action(self, i):
        return i
    
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