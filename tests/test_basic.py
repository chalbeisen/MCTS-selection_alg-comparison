from mcts.env import SimpleEnv
from mcts.uct import uct_search


def test_uct_basic():
    env = SimpleEnv(max_depth=5, branching=3, seed=1)
    action = uct_search(env, iterations=50, seed=2)
    assert action in env.legal_actions()