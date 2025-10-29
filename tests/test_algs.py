from mcts.env import SimpleEnv
from mcts.ments import ments_search
from mcts.dents import dents_search


def test_ments_and_dents_return_valid_action():
    env = SimpleEnv(max_depth=4, branching=4, seed=10)
    a1 = ments_search(env, iterations=30, seed=2)
    a2 = dents_search(env, iterations=30, seed=2)
    assert a1 in env.legal_actions()
    assert a2 in env.legal_actions()
