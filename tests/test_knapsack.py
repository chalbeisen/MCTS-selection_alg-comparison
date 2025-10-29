from mcts.knapsack import KnapsackEnv
from mcts.uct import uct_search
from mcts.ments import ments_search


def test_knapsack_env_basic():
    items = [(3, 4.0), (2, 3.0), (1, 2.0)]
    env = KnapsackEnv(items, capacity=3)
    # at index 0, we can take (weight 3) or skip
    assert set(env.legal_actions()) == {0, 1}
    # take item 0
    r = env.clone().step(1)
    # after taking item 0 we reached terminal (3 items -> step advanced 1 but not terminal yet) -> reward==0.0
    # but ensure total_value compute works after finishing
    env2 = KnapsackEnv(items, capacity=3)
    env2.step(0)  # skip item 0
    env2.step(1)  # take item 1
    env2.step(1)  # take item 2
    assert env2.is_terminal()
    assert env2.total_value() == 5.0


def test_algorithms_work_on_knapsack():
    items = [(3, 4.0), (2, 3.0), (1, 2.0)]
    env = KnapsackEnv(items, capacity=3)
    a = uct_search(env, iterations=50, seed=3)
    assert a in env.legal_actions()
    b = ments_search(env, iterations=50, seed=3)
    assert b in env.legal_actions()
