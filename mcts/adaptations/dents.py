import numpy as np
from typing import List, Optional
from env.env import SimpleEnv
from mcts.node import _Node

def entropy(probs: List[float]) -> float:
    probs = np.array(probs, dtype=float)
    probs = probs / (probs.sum() + 1e-8)
    return -np.sum(probs * np.log(probs + 1e-8))

def beta(N: int, alpha: float = 1.0) -> float:
    return alpha / np.log(np.e + max(1, N))

def f_tau(r, tau):
    r = np.array(r, dtype=float)
    if len(r) == 0:
        return np.array([])
    r_max = np.max(r)
    exp_r = np.exp((r - r_max) / (tau + 1e-8))
    return exp_r / (np.sum(exp_r) + 1e-8)

class Dents_Node(_Node):
    def __init__(self, parent: Optional["_Node"], action: Optional[int], untried_actions: List[int]):
        super().__init__(parent, action, untried_actions)
        self.Q_hat = {}
        self.HQ = {}
        self.HV = 0.0
        self.Q_sum = {}      # cumulative reward sum per action (for stochastic updates)
        self.N_sa = {}

    def _boltzmann_policy(self, temp: float, epsilon: float) -> tuple[List[float], List[int]]:
        
        actions = [child.action for child in self.children] + self.untried_actions
        if len(actions) == 0:
            return np.array([]), []
        
        beta_val = beta(self.visits)
        Qs = np.array([self.Q_hat.get(a, 0.0) for a in actions])
        Hs = np.array([self.HQ.get(a, 0.0) for a in actions])
        # equation 21
        Q_eff = Qs + beta_val * Hs
        # equation 21
        probs = f_tau(Q_eff, temp)

        # Add uniform mixture for exploration
        visit_count = sum([child.visits for child in self.children]) 
        lamb = epsilon * len(Qs) / np.log(visit_count + 2) 
        lamb = min(max(lamb, 0.0), 1.0) # clamp to [0,1]    
        uniform = np.ones_like(probs) / len(probs)
        # equation 20
        mixed_policy = (1 - lamb) * probs + lamb * uniform
        return mixed_policy, actions

    def _create_new_child(self, action: int, env: SimpleEnv) -> "_Node":
        untried_actions = self.update_untried_actions(action, env)
        child = Dents_Node(parent=self, action=action, untried_actions=untried_actions)
        self.children.append(child)
        child.state = self.state + [action] if self.state else [action]
        return child

    def _select_action(self, temp: float, epsilon: float) -> int:
        probs, actions = self._boltzmann_policy(temp, epsilon)
        return int(np.random.choice(actions, p=probs))

    def _select_child(self, env: SimpleEnv, temp: float, epsilon: float) -> tuple["_Node", float]:
        action = self._select_action(temp, epsilon)
        reward = env.step(action)
        selected_child = self._get_node_by_action(action)
        if selected_child is None:
            selected_child = self._create_new_child(action, env)
            selected_child.edge_reward = reward
        return selected_child, reward

    def _backpropagate(self, temp: float, reward: float, node: "_Node", epsilon: float):
        """Propagate reward and entropy information up the tree."""
        node.V_hat = reward
        cur = node
        while cur is not None:
            # increment visits
            cur.visits += 1

            # Bellman updates for each child (equation 17) - deterministic version
            for child in cur.children:
                cur.Q_hat[child.action] = child.edge_reward + child.V_hat
                cur.HQ[child.action] = child.HV

            # Value update (equation 18)
            if cur.Q_hat:
                cur.V_hat = max(cur.Q_hat.values())
            else:
                cur.V_hat = cur.V_hat

            # update entropy of this node’s Boltzmann policy (equation 22)
            probs, actions = cur._boltzmann_policy(temp, epsilon)
            cur.HV = entropy(probs) + np.sum(
                [p * cur.HQ.get(a, 0.0) for p, a in zip(probs, actions)]
            )
            cur.edge_reward = reward
            # move up
            cur = cur.parent
        
    def _backpropagate_stochastic(self, temp: float, reward: float, node: "_Node", epsilon: float):
        """Propagate reward and entropy information up the tree (stochastic version)."""
        node.V_hat = reward
        cur = node
        while cur is not None:
            # increment visits
            cur.visits += 1

            # Bellman updates for each child (equation 23)
            for child in cur.children:
                # update cumulative sum of Q for stochastic rewards
                cur.Q_sum[child.action] = cur.Q_sum.get(child.action, 0.0) + child.edge_reward + child.V_hat
                cur.N_sa[child.action] = cur.N_sa.get(child.action, 0) + 1

                # compute mean Q value for that (s,a)
                N_sa = cur.N_sa[child.action]
                cur.Q_hat[child.action] = cur.Q_sum[child.action] / N_sa

                # entropy contribution
                cur.HQ[child.action] = child.HV

            # update node value as max over child Qs (equation 18)
            if cur.Q_hat:
                cur.V_hat = max(cur.Q_hat.values())
            else:
                cur.V_hat = cur.V_hat

            # update entropy of Boltzmann policy (equation 22)
            probs, actions = cur._boltzmann_policy(temp, epsilon)
            cur.HV = entropy(probs) + np.sum(
                [p * cur.HQ.get(a, 0.0) for p, a in zip(probs, actions)]
            )

            # move up the tree
            cur = cur.parent


def dents_search(root_env: SimpleEnv,
                 iterations: int = 1000,
                 base_temp: float = 10.0,
                 decay: float = 0.001,
                 epsilon: float = 1.0,
                 seed: Optional[int] = None):
    if seed is not None:
        np.random.seed(seed)

    root = Dents_Node(parent=None, action=None, untried_actions=list(root_env.legal_actions))
    max_reward = -np.inf
    best_path = []
    best_iter = 0

    for i in range(iterations):
        node = root
        env = root_env.clone()
        path = []

        while not env.is_terminal():
            tempscale = base_temp / (1.0 + decay * node.visits)
            node, reward = node._select_child(env, tempscale, epsilon)
            path.append(node.action)

        node._backpropagate(tempscale, reward, node, epsilon)

        if reward > max_reward:
            max_reward = reward
            best_path = path.copy()
            best_iter = i

    return root, env.get_items_in_path(best_path), abs(max_reward), best_iter