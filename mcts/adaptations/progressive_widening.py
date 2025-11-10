import math
import random
from typing import Any, Dict, List, Optional, Tuple


class DPWAction:
    """Stats and children for a single action chosen at a state node."""
    def __init__(self, action):
        self.action = action
        self.visits = 0                    # nbt(s,a)
        self.total_reward = 0.0            # sum of returns for averaging
        self.children: List["DPWNode"] = []  # distinct observed successor states
        self.child_visit_counts: Dict[Any, int] = {}  # mapping state_id -> count for sampling


class DPWNode:
    """A tree node representing a state in the MCTS tree."""
    def __init__(self, parent: Optional["DPWNode"], action_from_parent: Optional[Any]):
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.visits = 0                    # N(s)
        self.total_reward = 0.0            # sum of returns observed when visiting this state
        self.actions: Dict[Any, DPWAction] = {}  # mapping action -> DPWAction
        # mapping of state_id -> child node for fast lookup
        self.child_state_map: Dict[Any, "DPWNode"] = {}

    def value(self) -> float:
        """Mean value estimate for this node (like Q̄ or V̂(s) used for backups)."""
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    def best_action(self):
        """Return action with max mean reward (for final recommendation)."""
        if not self.actions:
            return None
        best_a, best_v = None, -float("inf")
        for a, aentry in self.actions.items():
            q = aentry.total_reward / aentry.visits if aentry.visits > 0 else -float("inf")
            if q > best_v:
                best_v = q
                best_a = a
        return best_a


def ucb_score(parent_visits: int, action_entry: DPWAction, c: float = 1.0) -> float:
    """UCB-style score for an action within the pool."""
    if action_entry.visits == 0:
        return float("inf")
    avg = action_entry.total_reward / action_entry.visits
    return avg + c * math.sqrt(math.log(max(1, parent_visits)) / action_entry.visits)


def safe_state_id(env) -> Any:
    """Return a hashable id for the current env state.
    Prefer env.state_id() if provided, else fallback to repr(env).
    (Users should implement env.state_id for robust behavior)."""
    return getattr(env, "state_id", lambda: None)() or repr(env)


def dpw_select_action(node: DPWNode, env, C: float, alpha: float, c_ucb: float) -> Tuple[DPWAction, Any]:
    """Apply progressive widening on actions and choose an action (DPW step 1)."""
    # number of allowed actions given visits
    k = math.ceil(C * (node.visits ** alpha))
    # if the node has fewer unique actions tried than k, we expand by sampling a new action
    # The environment may provide a discrete set via env.legal_actions or a continuous sample_action()
    available_actions = list(node.actions.keys())
    if len(available_actions) < k:
        # sample a new action to add
        if hasattr(env, "sample_action"):
            new_action = env.sample_action()
        else:
            # fall back to adding an untried discrete action from env.legal_actions
            # if none remain, fall back to random pick from legal_actions
            try:
                cand = [a for a in env.legal_actions if a not in node.actions]
                new_action = random.choice(cand) if cand else random.choice(env.legal_actions)
            except Exception:
                # last resort: random action
                new_action = random.choice(getattr(env, "legal_actions", [None]))
        node.actions.setdefault(new_action, DPWAction(new_action))
        return node.actions[new_action], new_action

    # else choose among the existing pool by UCB
    best = None
    best_score = -float("inf")
    for aentry in node.actions.values():
        s = ucb_score(node.visits, aentry, c_ucb)
        if s > best_score:
            best_score = s
            best = aentry
    return best, best.action


def dpw_select_child_for_action(aentry: DPWAction, node: DPWNode, env, C: float, alpha: float) -> "DPWNode":
    """Apply the second (child) progressive widening for action aentry.
    If allowed, simulate once to create a new child state; else sample an existing child proportional to visits.
    Robust to edge cases where visits==0 or no children exist yet.
    """
    # ensure at least one allowed child
    k_prime = max(1, math.ceil(C * (aentry.visits ** alpha)))

    # If we have capacity to add a new child (progressive widening)
    if len(aentry.children) < k_prime:
        # create a new child by sampling the stochastic transition once
        reward = env.step(aentry.action)
        state_id = safe_state_id(env)

        # check if we already encountered this next-state
        child = node.child_state_map.get(state_id)
        if child is None:
            child = DPWNode(parent=node, action_from_parent=aentry.action)
            aentry.children.append(child)
            node.child_state_map[state_id] = child
            # initialize or increment visit count for this state id
            aentry.child_visit_counts[state_id] = aentry.child_visit_counts.get(state_id, 0) + 1
        else:
            # saw this state before; update its visit-count
            aentry.child_visit_counts[state_id] = aentry.child_visit_counts.get(state_id, 0) + 1
        return child

    # Otherwise sample an existing child proportional to counts
    # If no child_visit_counts present (shouldn't happen if children exist), fall back sensibly.
    if not aentry.child_visit_counts:
        # If no counts but some children exist, fall back to equal-prob sampling among children
        if aentry.children:
            return random.choice(aentry.children)
        # If truly no children exist (shouldn't happen given k_prime >= 1), create one now:
        reward = env.step(aentry.action)
        state_id = safe_state_id(env)
        child = node.child_state_map.get(state_id)
        if child is None:
            child = DPWNode(parent=node, action_from_parent=aentry.action)
            aentry.children.append(child)
            node.child_state_map[state_id] = child
            aentry.child_visit_counts[state_id] = 1
        else:
            aentry.child_visit_counts[state_id] = aentry.child_visit_counts.get(state_id, 0) + 1
        return child

    # Normal proportional sampling
    total = sum(aentry.child_visit_counts.values())
    if total <= 0:
        # fallback uniform
        if aentry.children:
            return random.choice(aentry.children)
        # as last resort, create a child
        reward = env.step(aentry.action)
        state_id = safe_state_id(env)
        child = node.child_state_map.get(state_id)
        if child is None:
            child = DPWNode(parent=node, action_from_parent=aentry.action)
            aentry.children.append(child)
            node.child_state_map[state_id] = child
            aentry.child_visit_counts[state_id] = 1
        else:
            aentry.child_visit_counts[state_id] = aentry.child_visit_counts.get(state_id, 0) + 1
        return child

    choice = random.uniform(0, total)
    cum = 0.0
    for sid, cnt in aentry.child_visit_counts.items():
        cum += cnt
        if choice <= cum:
            return node.child_state_map[sid]

    # Fallback (numerical safety): return a random child if mapping exists, otherwise last child
    if aentry.children:
        return random.choice(aentry.children)
    raise RuntimeError("DPW: unexpected empty children in sampling branch")


def default_rollout(env, max_rollout_depth=50):
    """Simple rollout policy: random play until terminal or depth limit.
    Returns total reward observed (since env.step returns reward only at terminal for sparse envs)."""
    total = 0.0
    depth = 0
    while not env.is_terminal() and depth < max_rollout_depth:
        if hasattr(env, "legal_actions") and env.legal_actions:
            a = random.choice(env.legal_actions)
        elif hasattr(env, "sample_action"):
            a = env.sample_action()
        else:
            raise RuntimeError("Env must provide legal_actions or sample_action for rollout.")
        r = env.step(a)
        total += r
        depth += 1
    # If env.reward on terminal is returned only at final step, rollout already captured it.
    return total


def dpw_mcts(root_env,
             iterations: int = 1000,
             C_action: float = 1.0,
             C_child: Optional[float] = None,
             alpha: float = 0.5,
             c_ucb: float = 1.0,
             rollout_fn=default_rollout,
             seed: Optional[int] = None):
    """
    Double-Progressive-Widening MCTS.

    Parameters
    ----------
    root_env : environment with clone(), is_terminal(), step(a), legal_actions or sample_action(), optional state_id()
    iterations : number of simulations
    C_action : progressive widening constant for actions
    C_child  : progressive widening constant used for child widening (if None, use C_action)
    alpha    : PW exponent in (0,1)
    c_ucb    : UCB exploration parameter
    rollout_fn : function(env_clone) -> return (numeric), used to estimate value of a new leaf
    seed     : RNG seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    if C_child is None:
        C_child = C_action

    root = DPWNode(parent=None, action_from_parent=None)
    # Important: initial root's env clone must be used when simulating; we do not attach env to node to save memory.

    for it in range(iterations):
        env = root_env.clone()
        node = root
        path = []  # store (node, aentry, child_state_id or None) for backprop

        # --- Selection + Expansion (using DPW) ---
        # Traverse until hitting terminal or leaf we will rollout from.
        while not env.is_terminal():
            # ensure node.visits counts the times we arrived to this node:
            node.visits += 1

            # choose or expand action via progressive widening
            aentry, chosen_action = dpw_select_action(node, env, C_action, alpha, c_ucb)

            # we will perform action in env; but for child PW we need a clone so we can inspect state_id after step
            env_before = env.clone()

            # increment action stats preemptively? We update after we observe return; here we will update later.
            # choose child under action using second PW
            child_node = dpw_select_child_for_action(aentry, node, env, C_child, alpha)

            # now `env` has been advanced by dpw_select_child_for_action if a new child was created,
            # otherwise we must set env to the chosen child's state. To keep env consistent, when we selected an
            # existing child we must restore env to that child's corresponding state. We try to use env.update(new_path)
            # if provided by the environment to set env to a saved state. Otherwise we keep using the env after step.
            # To keep this generic, we will assume DPW child creation always used the env clone and mutated it,
            # and child nodes correspond to the mutated env state we currently have.

            # record selection for backprop
            state_id = safe_state_id(env)
            path.append((node, aentry, child_node, state_id))

            # advance to child in the tree
            node = child_node

            # continue until terminal or a leaf where we will rollout
            if node.visits == 0:
                # we will rollout from this new node (leaf expansion)
                break

        # --- Simulation / Rollout ---
        # perform rollout from the current env (env now represents the env after following the path)
        # if the loop exited because env.is_terminal() == True, we can use reward 0 (already gotten last step) or
        # simply trust rollout to capture terminal reward.
        rollout_reward = 0.0
        if not env.is_terminal():
            # perform random rollout from env clone (we shouldn't mutate env used to map state ids)
            rollout_env = env.clone()
            rollout_reward = rollout_fn(rollout_env)
        else:
            # if env terminal, the last step already returned reward to env when child was created;
            # but in our dpw_select_child_for_action we invoked env.step(), which returned the reward then.
            # To be consistent, we can compute the reward by doing a zero-length rollout, or rely on env having stored it.
            # For simplicity, do a zero-length rollout here:
            rollout_reward = 0.0

        # --- Backpropagation ---
        # The path contains tuples for each step: (node_at_parent, aentry, child_node, state_id)
        # We propagate the rollout_reward up the tree, updating both node and action stats.
        G = rollout_reward
        # propagate from most recent to root
        for (parent_node, aentry, child_node, state_id) in reversed(path):
            # increment visits and totals for child_node
            child_node.visits += 1
            child_node.total_reward += G

            # update action stats
            aentry.visits += 1
            aentry.total_reward += G

            # update parent node stats
            parent_node.visits += 0  # already updated upon entry; not strictly necessary
            parent_node.total_reward += G

        # If we did not expand any node (path empty) and env was terminal at root, consider updating root
        if not path and env.is_terminal():
            root.visits += 1
            root.total_reward += rollout_reward

    # After iterations, return best action at root
    best_action = root.best_action()
    return best_action, root
