# rl_grid_compare.py
# Deterministic gridworld (3x4) — Value Iteration, Policy Iteration, Q-Learning
# Matches the assignment spec:
# States = {(x,y) | x in {0,1,2}, y in {0,1,2,3}}
# Terminal = {(1,1):-10, (1,2):+10, (2,1):-20, (2,3):+20}
# All other states have reward 0. Reward for a transition is R(s') (next-state reward).
# Deterministic actions: up(^), down(v), left(<), right(>); hitting a wall keeps you in place.

from collections import defaultdict
import random
import math

# --- Grid definition ---------------------------------------------------------
XS, YS = 3, 4  # x ∈ {0,1,2}, y ∈ {0,1,2,3}
ALL_STATES = [(x, y) for y in range(YS) for x in range(XS)]

REWARD = defaultdict(int)
REWARD[(1, 1)] = -10
REWARD[(2, 1)] = -20
REWARD[(1, 2)] = +10
REWARD[(2, 3)] = +20
# others default to 0

TERMINALS = {(1, 1), (1, 2), (2, 1), (2, 3)}

ACTIONS = {
    '>': (1, 0),
    '<': (-1, 0),
    '^': (0, -1),
    'v': (0, 1),
}
ACTION_LIST = ['>', '<', '^', 'v']


def in_bounds(x, y):
    return 0 <= x < XS and 0 <= y < YS


def step(s, a):
    """Deterministic transition. Returns next_state s' and reward R(s')."""
    if s in TERMINALS:
        return s, REWARD[s]  # absorb
    dx, dy = ACTIONS[a]
    nx, ny = s[0] + dx, s[1] + dy
    if not in_bounds(nx, ny):
        nx, ny = s  # bump into wall => stay
    s1 = (nx, ny)
    return s1, REWARD[s1]


def render_values(V):
    """Pretty 3x4 table of values (y from 3->0 so it prints like the diagram)."""
    rows = []
    for y in reversed(range(YS)):
        row = []
        for x in range(XS):
            row.append(f"{V[(x,y)]:6.1f}")
        rows.append(" ".join(row))
    return "\n".join(rows)


def greedy_policy_from_V(V, gamma):
    """Greedy policy w.r.t. one-step lookahead using V."""
    policy = {}
    for s in ALL_STATES:
        if s in TERMINALS:
            policy[s] = '•'  # terminal marker
            continue
        # pick argmax_a [ R(s') + gamma * V(s') ]
        best_a, best_val = None, -math.inf
        for a in ACTION_LIST:
            s1, r = step(s, a)
            val = r + gamma * V[s1]
            if val > best_val:
                best_val, best_a = val, a
        policy[s] = best_a
    return policy


def render_policy(policy):
    rows = []
    for y in reversed(range(YS)):
        row = []
        for x in range(XS):
            row.append(policy[(x, y)])
        rows.append(" ".join(row))
    return "\n".join(rows)


def greedy_path(policy, start=(0, 0), max_steps=20):
    """Follow arrows until terminal or step limit; returns list of arrows."""
    s = start
    path = []
    visited = set()
    for _ in range(max_steps):
        if s in TERMINALS:
            break
        a = policy[s]
        if a == '•':
            break
        path.append(a)
        s1, _ = step(s, a)
        s = s1
        # simple loop guard
        if (s, a) in visited:
            break
        visited.add((s, a))
    return path

# --- Value Iteration ---------------------------------------------------------


def value_iteration(gamma, tol=1e-6, max_iter=1000):
    V = defaultdict(float)
    # initialize terminals to their immediate rewards (absorbing)
    for s in TERMINALS:
        V[s] = REWARD[s]

    it = 0
    while it < max_iter:
        delta = 0.0
        newV = V.copy()
        for s in ALL_STATES:
            if s in TERMINALS:
                newV[s] = REWARD[s]
                continue
            best = -math.inf
            for a in ACTION_LIST:
                s1, r = step(s, a)
                best = max(best, r + gamma * V[s1])
            delta = max(delta, abs(best - V[s]))
            newV[s] = best
        V = newV
        it += 1
        if delta < tol:
            break

    policy = greedy_policy_from_V(V, gamma)
    return V, policy, it

# --- Policy Iteration --------------------------------------------------------


def policy_iteration(gamma, eval_tol=1e-8, max_eval_iter=10_000):
    # random initial non-terminal actions
    policy = {s: random.choice(ACTION_LIST) for s in ALL_STATES}
    for t in TERMINALS:
        policy[t] = '•'

    stable = False
    outer_iter = 0
    while not stable:
        outer_iter += 1

        # Policy evaluation (iterative)
        V = defaultdict(float)
        for s in TERMINALS:
            V[s] = REWARD[s]

        for _ in range(max_eval_iter):
            delta = 0.0
            for s in ALL_STATES:
                if s in TERMINALS:
                    continue
                a = policy[s]
                s1, r = step(s, a)
                v_new = r + gamma * V[s1]
                delta = max(delta, abs(v_new - V[s]))
                V[s] = v_new
            if delta < eval_tol:
                break

        # Policy improvement
        stable = True
        for s in ALL_STATES:
            if s in TERMINALS:
                continue
            old_a = policy[s]
            best_a, best_val = old_a, -math.inf
            for a in ACTION_LIST:
                s1, r = step(s, a)
                val = r + gamma * V[s1]
                if val > best_val:
                    best_val, best_a = val, a
            policy[s] = best_a
            if best_a != old_a:
                stable = False

    return V, policy, outer_iter

# --- Q-learning --------------------------------------------------------------


def q_learning(gamma, episodes=4000, alpha=0.3, epsilon=0.1, seed=7):
    random.seed(seed)
    Q = {(s, a): 0.0 for s in ALL_STATES for a in ACTION_LIST}
    # initialize terminal Q(s,a) = reward(s) so it stabilizes faster
    for t in TERMINALS:
        for a in ACTION_LIST:
            Q[(t, a)] = REWARD[t]

    for _ in range(episodes):
        # start from a random non-terminal
        s = random.choice([s for s in ALL_STATES if s not in TERMINALS])
        for _ in range(40):  # safety bound: short episodes on this tiny grid
            # epsilon-greedy action
            if random.random() < epsilon:
                a = random.choice(ACTION_LIST)
            else:
                a = max(ACTION_LIST, key=lambda aa: Q[(s, aa)])
            s1, r = step(s, a)
            # target: r + gamma * max_a' Q(s1, a')
            best_next = max(Q[(s1, aa)] for aa in ACTION_LIST)
            Q[(s, a)] = (1 - alpha) * Q[(s, a)] + alpha * (r + gamma * best_next)
            s = s1
            if s in TERMINALS:
                break

    # derive V and greedy policy from Q
    V = defaultdict(float)
    policy = {}
    for s in ALL_STATES:
        if s in TERMINALS:
            V[s] = REWARD[s]
            policy[s] = '•'
        else:
            best_a = max(ACTION_LIST, key=lambda aa: Q[(s, aa)])
            policy[s] = best_a
            V[s] = max(Q[(s, aa)] for aa in ACTION_LIST)
    # “iterations” notion for Q-learning: we’ll report episodes
    return V, policy, episodes, Q

# --- Runner & Pretty Output --------------------------------------------------


def summarize_row(alg, gamma, iters, policy, V):
    path = greedy_path(policy, start=(0, 0))
    # Pull two illustrative state values (pick two non-terminal, non-start, stable cells)
    v_02 = V[(0, 2)]
    v_03 = V[(0, 3)]
    print(f"{alg:15s}  γ={gamma:<3}  iters={iters:<5}  path={path}  V(0,2)={v_02:5.1f}  V(0,3)={v_03:5.1f}")


def run_all():
    for gamma in (0.9, 0.5, 0.1):
        print("=" * 70)
        print(f"γ = {gamma}")
        print("- VALUE ITERATION -")
        V_vi, P_vi, it_vi = value_iteration(gamma)
        print("Value table:\n" + render_values(V_vi))
        print("Policy (arrows):\n" + render_policy(P_vi))
        summarize_row("Value Iter.", gamma, it_vi, P_vi, V_vi)
        print()

        print("- POLICY ITERATION -")
        V_pi, P_pi, it_pi = policy_iteration(gamma)
        print("Value table:\n" + render_values(V_pi))
        print("Policy (arrows):\n" + render_policy(P_pi))
        summarize_row("Policy Iter.", gamma, it_pi, P_pi, V_pi)
        print()

        print("- Q-LEARNING -")
        V_q, P_q, ep_q, Q = q_learning(gamma)
        print("Value table (from Q):\n" + render_values(V_q))
        print("Policy (arrows):\n" + render_policy(P_q))
        summarize_row("Q-learning", gamma, ep_q, P_q, V_q)
        print()

    print("=" * 70)
    print("Legend: • = terminal. Arrows show the greedy policy. "
          "‘path’ is the greedy path from (0,0) until a terminal or loop guard.")


if __name__ == "__main__":
    run_all()
