import math
import heapq

import numpy as np


MIN = -9999999
MAX = 9999999

def policy(U, N_sas, rho, s, actions, gamma):
    best_a, best_v = None, MIN
    for a in actions:
        N_sa = np.sum(N_sas[s, a])
        R = rho[s, a] / N_sa
        T = N_sas[s, a] / N_sa

        v = R + gamma * np.dot(T, U)
        if v > best_v:
            best_a, best_v = a, v
    return best_a, best_v

def choose_action(U, N_sas, rho, s, actions, gamma, epsilon):
    n_s = U.shape

    if np.random.rand() < epsilon: # explore
        return np.random.choice(actions)
    else: # exploit
        best_a, best_v = policy(U, N_sas, rho, s, actions, gamma)
        return best_a

def update(U, N_sas, rho, s, actions, gamma, max_iter=10):
    # Prioritized updates
    n_s = U.shape[0]
    n_a = len(actions)

    pq = [(0, s)]
    pq_set = set()
    pq_set.add(s)
    i = 0
    while len(pq) > 0 and i < max_iter:
        _, ss = heapq.heappop(pq)
        pq_set.remove(ss)

        _, best_v = policy(U, N_sas, rho, ss, actions, gamma)
        diff = abs(best_v - U[ss])
        U[ss] = best_v

        for ns in range(ss):
            for na in range(min(ss - ns, n_a)):
                N_sa = np.sum(N_sas[ns, na])
                T = N_sas[ns, na] / N_sa
                if T[ss] > 0.001: # is pred of ss
                    p = -T[ss] * diff # negative since heapq pops smaller values first
                    if ns in pq_set:
                        j = [k for k, (_, sss) in enumerate(pq) if sss == ns][0]
                        pq[j] = (p, ns)
                    else:
                        heapq.heappush(pq, (p, ns))
                        pq_set.add(ns)

        i += 1

    return U

def run(env, n_episodes, gamma, is_eval=False, model=None):
    actions = list(range(env.n_a))
    if not is_eval:
        get_epsilon = lambda i: max(0.01, min(1, 1.0 - math.log10((i+1)/(n_episodes / 10)))) # discounted over time
    else:
        get_epsilon = lambda i: 0

    # Init model
    if model is None:
        N_sas = np.ones((env.n_s, env.n_a, env.n_s)) # uniform transition probability
        rho = np.zeros((env.n_s, env.n_a))
        U = np.zeros(env.n_s)
    else:
        N_sas = model['N_sas']
        rho = model['rho']
        U = model['U']

    # Train or eval
    total_rewards = 0
    for i_episode in range(n_episodes):
        epsilon = get_epsilon(i_episode)

        s = env.reset()
        done = False
        rewards = 0
        while not done:
            # Choose actoin & observe new state and reward
            a = choose_action(U, N_sas, rho, s, actions, gamma, epsilon)
            sp, r, done, info = env.step(a)
            rewards += r

            if not is_eval:
                # Update model
                N_sas[s, a, sp] += 1
                rho[s, a] += r

                # Update U
                U = update(U, N_sas, rho, s, actions, gamma)

            # Check end of episode
            if done:
                env.reset()
                break

            # Transition
            s = sp
        total_rewards += rewards
        print('Rewards: {}, total: {}'.format(rewards, total_rewards))

    avg_rewards = total_rewards / n_episodes
    return avg_rewards, { 'N_sas': N_sas, 'rho': rho, 'U': U }
