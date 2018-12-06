import math
import heapq

import numpy as np


MIN = -9999999
MAX = 9999999


def choose_action(Q, s, epsilon):
    n_s, n_a = Q.shape

    if np.random.rand() < epsilon: # explore
        return int(np.random.rand() * n_a)
    else: # exploit
        return np.argmax(Q[s])

def update(Q, s, a, N_sas, rho, gamma, max_iter=10):
    # Prioritized updates
    n_s, n_a = Q.shape

    pq = [(0, s, a)]
    pq_set = set()
    pq_set.add((s, a))
    i = 0
    while len(pq) > 0 and i < max_iter:
        _, ss, aa = heapq.heappop(pq)
        pq_set.remove((ss, aa))

        N_sa = np.sum(N_sas[ss, aa])
        R = rho[ss, aa] / N_sa
        T = N_sas[ss, aa] / N_sa

        max_Q = np.max(Q, axis=1)
        new_Q = R + gamma * np.dot(T, max_Q)
        diff = abs(new_Q - Q[ss, aa])
        Q[ss, aa] = new_Q

        for ns in range(ss):
            for na in range(min(ss - ns, n_a)):
                N_sa = np.sum(N_sas[ns, na])
                T = N_sas[ns, na] / N_sa
                if T[ss] > 0.001: # is pred of ss
                    p = -T[ss] * diff # negative since heapq pops smaller values first
                    if (ns, na) in pq_set:
                        j = [k for k, (_, sss, aaa) in enumerate(pq) if sss == ns and aaa == na][0]
                        pq[j] = (p, ns, na)
                    else:
                        heapq.heappush(pq, (p, ns, na))
                        pq_set.add((ns, na))

        i += 1

    return Q

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
        Q = np.zeros((env.n_s, env.n_a))
    else:
        N_sas = model['N_sas']
        rho = model['rho']
        Q = model['Q']

    # Train or eval
    total_rewards = 0
    for i_episode in range(n_episodes):
        epsilon = get_epsilon(i_episode)

        s = env.reset()
        done = False
        rewards = 0
        while not done:
            # Choose actoin & observe new state and reward
            a = choose_action(Q, s, epsilon)
            sp, r, done, info = env.step(a)
            rewards += r

            if not is_eval:
                # Update model
                N_sas[s, a, sp] += 1
                rho[s, a] += r

                # Update Q
                Q = update(Q, s, a, N_sas, rho, gamma)

            # Check end of episode
            if done:
                env.reset()
                break

            # Transition
            s = sp
        total_rewards += rewards
        print('Rewards: {}, total: {}'.format(rewards, total_rewards))

    avg_rewards = total_rewards / n_episodes
    return avg_rewards, { 'N_sas': N_sas, 'rho': rho, 'Q': Q }
