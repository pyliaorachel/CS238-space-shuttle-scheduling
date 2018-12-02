import math

import numpy as np


def choose_action(Q, s, epsilon):
    n_s, n_a = Q.shape

    if np.random.rand() < epsilon: # explore
        return int(np.random.rand() * n_a)
    else: # exploit
        return np.argmax(Q[s])

def update(Q, N_sas, rho, gamma):
    n_s, n_a, _ = N_sas.shape

    N_sa = np.sum(N_sas, axis=2)
    R = rho / N_sa
    N_sa = np.repeat(N_sa[:, :, np.newaxis], n_s, axis=2)
    T = N_sas / N_sa

    max_Q = np.max(Q, axis=1)
    Q = R + gamma * np.matmul(T, max_Q)

    return Q

def run(env, n_episodes, gamma, is_eval=False, model=None):
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
                Q = update(Q, N_sas, rho, gamma)

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
