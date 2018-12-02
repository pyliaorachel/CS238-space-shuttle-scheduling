import math

import numpy as np


def choose_action(Q, s, epsilon):
    n_s, n_a = Q.shape

    if np.random.rand() < epsilon: # explore
        return math.ceil(np.random.rand() * n_a) # cannot be 0
    else: # exploit
        return np.argmax(Q[s]) + 1 # offset 1

def update(Q, N_sas, rho, gamma):
    n_s, n_a, _ = N_sas.shape

    N_sa = np.sum(N_sas, axis=2)
    R = rho / N_sa
    N_sa = np.repeat(N_sa[:, :, np.newaxis], n_s, axis=2)
    T = N_sas / N_sa

    max_Q = np.max(Q, axis=1)
    Q = R + gamma * np.matmul(T, max_Q)

    return Q

def train(env, n_episodes, gamma, epsilon):
    # Init model
    N_sas = np.ones((env.n_s, env.n_a, env.n_s)) # uniform transition probability
    rho = np.zeros((env.n_s, env.n_a))
    Q = np.zeros((env.n_s, env.n_a))

    # Train
    total_rewards = 0
    for i_episode in range(n_episodes):
        s = env.reset()
        done = False
        rewards = 0
        while not done:
            # Choose actoin & observe new state and reward
            a = choose_action(Q, s, epsilon)
            sp, r, done, info = env.step(a)
            rewards += r

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
    return avg_rewards
