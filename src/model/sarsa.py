import math

import numpy as np


def choose_action(Q, s, epsilon):
    n_s, n_a = Q.shape

    if np.random.rand() < epsilon: # explore
        return int(np.random.rand() * n_a)
    else: # exploit
        return np.argmax(Q[s])

def update(Q, s, a, r, sp, lr, gamma):
    Q[s, a] += lr * (r + gamma * np.random.choice(Q[sp],1) - Q[s, a])
    return Q

def run(env, n_episodes, gamma, is_eval=False, model=None):
    if not is_eval:
        get_epsilon = lambda i: max(0.01, min(1, 1.0 - math.log10((i+1)/(n_episodes / 10)))) # discounted over time
        get_lr = lambda i: max(0.01, min(0.5, 1.0 - math.log10((i+1)/(n_episodes / 10))))
    else:
        get_epsilon = lambda i: 0
        get_lr = lambda i: 0

    # Init model
    if not is_eval:
        Q = np.zeros((env.n_s, env.n_a))
    else:
        Q = model

    # Train or eval
    total_rewards = 0
    for i_episode in range(n_episodes):
        epsilon = get_epsilon(i_episode)
        lr = get_lr(i_episode)

        s = env.reset()
        done = False
        rewards = 0
        while not done:
            # Choose actoin & observe new state and reward
            a = choose_action(Q, s, epsilon)
            sp, r, done, info = env.step(a)
            rewards += r

            if not is_eval:
                # Update Q
                Q = update(Q, s, a, r, sp, lr, gamma)

            # Check end of episode
            if done:
                env.reset()
                break

            # Transition
            s = sp
        total_rewards += rewards
        print('Rewards: {}, total: {}'.format(rewards, total_rewards))

    avg_rewards = total_rewards / n_episodes
    return avg_rewards, Q
