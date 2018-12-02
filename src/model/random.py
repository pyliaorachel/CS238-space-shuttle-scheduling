import math

import numpy as np


def choose_action(n_actions):
    return math.ceil(np.random.rand() * n_actions) # cannot be 0

def run(env, n_episodes, gamma, is_eval=False, model=None):
    total_rewards = 0
    for i_episode in range(n_episodes):
        s = env.reset()
        done = False
        rewards = 0
        while not done:
            a = choose_action(env.n_a)
            sp, r, done, info = env.step(a)

            rewards += r

            if done:
                env.reset()
                break

            s = sp
        total_rewards += rewards
        print('Rewards: {}, total: {}'.format(rewards, total_rewards))
 
    avg_rewards = total_rewards / n_episodes
    return avg_rewards, None
