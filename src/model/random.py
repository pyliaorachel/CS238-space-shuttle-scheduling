import math

import numpy as np


def choose_action(n_actions):
    return math.ceil(np.random.rand() * n_actions) # cannot be 0

def train(env, n_episodes):
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
 
    avg_rewards = total_rewards / n_episodes
    return avg_rewards
