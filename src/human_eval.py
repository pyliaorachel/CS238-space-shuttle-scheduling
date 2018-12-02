import argparse
import sys
import math

import numpy as np

from .env import SimEnv
from . import params


def choose_action(n_actions):
    action = None
    while action is None:
        action = input('Choose action (1-{}): '.format(n_actions))
        try:
            action = int(action)
            if action >= 1 and action <= n_actions:
                return action - 1
            else:
                print('Invalid action.')
                action = None
        except e:
            print(e)
            action = None

def run(env):
    total_rewards = 0
    i_episode = 0
    while True:
        s = env.reset()
        print('Initial date: {}'.format(s))
        done = False
        rewards = 0
        while not done:
            a = choose_action(env.n_a)
            sp, r, done, info = env.step(a)
            print('Chosen next launch: +{}, Actual launch date: {}, Reward: {}, Delay: {}'.format(a + 1, sp, r, info['delay']))

            rewards += r

            if done:
                env.reset()
                break

            s = sp
        total_rewards += rewards
        print('End of episode. Rewards: {}, total: {}'.format(rewards, total_rewards))

        i_episode += 1

        end = input('End evaluating? (Y/N)')
        if end == 'Y':
            break
 
    avg_rewards = total_rewards / i_episode
    return avg_rewards, i_episode

if __name__ == '__main__':
    env = SimEnv(params.N_S, params.N_A, params.MAX_R, params.PENALTY, params.W1, params.W2, params.W3, params.N_SHUTTLES)
    
    rewards, n_episodes = run(env)
    print('Num of episodes: {}, rewards: {}'.format(n_episodes, rewards))
