import argparse
import sys

from .model import random, q_learning, mle_vi, pomdp
from .env import SimEnv
from . import params

N_EPISODES = 1000

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shuttle Scheduling Model Training.')
    parser.add_argument('--algo', type=str, default='random',
                        help='One of random, q-learning, mle-vi, and pomdp. Default: random')
    args = parser.parse_args()

    if args.algo == 'random':
        algo = random.train
    elif args.algo == 'q-learning':
        algo = q_learning.train
    elif args.algo == 'mle-vi':
        algo = mle_vi.train
    elif args.algo == 'pomdp':
        algo = pomdp.train
    else:
        print('Algorithm not supported.')
        sys.exit(1)

    env = SimEnv(params.N_S, params.N_A, params.MAX_R, params.PENALTY, params.W1, params.W2, params.W3, params.N_SHUTTLES)
    
    rewards = algo(env, N_EPISODES)
    print('Num of episodes: {}, rewards: {}'.format(N_EPISODES, rewards))
