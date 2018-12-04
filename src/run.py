import argparse
import sys
import time

from .model import random, q_learning, mlrl, sarsa, mlrl_pu
from .env import SimEnv
from . import params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shuttle Scheduling Model Training.')
    parser.add_argument('--algo', type=str, default='random',
                        help='One of random, q-learning, mlrl, sarsa, and mlrl-pu. Default: random')
    parser.add_argument('--n-episodes', type=int, default=100,
                        help='Number of episodes to run. Default: 100')
    args = parser.parse_args()

    if args.algo == 'random':
        algo = random.run
    elif args.algo == 'q-learning':
        algo = q_learning.run
    elif args.algo == 'mlrl':
        algo = mlrl.run
    elif args.algo == 'sarsa':
        algo = sarsa.run
    elif args.algo == 'mlrl-pu':
        algo = mlrl_pu.run
    else:
        print('Algorithm not supported.')
        sys.exit(1)

    env = SimEnv(params.N_S, params.N_A, params.MAX_R, params.PENALTY, params.W1, params.W2, params.W3, params.N_SHUTTLES, params.MU, params.SIGMA)
    
    # Train
    tstart = time.time()
    rewards, model = algo(env, args.n_episodes, params.GAMMA)
    tend = time.time()
    print('Train {}- Num of episodes: {}, rewards: {}, time:{}'.format(args.algo,args.n_episodes, rewards, tend - tstart))

    # Evaluate
    rewards, model = algo(env, 10, params.GAMMA, is_eval=True, model=model)
    print('Eval - Num of episodes: {}, rewards: {}'.format(10, rewards))
