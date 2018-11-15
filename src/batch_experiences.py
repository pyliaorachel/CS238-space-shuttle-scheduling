import sys
import csv

import numpy as np

from .env import SimEnv


N_S = 365 * 3
N_A = 365
MAX_R = 250
PENALTY = 100
W1 = 5
W2 = 1
W3 = 0.6
N_SHUTTLES = 5

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please provide output filename.')
        sys.exit(1)
    output = sys.argv[1]

    env = SimEnv(N_S, N_A, MAX_R, PENALTY, W1, W2, W3, N_SHUTTLES)
    env.reset()
    experiences, delays = env.batch_experiences(1000)

    with open(output, 'w') as fout:
        writer = csv.writer(fout, delimiter=',')
        writer.writerow(['s', 'a', 'r', 'sp', 'delay'])
        for experience, delay in zip(experiences, delays):
            e, d = np.ndarray.tolist(experience), delay.item()
            writer.writerow(e + [d])
