import sys
import csv

import numpy as np

from .env import SimEnv
from . import params


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please provide output filename.')
        sys.exit(1)
    output = sys.argv[1]

    env = SimEnv(params.N_S, params.N_A, params.MAX_R, params.PENALTY, params.W1, params.W2, params.W3, params.N_SHUTTLES)
    env.reset()
    experiences, delays = env.batch_experiences(1000)

    with open(output, 'w') as fout:
        writer = csv.writer(fout, delimiter=',')
        writer.writerow(['s', 'a', 'r', 'sp', 'delay'])
        for experience, delay in zip(experiences, delays):
            e, d = np.ndarray.tolist(experience), delay.item()
            writer.writerow(e + [d])
