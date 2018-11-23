import numpy as np
from tqdm import tqdm


class QLearnPolicy(object):
    def __init__(self, data):
        self.data = data
        self.no_st = 365*3
        self.no_act = 365
        self.Q = np.zeros((self.no_st, self.no_act))
        self.policy = np.zeros(self.no_st)

    def Q_learning(self):
        for row in tqdm(self.data.iterrows(), total=len(self.data)):
            s, a, r, sp,delay = row[1]
            self.Q[s - 1, a - 1] = self.Q[s - 1, a - 1] + 0.1*(r + 0.95 * np.max(self.Q[sp - 1, a - 1]) - self.Q[s - 1, a - 1])

        self._update_policy()

    def _update_policy(self):
        for s in tqdm(range(self.no_st)):
            if not np.any(self.Q[s]):
                self.policy[s] = np.random.randint(1, 366)
            else:
                self.policy[s] = np.argmax(self.Q[s]) + 1

    def output_policy(self):
        with open('large.policy', 'w+') as f:
            f.writelines([str(int(x)) + '\n' for x in self.policy])

    def train(self):
      reward = 0
      for s in tqdm(range(self.no_st)):
        reward = reward+self.Q[s]
      return sum(reward)
