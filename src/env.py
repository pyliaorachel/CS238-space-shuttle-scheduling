import numpy as np


class SimEnv:
    def __init__(self, n_s, n_a, max_r, not_finish_penalty, w1, w2, w3, n_shuttles):
        self.n_s = n_s                                  # number of states (number of dates for the session)
        self.n_a = n_a                                  # number of actions (number of dates to choose for launching)
        self.max_r = max_r                              # maximum reward if no delay
        self.not_finish_penalty = not_finish_penalty    # penalty if cannot finish launching all shuttles within the period
        self.w1, self.w2, self.w3 = w1, w2, w3          # weighting for the three risks
        self.n_shuttles = n_shuttles                    # number of shuttles to launch in the whole period

        self.current_s = 0
        self.n_launched = 0
        self.reset()

    def reset(self):
        self.current_s = 0
        self.n_launched = 0
        return self.current_s

    def step(self, action):
        date = action + 1
        delay = int((self.w1 * np.random.rand() / date) \
                    + self.w2 * np.random.rand() * date \
                    + self.w3 * np.random.rand() * date)
        reward = self.max_r - delay

        self.current_s = self.current_s + date + delay
        done = False
        expires = False
        if self.current_s >= self.n_s:
            # Cannot finish launching all shuttles within the required period
            self.current_s = self.n_s - 1
            reward -= self.not_finish_penalty
            expires = True
            done = True
        
        self.n_launched += 1
        if self.n_launched == self.n_shuttles:
            done = True

        return self.current_s, reward, done, { 'delay': delay, 'expires': expires }

    def batch_experiences(self, n):
        experiences = np.zeros((n, 4), dtype=int)  # (s, a, r, sp)
        delays = np.zeros(n, dtype=int)
        for i in range(n):
            s = self.current_s
            a = int(np.random.rand() * self.n_a)
            sp, r, done, info = self.step(a)

            experiences[i] = [s, a, r, sp]
            delays[i] = info['delay']

            if done:
                self.reset()
        return experiences, delays
