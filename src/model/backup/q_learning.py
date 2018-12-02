from qlearnpolicy import QLearnPolicy
import time 
import pandas as pd

data = pd.read_csv('data.csv')
def train(env, n_episodes):
    QL = QLearnPolicy(large_data)
    start = time.time()
    QL.Q_learning()
    end = time.time()
    print(end-start)
    reward = QL.train()

    return reward
