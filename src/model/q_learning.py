from .qlearnpolicy import QLearnPolicy
import time 
import pandas as pd

data = pd.read_csv('data/data.csv')
def run(env, n_episodes, gamma, is_eval=False, model=None):
    if not is_eval:
        QL = QLearnPolicy(large_data)
    else:
        QL = model

    QL.Q_learning()
    reward = QL.train()

    return reward, QL
