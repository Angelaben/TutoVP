import numpy as np

np.random.seed(42)
class Stationary_Bandit() :

    def __init__(self, n_bandit = 10):#, is_gradient_bandit = False) :
        self.n_bandit = n_bandit
        self.bandit_list = []
        #self.is_gradient_bandit = is_gradient_bandit
        self.generate_bandit()

    def generate_bandit(self) :
        for i in range(self.n_bandit) :
            self.bandit_list.append(NormalBandit())
         #   if self.is_gradient_bandit:
         #       self.bandit_list[i].mean = 4 # Improve perf

    def step(self, action) :
        return self.bandit_list[action].get_reward()

    def reset(self) :
        self.bandit_list = []
        self.generate_bandit()

    @property
    def get_nb_bandit(self):
        return self.n_bandit

    def get_max_bandit(self):
        maxim = -1
        index = -1
        for ind, bandit in enumerate(self.bandit_list):
            if bandit.mean > maxim:
                maxim = bandit.mean
                index = ind
        return index, maxim


    def list_mean_bandit(self):
        for cpt, bandit in enumerate(self.bandit_list):
            print("Bandit :",cpt, " has a mean of ", bandit.mean, " and a sigma of 1")

import math
class NormalBandit() :

    def __init__(self) :
        self.mean = abs(np.random.normal(0, 2))
        self.sigma = 1

    def get_reward(self) :
        return np.random.normal(self.mean, self.sigma)
