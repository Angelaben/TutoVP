import numpy as np

np.random.seed(42)
class Contextual_bandit() :

    def __init__(self, n_bandit = 10) :
        self.n_bandit = n_bandit
        self.n_context = 2
        self.bandit_list_A = []
        self.bandit_list_B = []
        self.current_state_is_A = np.random.random() < 0.5
        self.generate_bandit()

    def generate_bandit(self) :
        self.bandit_list_A = []
        self.bandit_list_B = []
        for i in range(self.n_bandit):
            self.bandit_list_A.append(NormalBandit(i))
            self.bandit_list_B.append(NormalBandit(self.n_bandit - i))

    def step(self, action) :
        # Proba env A = 0.5
        if self.current_state_is_A:
            self.current_state_is_A = np.random.random() < 0.5
            return self.bandit_list_A[action].get_reward(), int(self.current_state_is_A) # Indicator of the env
        self.current_state_is_A = np.random.random() < 0.5 # Next state will be
        return self.bandit_list_B[action].get_reward(), int(self.current_state_is_A)

    def reset(self) :
        self.bandit_list = []
        self.generate_bandit()
        return int(self.current_state_is_A)

    @property
    def get_nb_bandit(self):
        return self.n_bandit

    @property
    def get_nb_context(self):
        return self.n_context

    def list_mean_bandit(self):
        for cpt, bandit in enumerate(self.bandit_list):
            print("Bandit :",cpt, " has a mean of ", bandit.mean, " and a sigma of 1")

class NormalBandit() :

    def __init__(self, mean = 1) :
        self.mean = mean
        self.sigma = 1

    def get_reward(self) :
        return np.random.normal(self.mean, self.sigma)
