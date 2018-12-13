class Q_solver() :

    def __init__(self, env, timestep = 1000) :
        self.env = env
        self.nb_bandits = env.get_nb_bandit
        self.Q_table = np.zeros((self.nb_bandits, 1))
        self.action_count = np.zeros((self.nb_bandits, 1))
        self.timestep = timestep
        # Parameters initialisation
        self.epsilon = 1.
        self.print_delay = 0.1
        self.epsilon_origin = 1.
        self.epsilon_min = 0.02
        self.epsilon_decay = timestep * 0.25
        # Logger
        self.logger = Logger()

    def act(self, act_epsilon_greedy) :
        if act_epsilon_greedy :
            return np.random.randint(0, self.nb_bandits, 1)[0]  # Transform [Action] to action (int)
        else :
            return np.argmax(self.Q_table)

    def updateQtable(self, action, reward, action_count) :
        self.Q_table[action] = self.Q_table[action] + 1 / action_count * (reward - self.Q_table[action])

    def run(self, force_epsilon = None) :
        self.boxplotter = np.array([[0]] * self.nb_bandits).tolist()
        if force_epsilon is not None :
            self.epsilon = force_epsilon
        env.reset()
        cumul_reward = 0

        for iteration in trange(1, self.timestep) :
            action = self.act(self.epsilon > np.random.random())
            self.action_count[action] += 1
            reward = env.step(action)
            cumul_reward += reward
            self.boxplotter[action].append(reward)
            self.updateQtable(action, reward, self.action_count[action][0])
            if force_epsilon is None:
                self.epsilon = max(self.epsilon - self.epsilon_origin / self.epsilon_decay, self.epsilon_min)
            self.logger.epsilon_log(self.epsilon)
            self.logger.reward_log(reward)
            if iteration % (self.timestep * self.print_delay) == 0 :
                self.logger.mean_reward_log(cumul_reward / (self.timestep * self.print_delay))
                self.logger.plot_log(2)
                cumul_reward = 0
                plt.boxplot(solver.boxplotter)
                plt.show()