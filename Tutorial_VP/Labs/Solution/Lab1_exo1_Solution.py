class Q_solver_policy(Q_solver) :

    def __init__(self, env, timestep = 1000) :
        Q_solver.__init__(self, env, timestep)
        self.alpha = 0.1

    def act(self, policy) :
        possible_actions = np.array(range(self.nb_bandits))
        return np.random.choice(possible_actions, size = 1, p = policy.flatten() / np.sum(policy))[0]

    def update_preferences(self, preferences, action, reward, mean_reward, policy) :
        preferences[action] += self.alpha * (reward - mean_reward) * (1 - policy[action])
        for act in range(self.nb_bandits) :
            if act != action :
                preferences[act] -= self.alpha * (reward - mean_reward) * policy[act]
        return preferences

    def compute_policy(self, preferences) :
        policy = np.exp(preferences) / np.sum(np.exp(preferences))
        return policy

    def run(self, force_epsilon = None) :
        self.boxplotter = np.array([[0]] * self.nb_bandits).tolist()
        env.reset()
        cumul_reward = 0
        self.preferences = np.zeros((self.nb_bandits, 1))
        average_reward = 0.0
        for iteration in trange(1, self.timestep) :
            policy = self.compute_policy(self.preferences)
            action = self.act(policy)
            reward = env.step(action)
            self.boxplotter[action].append(reward)
            self.logger.reward_log(reward)
            cumul_reward += reward
            average_reward += (reward - average_reward) / iteration
            self.preferences = self.update_preferences(self.preferences, action, reward, average_reward, policy)
            if iteration % (self.timestep * self.print_delay) == 0 :
                self.logger.plot_mean_reward()
                cumul_reward = 0
                plt.boxplot(solver.boxplotter)
                plt.show()