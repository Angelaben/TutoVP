class Q_solver_UCB(Q_solver) :

    def __init__(self, env, timestep = 1000) :
        Q_solver.__init__(self, env, timestep)
        self.c = 0.75

    def act(self, time, act_counter) :
        maxer = []
        for action in range(self.nb_bandits) :
            maxer.append(self.Q_table[action] + self.c * np.sqrt(math.log(time) / (1 + act_counter[action])))
        return np.argmax(maxer)

    def run(self, force_epsilon = None) :
        self.boxplotter = np.array([[0]] * self.nb_bandits).tolist()
        env.reset()
        cumul_reward = 0

        for iteration in trange(1, self.timestep) :
            action = self.act(iteration, self.action_count)
            self.action_count[action] += 1
            reward = env.step(action)
            cumul_reward += reward
            self.boxplotter[action].append(reward)
            self.updateQtable(action, reward, self.action_count[action][0])
            self.logger.reward_log(reward)
            if iteration % (self.timestep * self.print_delay) == 0 :
                self.logger.mean_reward_log(cumul_reward / (self.timestep * self.print_delay))
                self.logger.plot_log(5)
                cumul_reward = 0
                plt.boxplot(solver.boxplotter)
                plt.show()