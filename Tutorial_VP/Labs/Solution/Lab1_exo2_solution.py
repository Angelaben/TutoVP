class Q_solver_NS(Q_solver) :
    def __init__(self, env, timestep = 1000) :
        Q_solver.__init__(self, env, timestep)
        self.alpha = 0.1

    def updateQtable(self, action, reward, action_count) :
        self.Q_table[action] = self.Q_table[action] + self.alpha * (reward - self.Q_table[action])

    def run(self, force_epsilon = None) :

        if force_epsilon is not None :
            self.epsilon = force_epsilon
        env.reset()
        cumul_reward = 0

        for iteration in trange(1, self.timestep) :
            action = self.act(self.epsilon > np.random.random())
            self.action_count[action] += 1
            reward = env.step(action)
            cumul_reward += reward
            self.updateQtable(action, reward, self.action_count[action][0])
            self.logger.epsilon_log(self.epsilon)
            self.logger.reward_log(reward)
            if iteration % (self.timestep * self.print_delay) == 0 :
                self.logger.mean_reward_log(cumul_reward / (self.timestep * self.print_delay))
                self.logger.plot_log(3, 3)
                cumul_reward = 0