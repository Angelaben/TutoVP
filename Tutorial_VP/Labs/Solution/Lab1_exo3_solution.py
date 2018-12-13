class Q_solver_contextual(Q_solver) :

    def __init__(self, env, timestep = 1000) :
        Q_solver.__init__(self, env, timestep)
        self.nb_context = env.get_nb_context
        self.Q_table = np.zeros((self.nb_bandits, self.nb_context))
        self.action_count = np.zeros((self.nb_bandits, self.nb_context))
        self.timestep = timestep
        # Parameters initialisation
        self.epsilon = 1.
        self.epsilon_origin = 1.
        self.epsilon_decay = timestep * 0.25
        self.epsilon_min = 0.02
        self.print_delay = 0.1

    def act(self, act_epsilon_greedy, context) :
        if act_epsilon_greedy :
            return np.random.randint(0, self.nb_bandits, 1)[0]  # Transform [Action] to action (int)
        else :
            return np.argmax(self.Q_table, axis = 1)[context]

    def updateQtable(self, action, reward, action_count, context) :
        self.Q_table[action][context] = \
            self.Q_table[action][context] + 1 / action_count * (reward - self.Q_table[action][context])

    def run(self, force_epsilon = None) :
        context = env.reset()
        cumul_reward = 0

        for iteration in trange(1, self.timestep) :
            action = self.act(self.epsilon > np.random.random(), context)
            self.action_count[action] += 1
            reward, context = env.step(action)
            self.updateQtable(action, reward, self.action_count[action][0], context)
            # Update epsilon with linear decay
            self.epsilon = max(self.epsilon - self.epsilon_origin / self.epsilon_decay, self.epsilon_min)
            if force_epsilon is not None :
                self.epsilon = force_epsilon
            cumul_reward += reward
            self.logger.epsilon_log(self.epsilon)
            self.logger.reward_log(reward)
            if iteration % (self.timestep * self.print_delay) == 0 :
                self.logger.mean_reward_log(cumul_reward / (self.timestep * self.print_delay))
                self.logger.plot_log(10, 10)
                cumul_reward = 0