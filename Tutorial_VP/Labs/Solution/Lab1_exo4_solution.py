class DQNAgent :
    def __init__(self, env, state_size, action_size) :

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen = 5000)
        self.gamma = 0.995  # discount rate
        self.exploration_rate = 1.  # exploration rate
        self.original_epsilon = 1.
        self.min_epsilon = 0.01
        self.exploration_rate_decay = 150
        #        self.n_episodes = 2500 * 200
        self.n_game_max = 300
        self.model = self._build_model()
        self.model.summary()
        np.random.seed(42)

    def _build_model(self) :
        input_state = Input((self.state_size,))
        model = Dense(30, activation = "tanh")(input_state)
        q_value = Dense(self.action_size, activation = "linear")(model)
        dqn = Model([input_state], q_value)
        dqn.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.01), loss = "mse")
        return dqn

    def remember(self, state, action, reward, next_state, done) :
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state) :
        if np.random.rand() <= self.exploration_rate :
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_s) :

        input_batch, action_batch, reward_batch, next_state_batch = [], [], [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_s))
        targets = []
        states = []
        for state, action, reward, next_state, done in minibatch :
            # DQN FIT
            vanilla_target = reward
            if not done :
                # Double DQN
                vanilla_target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            vanilla_target_f = self.model.predict(state)
            vanilla_target_f[0][action] = vanilla_target
            # Train the Neural Net with the state and target_f
            states.append(state[0])
            targets.append(vanilla_target_f[0])
        self.vanillaLoss = self.model.fit(np.array(states), np.array(targets), epochs = 1, verbose = 0).history["loss"][
            0]

    def run(self) :
        done = False
        batch_size = 32
        state = env.reset()
        state = np.reshape(state, (1, self.state_size))
        self.cumulative_reward = 0
        score = []
        epsilon_logger = []
        score_average = []
        cpt_game = 0

        for timer in range(100000) :
            action = self.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, (1, self.state_size))
            self.remember(state, action, reward, next_state, done)

            self.cumulative_reward += reward
            state = next_state
            if done :
                cpt_game += 1
                state = env.reset()
                score_average.append(np.mean(score[-10 :]))
                state = np.reshape(state, (1, self.state_size))
                score.append(self.cumulative_reward)
                print("episode: {}, score: {}, e: {:.2} "
                      .format(cpt_game, self.cumulative_reward, self.exploration_rate))
                self.cumulative_reward = 0
                loss = self.replay(batch_size)
                self.exploration_rate = max(
                    self.exploration_rate - (self.original_epsilon / self.exploration_rate_decay), \
                    self.min_epsilon)
                epsilon_logger.append(self.exploration_rate)
            if timer % 1000 == 0 :
                clear_output(True)
                self.model.summary()
                plt.subplot(311)
                axis = plt.gca()
                axis.set_ylim([0, 200])
                plt.plot(score, label = "Cumulative reward ")
                plt.legend()
                plt.grid()
                plt.show()
                plt.subplot(312)

                axis = plt.gca()
                axis.set_ylim([0, 200])
                plt.plot(score_average, label = "Cumulative Mean reward ")
                plt.legend()
                plt.grid()
                plt.show()
                plt.subplot(313)

                axis = plt.gca()
                axis.set_ylim([0, 1])
                plt.plot(epsilon_logger, label = "Epsilon")
                plt.grid()
                plt.legend()
                plt.show()