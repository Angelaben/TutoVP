import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
class Logger:
    def __init__(self):
        self.epsilon_logger = []
        self.reward_logger = []
        self.mean_reward_logger = [0]
        self.cumulative_mean = [0]
    def epsilon_log(self, epsilon):
        self.epsilon_logger.append(epsilon)

    def reward_log(self, reward):
        self.reward_logger.append(reward)
        self.cumulative_mean.append(np.mean(self.reward_logger))
    def mean_reward_log(self, mean_reward):
        self.mean_reward_logger.append(mean_reward)



    def plot_log(self, limit_mean_reward = 10, limit_reward = 5):
        clear_output(True)  # Delete me if you want to make your own print
        plt.subplot(411)
        axis = plt.gca()
        axis.set_ylim([-limit_reward, limit_reward])
        plt.grid()
        plt.plot(self.reward_logger, label = "Reward over time")
        plt.legend()
        plt.subplot(412)
        plt.grid()
        axis = plt.gca()
        axis.set_ylim([-limit_mean_reward, limit_mean_reward])
        plt.plot(self.mean_reward_logger, label = "Mean reward over time")
        plt.legend()
        plt.subplot(413)
        axis = plt.gca()
        axis.set_ylim([0, 1])
        plt.grid()
        plt.plot(self.epsilon_logger, label = "Epsilon over time ")
        plt.legend()
        plt.subplot(414)
        plt.grid()
        axis = plt.gca()
        axis.set_ylim([-limit_mean_reward, limit_mean_reward])
        plt.plot(self.cumulative_mean, label ="Cumulative mean")
        plt.legend()
        plt.show()
        print("Maximum mean over time : ", np.max(self.mean_reward_logger))

    def plot_mean_reward(self):
        clear_output(True)  # Delete me if you want to make your own print
        plt.grid()
        axis = plt.gca()
        axis.set_ylim([-2, 10])
        plt.plot(self.cumulative_mean, label = "Cumulative mean")
        plt.legend()
        plt.show()