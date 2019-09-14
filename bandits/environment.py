import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats

from bandits.agent import BetaAgent


class Environment(object):
    """ A environment is a combination o a bandit and a number of agents in 
    which experiments are made. An environment object can run a number of trials 
    and take the average of a number of experiments.
    """

    def __init__(self, bandit, agents, label='Multi-Armed Bandit'):
        """ Initializes Environment object
        
        Arguments:
            bandit {Bandit} -- bandit object
            agents {list(Agent)} -- list of agents to be part of experimentations
        
        Keyword Arguments:
            label {str} -- (default: {'Multi-Armed Bandit'})
        """
        self.bandit = bandit
        self.agents = agents
        self.label = label

    def reset(self):
        """ resets bandit and all agents
        """
        self.bandit.reset()
        for agent in self.agents:
            agent.reset()

    def run(self, trials=100, experiments=1):
        """ Runs experiments on environment 
        
        Keyword Arguments:
            trials {int} -- number of trials (default: {100})
            experiments {int} -- number of experiments to be averaged (default: {1})
        
        Returns:
            [np.array] -- average scores by agents for each trial
            [np.array] -- average optimal of scores for each agent
            
        """
        scores = np.zeros((trials, len(self.agents)))
        optimal = np.zeros_like(scores)

        for _ in range(experiments):
            self.reset()
            for trial_index in range(trials):
                for agent_index, agent in enumerate(self.agents):
                    action = agent.choose()
                    reward, is_optimal = self.bandit.pull(action)
                    agent.observe(reward)

                    scores[trial_index, agent_index] += reward
                    if is_optimal:
                        optimal[trial_index, agent_index] += 1

        return scores / experiments, optimal / experiments

    def plot_results(self, scores, optimal, figsize=(16,10)):
        sns.set_style('white')
        sns.set_context('talk')
        fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=figsize)
        ax[0].set_title(self.label)
        ax[0].plot(scores)
        ax[0].set_ylabel('Average Reward')
        ax[0].legend(self.agents, loc=4)
        
        ax[1].plot(optimal * 100)
        ax[1].set_ylim(0, 100)
        ax[1].set_ylabel('% Optimal Action')
        ax[1].set_xlabel('Time Step')
        ax[1].legend(self.agents, loc=4)
        sns.despine()
        plt.show()

    def plot_beliefs(self):
        sns.set_context('talk')
        pal = sns.color_palette("cubehelix", n_colors=len(self.agents))
        plt.title(self.label + ' - Agent Beliefs')

        rows = 2
        cols = int(self.bandit.k / 2)

        axes = [plt.subplot(rows, cols, i+1) for i in range(self.bandit.k)]
        for i, val in enumerate(self.bandit.action_values):
            color = 'r' if i == self.bandit.optimal else 'k'
            axes[i].vlines(val, 0, 1, colors=color)

        for i, agent in enumerate(self.agents):
            if type(agent) is not BetaAgent:
                for j, val in enumerate(agent.value_estimates):
                    axes[j].vlines(val, 0, 0.75, colors=pal[i], alpha=0.8)
            else:
                x = np.arange(0, 1, 0.001)
                y = np.array([stats.beta.pdf(x, a, b) for a, b in
                             zip(agent.alpha, agent.beta)])
                y /= np.max(y)
                for j, _y in enumerate(y):
                    axes[j].plot(x, _y, color=pal[i], alpha=0.8)

        min_p = np.argmin(self.bandit.action_values)
        for i, ax in enumerate(axes):
            ax.set_xlim(0, 1)
            if i % cols != 0:
                ax.set_yticklabels([])
            if i < cols:
                ax.set_xticklabels([])
            else:
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(['0', '', '0.5', '', '1'])
            if i == int(cols/2):
                title = '{}-arm Bandit - Agent Estimators'.format(self.bandit.k)
                ax.set_title(title)
            if i == min_p:
                ax.legend(self.agents)

        sns.despine()
        plt.show()
