import numpy as np


class Policy(object):
    """ Base class for Policy. A policy prescribes an action to be taken based
    on the memory of an agent.
    """
    def __str__(self):
        return 'generic policy'

    def choose(self, agent):
        """ Chooses action index for an agent
        
        Arguments:
            agent {Agent} -- agent to take the action based on a policy
        
        Returns:
            [int] -- action index
        """
        return 0

    def reset(self):
        # Only applicable to epsilon-greedy with annealing
        pass

class EpsilonGreedyPolicy(Policy):
    """
    The Epsilon-Greedy policy will choose a random action with probability
    epsilon and take the best apparent approach with probability 1-epsilon. If
    multiple actions are tied for best choice, then a random action from that
    subset is selected.
    """
    def __init__(self, epsilon, annealing=False, annealing_rate=.95):
        """Initializes EpsilonGreedyPolicy object
        
        Arguments:            
            epsilon {float} -- random action probability in (0,1)
        
        Keyword Arguments:
            annealing {bool} -- whether or not epsilon decreases (default: {False})
            annealing_rate {float} -- rate at which epsilon decreases (default: {.95})
        """
        self.epsilon = epsilon
        self.init_epsilon = epsilon
        self.annealing = annealing
        self.annealing_rate = annealing_rate        

    def __str__(self):
        if (not self.annealing):
            return '\u03B5-greedy (\u03B5={})'.format(self.init_epsilon)
        return '\u03B5-greedy (\u03B5={}, annealing={}, annealing_rate={})'.format(
                    self.init_epsilon, self.annealing, self.annealing_rate)

    def choose(self, agent):  
        if (self.annealing):
            self.anneal()

        if np.random.random() < self.epsilon:
            return np.random.choice(len(agent.value_estimates))
        else:
            action = np.argmax(agent.value_estimates)
            check = np.where(agent.value_estimates == action)[0]
            if len(check) == 0:
                return action
            else:
                return np.random.choice(check)

    def anneal(self):
        self.epsilon *= self.annealing_rate

    def reset(self):
        self.epsilon = self.init_epsilon

class GreedyPolicy(EpsilonGreedyPolicy):
    """
    The Greedy policy only takes the best apparent action, with ties broken by
    random selection. This can be seen as a special case of EpsilonGreedy where
    epsilon = 0 i.e. always exploit.
    """
    def __init__(self):
        super(GreedyPolicy, self).__init__(0)

    def __str__(self):
        return 'greedy'


class RandomPolicy(EpsilonGreedyPolicy):
    """
    The Random policy randomly selects from all available actions with no
    consideration to which is apparently best. This can be seen as a special
    case of EpsilonGreedy where epsilon = 1 i.e. always explore.
    """
    def __init__(self):
        super(RandomPolicy, self).__init__(1)

    def __str__(self):
        return 'random'


class UCBPolicy(Policy):
    """
    The Upper Confidence Bound algorithm (UCB1). It applies an exploration
    factor to the expected value of each arm which can influence a greedy
    selection strategy to more intelligently explore less confident options.
    """
    def __init__(self, c):
        """ Initializes UCBPolicy object
        
        Arguments:            
            c {float} -- degree of exploration control (c > 0)
        """
        self.c = c

    def __str__(self):
        return 'UCB (c={})'.format(self.c)

    def choose(self, agent):
        exploration = np.log(agent.t+1) / agent.action_attempts
        exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, 1/self.c)

        q = agent.value_estimates + exploration
        action = np.argmax(q)
        check = np.where(q == action)[0]
        if len(check) == 0:
            return action
        else:
            return np.random.choice(check)


class SoftmaxPolicy(Policy):
    """
    The Softmax policy converts the estimated arm rewards into probabilities
    then randomly samples from the resultant distribution. This policy is
    primarily employed by the Gradient Agent for learning relative preferences.
    """
    def __str__(self):
        return 'SM'

    def choose(self, agent):        
        a = agent.value_estimates
        pi = np.exp(a) / np.sum(np.exp(a)) # prob of taking action a on time t
        cdf = np.cumsum(pi)
        s = np.random.random()
        return np.where(s < cdf)[0][0]
