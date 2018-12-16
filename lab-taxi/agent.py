import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

        self.alpha = 0.5 #0.8
        self.gamma = 0.8
        self.epsilon = 0.5 # 0.1
        self.alpha_decay = 0.999 #0.9999
        self.epsilon_decay = 0.9998 #0.9999
        self.alpha_min = 0.001
        self.epsilon_min = 0.00001

    def select_action(self, state, i_episode, num_episodes):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if i_episode % (num_episodes/2000) == 0:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        if i_episode % (num_episodes/200) == 0:
            self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
        if i_episode % (num_episodes/20) == 0:
            print(self.alpha, self.epsilon)
        # epsilon-greedy
        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        best_s = np.argmax(self.Q[state])
        policy_s[best_s] = 1 - self.epsilon + self.epsilon /self.nA

        return np.random.choice(np.arange(self.nA), p=policy_s)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        Qsa_next = np.max(self.Q[next_state])
        target = reward + self.gamma * Qsa_next
        self.Q[state][action] = self.Q[state][action] + self.alpha * (target - self.Q[state][action])