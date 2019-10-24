import gym
import torch
import copy

#used for neural episodic control
class Episodic_Controller():
    def __init__(self):
        super(Episodic_Controller, self).__init__()
        self.brain = {}

    def read(self, x):
        pass

    def write(self, x):
        pass

#used for training agents with pure gradient methods
class Agent_Controller():
    def __init__(self, agent, n_agents=1, batch=128):
        super(Agent_Controller, self).__init__()
        self.batch = batch
        self.n_agents = n_agents
        self.agent = agent
        self.initialize_agents()

    def initialize_agents(self):
        self.agents = [copy.copy(self.agent) for _ in range(self.n_agents)]
        self.target_agent = self.agent

    def step(self, x):
        return [agent.forward(x[i]) for i,agent in enumerate(self.agents)]
