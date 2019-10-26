import gym
import numpy as np
import copy
import random
import torch
import time

#used for neural episodic control
#
#todo: implement episode replay
#todo: implement update NEC w/ belleman
#todo: implement neural trainer
#class Actor(torch.nn.Module)

class Episodic_Controller():
    def __init__(self, k, n_actions=0):
        super(Episodic_Controller, self).__init__()
        if n_actions <= 1:
            raise Exception("Needs more than 1 action")

        self.brain = {}
        self.states = 0
        self.k = k
        self.actions = n_actions

    def read(self, x):
        keys = self.nearest_neighbors(x)
        actions = torch.tensor([self.brain[key[1]] for key in keys]).mean(dim=0)
        return actions

    def write(self, obv):
        obv = obv
        #initialize state pair
        #start zero or random values?
        self.brain[obv] = np.zeros(self.actions)

    def nearest_neighbors(self, obv):
        keys = list(self.brain.keys())
        #diff = torch.nn.functional.cosine_similarity(keys, obv, dim=1, eps=1e-8)?
        diff = torch.abs(obv-torch.stack(keys)).sum(dim=1)
        keys = list(zip(list(diff), keys))
        keys.sort(key=lambda value: value[0])
        return keys[:self.k]

    #agent takes a step
    #reading and finding nearest neighbors
    #taking action, env step, return rewards
    def step(self, obv):
        obv = torch.tensor(obv).int().view(-1)
        if obv not in self.brain:
            if len(list(self.brain.keys())) < self.k:
                action = [random.choice(list(range(self.actions)))]
            else:
                action = self.read(obv)
                action = [int(torch.argmax(action))]
            #write new obervation only after kNN search
            self.write(obv)
        return action

    def update_model(self, state_value_pairs):
        print("pairs",state_value_pairs[0])
        print("Keys:",len(self.brain.keys()))
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
