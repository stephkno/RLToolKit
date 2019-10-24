import torch
import random

class Softmax_Agent(torch.nn.Module):
    def __init__(self, model):
        super(Softmax_Agent, self).__init__()
        self.model = model
        self.memory = []

    def push(self, transition):
        self.memory.push(transition)

    def forward(self, x):
        y = self.model(x)
        y = torch.nn.Softmax(dim=-1)(y)
        dist = torch.distributions.Categorical(y)
        action = dist.sample()
        logit = dist.log_prob(action)
        return (y, action, logit)

class RandomAgent():
    def __init__(self, actions):
        super(RandomAgent, self).__init__()
        self.actions = actions

    def forward(self, x):
        return (random.choice(range(self.actions)))
