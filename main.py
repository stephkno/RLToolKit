from models import Actor
from utils import Coach
import torch
import gym
import boardgame2
import torch
import numpy

discount = 1.0
batch_size = 64
agents = 3

actor = Actor(torch.optim.Adam,
            lr=0.0001,
            gamma=discount,
            batch=batch_size,
            agents=agents,
            model=torch.nn.Sequential(
            torch.nn.Linear(10, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 3)))

env = Coach(env="CartPole-v0", n=agents)
state = env.reset()

returns = env.step([0,0,0,0])
print(returns)
observation = torch.tensor(numpy.array(returns))

logit = actor.forward(observation)
dist = torch.distributions.Categorical(logit)
action = dist.sample()
logit = -dist.log_prob(action)

print(logit)
env.push({"state":state, "logit":logit, "reward":reward})
