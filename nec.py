from coach import Coach
from controllers import Episodic_Controller
import torch
import gym
import boardgame2
import torch
import numpy

discount = 0.99
batch_size = 64
n_agents = 1
n_episodes = 10
n_epochs = 100
rate = 0.001
k=100

coach = Coach(env="CartPole-v0",
              loss_fn=Coach.reinforce,
              lr=rate,
              optim=torch.optim.Adam,
              n_agents=n_agents
              )

agent = Episodic_Controller(n_actions=coach.actions, k=k)

for epoch in range(n_epochs):

    for episode in range(n_episodes):
        returns = coach.run_episode(agent, return_tau=True, render=True)
