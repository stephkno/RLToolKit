#!/Users/stephen/miniconda3/bin/python
from coach import Coach
from controllers import Agent_Controller
from agents import RandomAgent
import torch
import gym
import boardgame2
import torch
import numpy

n_agents = 2
n_episodes = 10

coach = Coach(env="CartPole-v0",
              loss_fn=Coach.reinforce,
              lr=rate,
              optim=torch.optim.Adam,
              n_agents=n_agents
              )

target_agent = RandomAgent(actions=coach.actions)

agent = Agent_Controller(
            batch=batch_size,
            n_agents=n_agents,
            agent=target_agent
            )

for epoch in range(n_epochs):
    for episode in range(n_episodes):
        returns = coach.run_episode(agent, return_tau=True, render=True)
        print(returns)

    loss = coach.train(gamma=discount, epochs=n_epochs)
