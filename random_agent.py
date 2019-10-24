from coach import Coach
from controllers import Agent_Controller
from agents import RandomAgent
import torch
import gym
import boardgame2
import torch
import numpy

N_AGENTS = 1
N_EPISODES = 10
N_EPOCHS = 1

coach = Coach(env="CartPole-v0",
              loss_fn=Coach.reinforce,
              optim=torch.optim.Adam,
              n_agents=N_AGENTS
              )

target_agent = RandomAgent(actions=coach.actions)

agent = Agent_Controller(
            n_agents=N_AGENTS,
            agent=target_agent
            )

for epoch in range(N_EPOCHS):
    for episode in range(N_EPISODES):
        returns = coach.run_episode(agent, return_tau=True, render=True)
        print(returns)
