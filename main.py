import models
from utils import Coach
import torch
import gym
import boardgame2
import torch
import numpy

discount = 1.0
batch_size = 64
n_agents = 3
n_episodes = 10
n_epochs = 100
rate = 0.0001

agent = models.Agent_Controller(
            batch=batch_size,
            agents=n_agents,
            model=torch.nn.Sequential(
                    torch.nn.Linear(4, 128),
                    torch.nn.Tanh(),
                    torch.nn.Linear(128, 3)
            ),
            agent_type=models.Softmax_Agent
            )

coach = Coach(env="CartPole-v0",
              target_agent=agent,
              loss_fn=Coach.reinforce,
              lr=rate,
              optim=torch.optim.Adam,
              n_agents=n_agents
              )

for epoch in range(n_epochs):
    for episode in range(n_episodes):
        returns = coach.run_episode(return_tau=True, render=False)
        print(returns)

    loss = coach.train(gamma=discount, epochs=n_epochs)
