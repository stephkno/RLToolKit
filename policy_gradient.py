from coach import Coach
from agents import Softmax_Agent
import torch
import gym
import boardgame2
import torch
import numpy

discount = 0.9
batch_size = 512
n_agents = 1
update_interval = 25
games = 20000
rate = 0.001

coach = Coach(env="CartPole-v0",
              loss_fn=Coach.reinforce,
              optim=torch.optim.RMSprop,
              n_agents=n_agents,
              flatten=True,
              )

agent = Softmax_Agent(model=
                            torch.nn.Sequential(torch.nn.Linear(8,32),
                                                torch.nn.Dropout(0.06),
                                                torch.nn.Tanh(),
                                                torch.nn.Linear(32,32),
                                                torch.nn.Dropout(0.06),
                                                torch.nn.Tanh(),
                                                torch.nn.Linear(32, coach.actions),
                                                ),
                        batch=batch_size,
                        n_agents=n_agents
                      )

coach.set_agent(agent, rate)
coach.reset()

episode_count = 0
best = 0

for epoch in range(games):

    avg_reward = []
    avg_steps = []

    for episode in range(update_interval):
        rewards, steps = coach.run_episode(render=(best>200), gamma=discount)
        episode_count += 1

        if rewards > best:
            best = rewards
            coach.best_episode = coach.last_episode
            coach.best_agent_params = coach.agent
            new = " New!"
        else:
            new = ""
        avg_reward.append(rewards)
        avg_steps.append(steps)

        print("Game {} Score: {} Best:{}{}".format(episode_count, rewards, best, new))

    print("{} Avg Reward:{} Avg Steps:{}".format(epoch, torch.tensor(avg_reward).float().mean(), torch.tensor(steps).float().mean()))

    loss = coach.reinforce(discount)
    print(loss)
