from coach import Coach
from agents.softmax_agent import Softmax_Agent
import torch
import gym
import torch
from termcolor import colored
import numpy
import cv2
import numpy as np
import random

in_features = 2
hidden = 512
layers = 2
discount = 0.95
batch_size = 64
update_interval = 1
explore_interval = 10
episodes = 8
epochs = 20000
rate = 0.001
game = 0
steps = 0
init = 1
mid = 0
cuda = False

def preprocess_frame(state):
    state = cv2.resize(np.array(state[0]), (64, 64)).mean(axis=2)
    state = torch.tensor(state).float() / 255

    swap = state
    state = coach.p_obs + state
    coach.p_obs = swap
    coach.steps += 1

    # state = torch.tensor(pstate) - state
    #mean = state.mean()
    #std = state.std()
    #state = (state - mean) / (std + 1e-10)

    if coach.preview:
        preview = state.numpy()
        preview = cv2.resize(preview, (256, 256))
        cv2.imshow("prev", preview)
        cv2.waitKey(1)

    return state.unsqueeze(0).unsqueeze(0)
def reinforce(gamma, memory, optimizer):

        print("Epoch {}".format(epoch))
        loss = []

        # return list of episodes
        #   episodes: list of dicts
        for episode in memory.get_episodes():
            sample = list(zip(episode["logit"], episode["value"], episode["entropy"]))
            sample = random.sample(sample, len(sample))
            logits, values, entropies = zip(*sample)

            expectation = torch.stack(logits) * torch.tensor(values)# + (0.001*torch.tensor(entropies))
            loss.append(expectation.sum())

        optimizer.zero_grad()

        loss = torch.stack(loss).view(-1).mean()
        loss.backward(retain_graph=True)

        optimizer.step()
        memory.reset()

        return float(loss)

#ready
coach = Coach(env="SpaceInvaders-v4",
              loss_fn=reinforce,
              optim=torch.optim.RMSprop,
              flatten=True,
              frameskip=1,
              batch_size=batch_size,
              preprocess_func=preprocess_frame,
              utility_func=reinforce,
              cuda=cuda,
              init_skip=init,
              mid_skip=mid,
              caution=0.9,
              ale=False,
              render_skip=8
              )

agent = Softmax_Agent(
                          model=torch.nn.Sequential(
                              torch.nn.Conv2d(in_features, 8, 8, 2),
                              #torch.nn.Dropout2d(0.2),
                              torch.nn.ReLU(),

                              torch.nn.Conv2d(8, 16, 4, 2),
                              #torch.nn.Dropout2d(0.2),
                              torch.nn.ReLU(),

                              torch.nn.Conv2d(16, 16, 4, 1),
                              #torch.nn.Dropout2d(0.2),
                              torch.nn.ReLU(),

                              torch.nn.Conv2d(16, 32, 4, 2),
                              #torch.nn.Dropout2d(0.2),
                              torch.nn.ReLU(),

                          ),
                          head=torch.nn.Sequential(
                              torch.nn.Linear(512, 512),
                              torch.nn.Dropout2d(0.2),
                              torch.nn.Tanh(),
                              torch.nn.Linear(512, coach.actions),
                          ),
                          batch=batch_size
                    )

#steady
coach.set_agent(agent, rate)
coach.set_rate(rate)

coach.reset()

episode_count = 0
best = 0

#go
for epoch in range(1,epochs):

    avg_reward = []
    avg_steps = []
    explore = not (epoch%explore_interval==0)

    if explore:
        mode = "Explore"
    else:
        mode = "Exploit"

    print("{} mode.".format(mode))

    for episode in range(episodes):
        rewards, episode_steps = coach.run_episode(
            render=False,
            gamma=discount,
            max_steps=2500,
            explore=explore
        )

        print("Ran {} steps. Returns: {}".format(episode_steps, rewards))
        steps += episode_steps
        game += 1
        episode_count += 1

        if rewards > best:
            best = rewards
            coach.best_episode = coach.last_episode
            coach.best_agent_params = coach.agent
            new = " New!"
        else:
            new = ""
        avg_reward.append(rewards)
        avg_steps.append(episode_steps)
        #print("Game {} Score: {} Best:{}{}".format(episode_count, rewards, best, new))

    print("\n  ⃤[{}:{}/{}][Game {} Steps {}] [Rewards:{}] [Avg Steps:{}] [Best: {}]\n".format(epoch, epoch%explore_interval, explore_interval, game, steps, int(torch.tensor(avg_reward).sum()), torch.tensor(episode_steps).float().mean(), best))

    loss = coach.learn(discount)

    if epoch % update_interval == 0:
        coach.target_net_update()

    print(colored("loss: {}".format(loss),attrs=["reverse"]))
