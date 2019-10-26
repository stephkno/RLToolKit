import gym
import datetime
import numpy as np
import torch
import cv2

## Reinforcement Learning Coach
###
### Handles OpenAI Gym environments
#

class Embedder(torch.nn.Module):
    def __init__(self):
        super(Embedder, self).__init__()
        self.embed = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 8, 2),
            torch.nn.ELU(),
            torch.nn.Conv2d(32, 64, 4, 2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ELU(),
            torch.nn.Conv2d(64, 128, 4, 2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ELU(),
            torch.nn.Conv2d(128, 512, 5),
            torch.nn.ELU(),
        )
        self.decode = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 128, 5),
            torch.nn.BatchNorm2d(128),
            torch.nn.ELU(),
            torch.nn.ConvTranspose2d(128, 64, 4, 2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ELU(),
            torch.nn.ConvTranspose2d(64, 64, 4, 2),
            torch.nn.ELU(),
            torch.nn.ConvTranspose2d(64, 64, 4, 2),
            torch.nn.ELU(),
            torch.nn.ConvTranspose2d(64, 32, 4),
            torch.nn.BatchNorm2d(32),
            torch.nn.ELU(),
            torch.nn.ConvTranspose2d(32, 1, 8),
        )
        self.optimizer = torch.optim.Adam(lr=1e-3, params=self.parameters())
        self.train = []
        self.targets = []
        self.epochs = 2
        self.count = 0
        self.batch = 5

    def forward(self,x):
        y = self.embed(x)
        train = self.decode(y)
        self.train.append(train)
        self.targets.append(x)
        if self.count % self.batch == 0:
            self.update_embedding()

        preview = train.squeeze(0).squeeze(0).detach().numpy()
        preview = cv2.resize(preview, (10 * 45, 10 * 45), interpolation=cv2.INTER_AREA)
        cv2.imshow("state", preview)
        cv2.waitKey(1)
        #cv2.imshow("state", x.squeeze(0).squeeze(0).detach().numpy())
        #cv2.waitKey(1)
        self.count+=1
        return y

    def update_embedding(self):
        train = torch.stack(self.train)
        targets = torch.stack(self.targets)

        for _ in range(self.epochs):
            loss = torch.nn.functional.smooth_l1_loss(train, targets)
            loss.backward(retain_graph=True)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.train.clear()
            self.targets.clear()


class Coach():
    def __init__(self, env, loss_fn, optim, n_agents, lr=0.1, target_agent=None, embed=False):

        if n_agents < 1:
            raise Exception("Needs at least 1 agent.")

        super(Coach, self).__init__()
        self.name = env
        self.n_agents = n_agents
        self.dones = [False for _ in range(n_agents)]
        self.length = 0
        self.memory = []
        self.embedder = Embedder()
        self.embedding = embed

        if not target_agent == None:
            self.loss_fn=loss_fn
            self.lr=lr
            self.optim=optim(lr=self.lr, params=self.target_agent.params())
            self.target_agent=target_agent

        #create environments
        self.envs = [gym.make(self.name) for _ in range(self.n_agents)]
        self.actions = self.envs[0].action_space.n

    def preprocess_frame(self, state):
        state = (np.sum(state, axis=2) / 3.0)/255
        state = cv2.resize(state, (64, 64))
        state = torch.tensor(state).float()
        if self.embedding:
            state = self.embedder.forward(state.unsqueeze(0).unsqueeze(0))
        return state

    #reset all environments
    def reset(self):
        self.memory.clear()
        return [env.reset() for env in self.envs]

    def end(self):
        [env.close() for env in self.envs]

    #environment step
    def step(self, actions):
        obs = []
        rewards = []
        dones = []
        infos = []

        #take step for each agent/environment
        for i,env in enumerate(self.envs):
            obv, reward, done, info = env.step(actions[i])
            obs.append(obv)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        #send each return as separate vecs
        return obs, rewards, dones, infos

    #loss function for vanilla reinforce algorithm
    def reinforce(self, gamma):
        raise NotImplementedError

    #generate single trajectory
    def run_episode(self, agent, return_tau=False, render=False):
        state = self.reset()[0]
        observation = self.preprocess_frame(state)
        steps = np.zeros(self.n_agents)
        rewards = np.expand_dims(np.zeros(self.n_agents),axis=0)

        while True:
            action = agent.step(observation)
            state = self.step(action)
            observation, reward, done, info = state
            observation = self.preprocess_frame(observation[0])
            dones = [not d for d in done]

            steps += dones * np.ones(self.n_agents)
            rewards += dones * np.array([reward for _ in range(self.n_agents)])

            self.memory.append((observation, action, reward))

            if render:
                [env.render() for env in self.envs]
            if np.prod(done, 0):
                break

        return rewards, steps

    def discount(self, gamma):
        R = 0.0
        values = []
        rewards = list(reversed(list(zip(*self.memory))))[0]
        states = list(reversed(list(zip(*self.memory))))[2]

        for r in rewards:
            R = r[0] + R * gamma
            values.insert(0,R)
        print(len(states), len(values))
        return list(zip(states,values))
