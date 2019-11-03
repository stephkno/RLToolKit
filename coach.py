import gym
import datetime
import numpy as np
import torch
import cv2
import pygame
from memory import Memory
from embedder import Embedder
pygame.init()

## Reinforcement Learning Coach
###
### Handles OpenAI Gym environments
#
class Coach():
    def __init__(self, env, loss_fn, optim, n_agents, embed=False, flatten=False):

        if n_agents < 1:
            raise Exception("Need at least 1 agent.")
        if n_agents > 1:
            raise NotImplementedError

        super(Coach, self).__init__()
        self.name = env
        self.n_agents = n_agents
        self.dones = [False for _ in range(n_agents)]
        self.length = 0
        self.memory = Memory(n_agents)
        self.episode = []
        self.embedder = Embedder()
        self.embedding = embed
        self.flatten = flatten
        self.best_episode = []
        self.p_obs = torch.zeros(1)
        self.values = []
        self.frameskip = 1
        self.loss_fn=loss_fn
        self.optim = optim
        self.preview = False
        self.norm = False
        self.render = False

        #create environments
        self.envs = [gym.make(self.name) for _ in range(self.n_agents)]
        self.actions = self.envs[0].action_space.n

    def set_agent(self, agent, lr):
        self.agent = agent
        self.optimizer = self.optim(lr=lr, params=self.agent.parameters())
        self.memory.set_keys(agent.transition_keys)

    def preprocess_frame(self, state):
        state = torch.tensor(state).float()

        if self.preview:
            cv2.imshow("prev", state.np())
            cv2.waitKey(1)
        if self.norm:
            mean = state.mean()
            std = state.std()
            state = (state - mean) / (std + 1e-10)
        return state

    def replay_episode(self, episode):
        for t in episode:
            t = cv2.resize(t, (512, 512))
            cv2.imshow("replay", t)
            cv2.waitKey(2)

    #reset all environments
    def reset(self):
        if self.agent == None:
            raise("Need agent")
        states = []
        for env in self.envs:
            state=env.reset()
            states.append(state)

        return states

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
            reward = 0.0
            for _ in range(self.frameskip):
                obv, r, done, info = env.step(int(actions[i]))
                reward += r
            obs.append(obv)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        #send each return as separate vecs
        return obs, rewards, dones, infos

    #loss function for vanilla reinforce algorithm
    def reinforce(self, gamma):
        loss = []

        #return list of episodes
        #   episodes: list of dicts
        for episode in self.memory.get_episodes():
            logits = episode["logit"]
            values = episode["value"]
            entropies = episode["entropy"]
            expectation = torch.stack(logits) * torch.tensor(values) + (0.001*torch.tensor(entropies))
            loss.append(expectation.sum())
        self.optimizer.zero_grad()

        loss = torch.stack(loss).mean().backward(retain_graph=True)
        self.optimizer.step()
        self.memory.reset()
        return loss

    def get_keys(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
            if event.type == pygame.KEYDOWN:
                # gets the key name
                key = pygame.key.name(event.key)
                if key == 'v':
                    self.render = not self.render
                    print("Render {}".format(self.render))
                if key == 'r':
                    self.replay_episode(self.best_episode)

    #generate single trajectory
    def run_episode(self, render=False, gamma=0.9):
        self.last_episode = []

        state = self.reset()
        observation = self.preprocess_frame(state)
        steps = np.zeros(self.n_agents)
        rewards = np.expand_dims(np.zeros(self.n_agents),axis=0)
        score = 0

        while True:
            self.get_keys()

            #gets action/s and replay memory info
            self.memory.add_value(("state",observation.view(-1)))

            action, memory_info = self.agent.step(observation, self.memory)
            obs, reward, done, info = self.step(action)
            score += reward[0]

            self.memory.add_value(("action",action[0]))
            observation = self.preprocess_frame(obs)

            dones = [not d for d in done]

            steps += dones * np.ones(self.n_agents)
            rewards += np.array(dones) * np.array(reward[0])

            if (self.render or render) and steps % 1 == 0:
                [env.render() for env in self.envs]

            for i,r in enumerate(reward):
                self.memory.add_value(("reward",reward[0]))

            if np.prod(done, 0):
                #when episode is finished
                self.memory.finish_episode(gamma)
                break

        #after series of episode, get episodes into own arrays
        return int(score), int(steps)

    def discount(self, gamma):
        R = 0.0

        values = [[] for _ in range(self.n_agents)]
        for step in reversed(self.memory.rewards):
            for i,r in enumerate(step):
                R = r + R * gamma
                values[i].insert(0,R)

        values = torch.tensor(np.array(values))
        return values
