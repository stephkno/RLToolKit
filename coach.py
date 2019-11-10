import gym
import gym_minigrid
import datetime
import numpy as np
import torch
import random
import cv2
import pygame
from memory import Memory
from embedder import Embedder
import os
import copy
import atexit
pygame.init()

## Reinforcement Learning Coach
###
### Handles OpenAI Gym environments
#
class Coach():
    def __init__(self, env, loss_fn, optim, gym=gym, preprocess_func=None, utility_func=None, embed=False, flatten=False, frameskip=1, batch_size=32, cuda=False, ale=False, init_skip=0, mid_skip=0, caution=0.0, render_skip=1):

        self.n_agents = 1

        if utility_func == None:
            raise Exception("Need utility function")
        if preprocess_func == None:
            raise Exception("Need preprocess function")
        if self.n_agents < 1:
            raise Exception("Need at least 1 agent.")
        if self.n_agents > 1:
            raise NotImplementedError

        super(Coach, self).__init__()
        print("Initializing Coach")

        self.ale = ale
        self.name = env
        self.gym = gym
        self.dones = [False for _ in range(self.n_agents)]
        self.length = 0
        self.memory = Memory(self.n_agents)
        self.episode = []
        self.embedder = Embedder()
        self.embedding = embed
        self.flatten = flatten
        self.best_episode = []
        self.p_obs = torch.zeros(1)
        self.values = []
        self.frameskip = frameskip
        self.loss_fn=loss_fn
        self.optim = optim
        self.preview = False
        self.single_step = False
        self.norm = True
        self.render = False
        self.debug_render = False
        self.caution = caution#0.95
        self.life = -1
        self.life = -1
        self.init_skip = init_skip#70
        self.render_frameskip = render_skip
        self.mid_skip = mid_skip#50
        self.steps = 0
        self.batch_size = batch_size
        self.preprocess_frame = preprocess_func
        self.update = utility_func
        self.cuda = cuda

        atexit.register(self.end)
        #create environments
        self.envs = [self.gym.make(self.name) for _ in range(self.n_agents)]
        self.env = 0

        self.actions = self.envs[0].action_space.n

        os.system("toilet {}".format(self.name))

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.Xavier(m.weight.data, 0.0, 0.05)
            torch.nn.fill(m.weight.bias, 1.0)
        if classname.find('Conv2d') != -1:
            torch.nn.init.Xavier(m.weight.data, 1.0, 0.05)
            torch.nn.fill(m.weight.bias, 0.0)

    def set_agent(self, agent, lr):
        self.agent = agent
        self.weights_init(self.agent)
        self.target_agent = copy.copy(agent)
        print(self.agent)

        if lr > 0.0:
            self.optimizer = self.optim(lr=lr, weight_decay=0.99, params=self.target_agent.parameters())
        self.memory.set_keys(agent.transition_keys)

    def set_rate(self, lr):
        print("Learning rate: {}".format(lr))
        self.optimizer = self.optim(lr=lr, params=self.agent.parameters())


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
            env.render()

            state=env.reset()
            for _ in range(self.init_skip):
                state,_,_,info = env.step(1)
            states.append(state)

        if self.ale and ("ale.lives" in info.keys()):
            self.life = info["ale.lives"]
            self.life = self.life
        return states

    def end(self):
        print("\nClosing gym")
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

        #send each return separately
        return obs, rewards, dones, infos

    def get_keys(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
            if event.type == pygame.KEYDOWN:
                # gets the key name
                key = pygame.key.name(event.key)
                if key == 'v':
                    self.debug_render = not self.debug_render
                    print("Render {}".format(self.debug_render))
                if key == 'r':
                    self.replay_episode(self.best_episode)
                if key == 'f':
                    if self.render_frameskip == 1:
                        self.render_frameskip = 10
                        print("Frameskip True")
                    else:
                        self.render_frameskip = 1
                        print("Frameskip False")
                if key == 's':
                    self.single_step = not self.single_step
                    self.debug_render = self.single_step
                    print("Render {}".format(self.debug_render))
                    print("Single step {}".format(self.single_step))


    def render_env(self):
        if self.steps % self.render_frameskip == 0:
            for env in self.envs:
                env.render()

    def learn(self, gamma):
        return self.update(gamma, self.memory, self.optimizer)

    def target_net_update(self):
        print("Agent parameter update.")
        self.agent.load_state_dict(self.target_agent.state_dict())

    def run_episode(self, render=False, gamma=0.99, max_steps=100, explore=False):

        self.render = render

        for env in self.envs:
            env._max_episode_steps = max_steps

        self.last_episode = []

        state = self.reset()
        observation = self.preprocess_frame(state)
        steps = np.zeros(self.n_agents)
        rewards = np.expand_dims(np.zeros(self.n_agents),axis=0)
        score = 0
        confidence = 1.0

        for _ in range(max_steps):
            self.get_keys()

            action, memory_info = self.agent.step(observation, self.memory, confidence, self.render, explore, self.single_step)

            confidence = memory_info['confidence']

            obs, reward, done, info = self.step(action)
            score += reward[0]

            if self.render or self.debug_render:
                self.render_env()

            observation = self.preprocess_frame(obs)

            confidence *= self.caution
            confidence += reward[0]
            confidence = max(min(confidence, 1), 0.2)

            self.memory.add_value(("action",action[0]))

            dones = [not d for d in done]

            steps += dones * np.ones(self.n_agents)
            rewards += np.array(dones) * np.array(reward[0])

            if np.prod(done, 0) or (self.ale and info[0]["ale.lives"] < self.life):
                reward[0] = -1.0
                score -= 1
                for i, r in enumerate(reward):
                    self.memory.add_value(("reward", r))

                self.confidence = 1.0
                self.memory.finish_episode(gamma)
                self.life -= 1
                states = []
                for env in self.envs:
                    for _ in range(self.mid_skip):
                        state, _, _, info = env.step(0)
                    states.append(state)
            else:
                for i, r in enumerate(reward):
                    self.memory.add_value(("reward", r))

            if np.prod(done, 0) or self.life == 0:
                break

        self.agent.steps = 0
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
