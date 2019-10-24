import gym
import numpy as np
#
## Reinforcement Learning Coach
#
### Handles OpenAI Gym environments
#
class Coach():
    def __init__(self, env, loss_fn, optim, n_agents, lr=0.1, target_agent=None):

        if n_agents < 1:
            raise Exception("Needs at least 1 agent.")

        super(Coach, self).__init__()
        self.name = env
        self.n_agents = n_agents
        self.dones = [False for _ in range(n_agents)]
        self.length = 0

        if not target_agent == None:
            self.loss_fn=loss_fn
            self.lr=lr
            self.optim=optim(lr=self.lr, params=self.target_agent.params())
            self.target_agent=target_agent

        #create environments
        self.envs = [gym.make(self.name) for _ in range(self.n_agents)]
        self.actions = self.envs[0].action_space.n

    #reset all environments
    def reset(self):
        return [env.reset() for env in self.envs]

    def end(self):
        [env.close() for env in self.envs]

    #agent takes a step
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
        pass

    #generate single trajectory
    def run_episode(self, agent, return_tau=False, render=False):
        state = self.reset()
        steps = np.zeros(self.n_agents)
        rewards = np.expand_dims(np.zeros(self.n_agents),axis=0)

        actions = agent.step(state)
        while True:
            state = self.step(actions)
            observation, reward, done, info = state
            dones = [not d for d in done]

            steps += dones * np.ones(self.n_agents)
            rewards += dones * np.array([reward for _ in range(self.n_agents)])

            if render:
                [env.render() for env in self.envs]
            if np.prod(done, 0):
                break

        return rewards, steps

    #run target parameter update with loss fn
    def train(self, gamma=0.99, epochs=10):
        pass
