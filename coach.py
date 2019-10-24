import gym
#
# Reinforcement Learning Coach
#
class Coach():
    def __init__(self, env, target_agent, loss_fn, lr, optim, n_agents):
        super(Coach, self).__init__()
        self.name = env
        self.envs = []
        self.n_agents = n_agents
        self.dones = [False for _ in range(n_agents)]
        self.length = 0

        #create environments
        for _ in range(self.n_agents):
            self.envs.append(gym.make(self.name))

    #reset all environments
    def reset(self):
        states = []

        for env in self.envs:
            states.append(env.reset())

        return states

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

    def reinforce(self, gamma):
        pass

    def run_episode(self, return_tau=False, render=False):
        pass

    def train(self, gamma=0.99, epochs=10):
        pass
