import gym
#
# Reinforcement Learning Coach
#
class Coach():
    def __init__(self, env, target_agent, loss_fn, lr, optim, n_agents):
        super(Coach, self).__init__()
        self.name = env
        self.n_agents = n_agents
        self.dones = [False for _ in range(n_agents)]
        self.length = 0
        self.loss_fn=loss_fn
        self.lr=lr
        self.optim=optim
        self.target_agent=target_agent

        #create environments
        self.envs = [gym.make(self.name) for _ in range(self.n_agents)]

    #reset all environments
    def reset(self):
        return [env.reset() for _ in self.envs]

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
    def run_episode(self, return_tau=False, render=False):
        pass

    #run target parameter update with loss fn
    def train(self, gamma=0.99, epochs=10):
        pass
