import gym

class Coach():
    def __init__(self, env, n):
        super(Coach, self).__init__()
        self.name = env
        self.envs = []
        self.n = n
        self.dones = [False for _ in range(n)]
        self.memory = []
        self.length = 0

        for _ in range(self.n):
            self.envs.append(gym.make(self.name))

    def reset(self):
        states = []

        for env in self.envs:
            states.append(env.reset())

        return states

    def step(self, actions):
        return [x.step(actions[i]) for i,x in enumerate(self.envs)]

    def push(self, sample):
        memory.append(sample)
        self.length += 1
