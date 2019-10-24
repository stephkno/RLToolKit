import torch
import copy

#used for neural episodic control
class Episodic_Controller():
    def __init__(self):
        super(Episodic_Controller, self).__init__()
        self.brain = {}

    def read(self, x):
        pass

    def write(self, x):
        pass


#used for training agents with pure gradient methods
class Agent_Controller():
    def __init__(self, agents, batch, model, agent_type):
        super(Agent_Controller, self).__init__()
        self.model = model
        self.batch = batch
        self.n_agents = agents
        self.agent_type = agent_type
        self.initialize_agents()

    def initialize_agents(self):
        self.target_agent = self.agent_type(self.model)
        self.agents = [copy.copy(self.target_agent) for _ in range(self.n_agents)]

    def step(self, x):
        return [agent(x) for agent in self.agents]

class Softmax_Agent(torch.nn.Module):
    def __init__(self, model):
        super(Softmax_Agent, self).__init__()
        self.model = model
        self.memory = []

    def push(self, transition):
        self.memory.push(transition)

    def forward(self, x):
        y = self.model(x)
        y = torch.nn.Softmax(dim=-1)(y)
        return y
