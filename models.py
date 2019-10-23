import torch

class Actor():
    def __init__(self, optim, lr, agents, gamma, batch, model):
        super(Actor, self).__init__()
        self.optimizer = optim(lr=lr, params=self.parameters())
        self.model = model
        self.gamma = gamma
        self.batch = batch
        self.n_agents = agents
        self.agents = [Network(model) for _ in range(self.n_agents)]

class Network(torch.nn.Module):
    def __init__(self, model):
        super(Actor, self).__init__()
        self.model = model
        self.activation = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        y = self.model(x)
        y = self.activation(y)
        return y
