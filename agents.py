import torch
import random

class Softmax_Agent(torch.nn.Module):
    def __init__(self, model=None, batch=128, n_agents=1):
        super(Softmax_Agent, self).__init__()
        self.model = model
        self.memory = []
        self.n_agents = n_agents
        self.batch = batch
        self.transition_keys = ["state", "action", "reward", "logit", "entropy", "value"]
        self.p_state = 0

    def push(self, transition):
        self.memory.push(transition)

    def step(self, x, memory):
        ys = []
        infos = {}

        y, logit, entropy = self.forward(x)
        memory.add_value(("logit",logit[0]))
        memory.add_value(("entropy",entropy[0]))
        return y, infos

    def forward(self, x):
        actions = []
        logits = []
        entropies = []

        if type(self.p_state) != torch.tensor:
            x = torch.cat((x, x),dim=1)
        else:
            x = torch.cat((x, self.p_state),dim=1)

        self.p_state = x
        y = self.model(x)
        y = torch.nn.Softmax(dim=1)(y)
        for z in list(y):
            dist = torch.distributions.Categorical(z)
            action = dist.sample()
            logit = -dist.log_prob(action).view(-1)
            logits.append(logit)
            actions.append(action)
            entropies.append(dist.entropy())

        return actions, logits, entropies

class RandomAgent():
    def __init__(self, actions):
        super(RandomAgent, self).__init__()
        self.actions = actions

    def forward(self, x):
        return (random.choice(range(self.actions)))
