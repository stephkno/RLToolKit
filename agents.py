import torch
import random


class Softmax_Agent(torch.nn.Module):
    def __init__(self, model=None, batch=128, head=None):
        super(Softmax_Agent, self).__init__()
        self.model = model
        self.n_agents = 1
        self.batch = batch
        self.transition_keys = ["state", "action", "reward", "logit", "entropy", "value"]
        self.p_state = 0
        self.single_step = False
        self.head = head
        self.steps = 0
        self.action = -1

    def step(self, x, memory, confidence, render, explore=False):
        ys = []
        infos = {}

        y, logit, entropy = self.forward(x, confidence, render, explore)
        memory.add_value(("logit",logit[0]))
        memory.add_value(("entropy",entropy[0]))
        self.steps += 1
        return y, infos

    def forward(self, x, confidence=1.0, render=False, explore=False):
        actions = []
        logits = []
        entropies = []

        if type(self.p_state) != torch.tensor:
            x = torch.cat((x, x),dim=1)
        else:
            x = torch.cat((x, self.p_state),dim=1)

        self.p_state = x

        y = self.model(x)

        if self.head != None:
            y = self.head(y.view(self.n_agents,-1))

        y = torch.nn.Softmax(dim=1)(y.view(self.n_agents,-1))

        for z in list(y):
            dist = torch.distributions.Categorical(z)
            anxiety = torch.nn.Sigmoid()(torch.randn(1))

            if confidence < anxiety or self.steps < 1:
                self.action = dist.sample()
                confidence = 5.0
                mode = "Explore"
            else:
                if explore:
                    pass
                else:
                    self.action = torch.argmax(dist.probs)
                mode = "Exploit"

            #if render:
            #    print("{}% Action {}".format(int(100*y[0][action]), action))

            logit = -dist.log_prob(self.action).view(-1)
            logits.append(logit)
            actions.append(self.action)
            entropies.append(dist.entropy())
            if self.single_step:
                print("Step:{} Action:{} Prob:{} LogProb:{} Confidence:{} Mode:{}".format(self.steps, self.action, z[self.action], logit, confidence, mode))
                input()

        return actions, logits, entropies

class Softmax_RNN_Agent(torch.nn.Module):
    def __init__(self, in_features, hidden, layers, model=None, batch=128, n_agents=1):
        super(Softmax_RNN_Agent, self).__init__()
        self.model = model
        self.hidden_size = hidden
        self.layers = layers
        self.rnn = torch.nn.GRU(in_features, hidden, layers)
        self.memory = []
        self.n_agents = n_agents
        self.batch = batch
        self.transition_keys = ["state", "action", "reward", "logit", "entropy", "value"]
        self.p_state = 0
        self.hidden = torch.zeros(self.layers, 1, self.hidden_size)

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
        y,self.hidden = self.rnn(x, self.hidden)
        y = self.model(y)

        y = torch.nn.Softmax(dim=2)(y)

        for z in list(y):
            dist = torch.distributions.Categorical(z)
            action = dist.sample()
            logit = -dist.log_prob(action).view(-1)
            logits.append(logit)
            actions.append(action)
            entropies.append(dist.entropy())

        return actions, logits, entropies

    def reset_hidden(self):
        self.hidden = torch.zeros(self.layers, 1, self.hidden_size)

class RandomAgent():
    def __init__(self, actions):
        super(RandomAgent, self).__init__()
        self.actions = actions
        self.transition_keys = ["state","reward","action"]

    def forward(self, x):
        return (random.choice(range(self.actions)))
