import torch
import random
import sys

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

    def step(self, x, memory, confidence, render, explore=False, single_step=False):
        infos = {}

        y, logit, entropy, confidence = self.forward(x, confidence, render, explore, single_step)
        memory.add_value(("logit",logit[0]))
        memory.add_value(("entropy",entropy[0]))
        self.steps += 1
        infos["confidence"] = confidence

        return y, infos

    def forward(self, x, confidence=1.0, render=False, explore=True, single_step=False):
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

            #for digit in z:
            #    print("{}% ".format(int(digit*100)),end="")
            #print("")
            #sys.stdout.flush()

            anxiety = torch.nn.Sigmoid()(torch.randn(1))
            dist = torch.distributions.Categorical(z)
            action = dist.sample()

            if explore:
                if confidence < anxiety or self.steps < 1:
                    if z[action] > z.mean():
                        action = torch.argmax(z)
                    else:
                        confidence = 1.0
                    self.action = action
                    mode = "Sample"
                else:
                    pass
                    mode = "Bypass"
            else:
                mode = "Exploit"

            logit = -dist.log_prob(self.action).view(-1)
            logits.append(logit)
            actions.append(self.action)
            entropies.append(dist.entropy())
            if single_step:
                print("Step:{} Action:{} Prob:{} LogProb:{} Confidence:{} Mode:{}".format(self.steps, self.action, z[self.action], logit, confidence, mode))
                input()

        return actions, logits, entropies, confidence


class RandomAgent():
    def __init__(self, actions):
        super(RandomAgent, self).__init__()
        self.actions = actions
        self.transition_keys = ["state","reward","action"]

    def forward(self, x):
        return (random.choice(range(self.actions)))
