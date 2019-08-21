"""A dual-headed actor/critic model"""
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    """The action and value heads share the first couple of hidden layers.
    This may be useful as both heads should make their decisions on similar features, so
    why should both heads learn how to extract those features?"""

    def __init__(self, n_inputs, n_actions):
        super(ActorCritic, self).__init__()

        self.linear1 = nn.Linear(n_inputs, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)

        self.actor_linear = nn.Linear(64, n_actions)
        self.critic_linear = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return F.relu(self.linear3(x))

    def action_head(self, x):
        x = self(x)
        return F.softmax(self.actor_linear(x), dim=1)

    def value_head(self, x):
        x = self(x)
        return self.critic_linear(x)

    def evaluate(self, x):
        x = self(x)
        actor_head = F.softmax(self.actor_linear(x), dim=1)
        value_head = self.critic_linear(x)
        return actor_head, value_head


class ActorCriticSeparate(nn.Module):
    """The action and value heads share no weights"""

    def __init__(self, n_inputs, n_actions):
        super(ActorCriticSeparate, self).__init__()

        self.action1 = nn.Linear(n_inputs, 64)
        self.action2 = nn.Linear(64, 128)
        self.action3 = nn.Linear(128, 64)
        self.action4 = nn.Linear(64, n_actions)

        self.value1 = nn.Linear(n_inputs, 64)
        self.value2 = nn.Linear(64, 128)
        self.value3 = nn.Linear(128, 64)
        self.value4 = nn.Linear(64, 1)

    def forward(self, x):
        action = F.relu(self.action1(x))
        action = F.relu(self.action2(action))
        action = F.relu(self.action3(action))
        action = F.softmax(self.action4(action), dim=1)

        value = F.relu(self.value1(x))
        value = F.relu(self.value2(value))
        value = F.relu(self.value3(value))
        value = self.value4(value)

        return action, value

    def action_head(self, x):
        action = F.relu(self.action1(x))
        action = F.relu(self.action2(action))
        action = F.relu(self.action3(action))
        return F.softmax(self.action4(action), dim=1)

    def value_head(self, x):
        value = F.relu(self.value1(x))
        value = F.relu(self.value2(value))
        value = F.relu(self.value3(value))
        return self.value4(value)

    def evaluate(self, x):
        return self(x)
