import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


# ----------------------------
# NETWORK
# ----------------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# AGENT
# ----------------------------
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=5000)

        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.05

        self.update_target_every = 50
        self.step_count = 0

    # ----------------------------
    # ACTION
    # ----------------------------
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    # ----------------------------
    # MEMORY
    # ----------------------------
    def remember(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    # ----------------------------
    # TRAINING
    # ----------------------------
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        states = []
        targets = []

        for s, a, r, s_next, done in batch:
            s = torch.FloatTensor(s)
            s_next = torch.FloatTensor(s_next)

            target = r
            if not done:
                target += self.gamma * torch.max(self.target_model(s_next)).item()

            target_f = self.model(s).detach().clone()
            target_f[a] = target

            states.append(s)
            targets.append(target_f)

        states = torch.stack(states)
        targets = torch.stack(targets)

        loss = self.criterion(self.model(states), targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # update target network
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())