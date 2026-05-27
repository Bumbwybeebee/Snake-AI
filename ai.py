import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)


# ── Original flat-input network (kept for play.py / backward compat) ──────────

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        return self.linear2(x)

    def save(self, checkpoint, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(checkpoint, file_path)
        print(f"--> Brain saved successfully to {file_path}")


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma

    def train_step(self, state, action, reward, next_state, dead):
        state      = torch.tensor(state,      dtype=torch.float, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=device)
        action     = torch.tensor(action,     dtype=torch.long,  device=device)
        reward     = torch.tensor(reward,     dtype=torch.float, device=device)

        if len(state.shape) == 1:
            state      = torch.unsqueeze(state,      0)
            next_state = torch.unsqueeze(next_state, 0)
            action     = torch.unsqueeze(action,     0)
            reward     = torch.unsqueeze(reward,     0)
            dead = (dead,)

        pred = self.model(state)
        target = pred.clone()
        with torch.no_grad():
            next_q = self.model(next_state).max(dim=1).values
        dead_tensor = torch.tensor(dead, dtype=torch.bool, device=device)
        Q_new = reward + self.gamma * next_q * (~dead_tensor)
        action_indices = torch.argmax(action, dim=1)
        target[torch.arange(len(dead_tensor), device=device), action_indices] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


class Agent:
    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer
        self.memory = deque(maxlen=100_000)
        self.batch_size = 512

    def remember(self, state, action, reward, next_state, dead):
        self.memory.append((state, action, reward, next_state, dead))

    def train_long_memory(self):
        if len(self.memory) < self.batch_size:
            return
        mini_sample = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, deads = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, deads)


# ── Convolutional network ──────────────────────────────────────────────────────

class Conv_QNet(nn.Module):
    def __init__(self, res, flat_input_size, hidden_size, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        conv_flat_size = 64 * (res * res)

        self.fc_flat = nn.Linear(flat_input_size, 64)
        self.fc1     = nn.Linear(conv_flat_size + 64, hidden_size)
        self.fc2     = nn.Linear(hidden_size, output_size)

    def forward(self, grid, flat):
        x = F.relu(self.conv1(grid))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)

        f = F.relu(self.fc_flat(flat))

        combined = torch.cat([x, f], dim=1)
        combined = F.relu(self.fc1(combined))
        return self.fc2(combined)

    def save(self, checkpoint, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(checkpoint, file_path)
        print(f"--> Brain saved successfully to {file_path}")


class QTrainerCNN:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma

    def train_step(self, grids, flats, actions, rewards, next_grids, next_flats, deads):
        grids      = torch.tensor(np.array(grids),      dtype=torch.float, device=device)
        flats      = torch.tensor(np.array(flats),      dtype=torch.float, device=device)
        next_grids = torch.tensor(np.array(next_grids), dtype=torch.float, device=device)
        next_flats = torch.tensor(np.array(next_flats), dtype=torch.float, device=device)
        actions    = torch.tensor(np.array(actions),    dtype=torch.long,  device=device)
        rewards    = torch.tensor(np.array(rewards),    dtype=torch.float, device=device)

        # Handle single-sample case
        if grids.dim() == 3:
            grids      = grids.unsqueeze(0)
            next_grids = next_grids.unsqueeze(0)
            flats      = flats.unsqueeze(0)
            next_flats = next_flats.unsqueeze(0)
            actions    = actions.unsqueeze(0)
            rewards    = rewards.unsqueeze(0)
            deads = (deads,)

        pred = self.model(grids, flats)
        target = pred.clone()

        with torch.no_grad():
            next_q = self.model(next_grids, next_flats).max(dim=1).values
        dead_tensor    = torch.tensor(deads, dtype=torch.bool, device=device)
        Q_new          = rewards + self.gamma * next_q * (~dead_tensor)
        action_indices = torch.argmax(actions, dim=1)
        target[torch.arange(len(dead_tensor), device=device), action_indices] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


class AgentCNN:
    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer
        self.memory = deque(maxlen=100_000)
        self.batch_size = 512

    def remember(self, grid, flat, action, reward, next_grid, next_flat, dead):
        self.memory.append((grid, flat, action, reward, next_grid, next_flat, dead))

    def train_long_memory(self):
        if len(self.memory) < self.batch_size:
            return
        mini_sample = random.sample(self.memory, self.batch_size)
        grids, flats, actions, rewards, next_grids, next_flats, deads = zip(*mini_sample)
        self.trainer.train_step(grids, flats, actions, rewards, next_grids, next_flats, deads)