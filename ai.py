import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        input_size is how many inputs it takes (ex: position of head, position of apple, etc)
        hidden_size is how many neurons process these inputs (the middle layer)
        output_size is how many different decisions the ai can do
        """
        super().__init__() # initializes the nn.Module (base class for neural network)

        # these linear things are the trainable layers
        self.linear1 = nn.Linear(input_size, hidden_size) #this one is taking in the game information (the input_size) and giving them to the "neuron" layer (the hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size) #this one is taking the processed info from the neuron layer (hidden_size) and condenses it into simple inputs (the output_size)

    def forward(self, x):
        """
        defines how the neural network passes the data through the layers to make a decision
        x is like the function variable (like f(x) = x² + 3 for example)
        so here, x is the inputs (the game information the model is taking in)
        """
        x = F.relu(self.linear1(x))
        """
        self.linear1(x) takes (game inputs) and outputs 256 hidden numbers, which are the computer's representation of the game state it took in
        torch.relu(self.linear1(x)) changes all the negative numbers in the 256 hidden numbers to 0
        this un-linearizes the between the layers of the neural network (basically, since now f(-#) = 0, it can't be linear), which allows it to learn more complex things) (this is really confusing and I don't totally get it)
        """

        return self.linear2(x) # self.linear2(x) takes the processed info from the neuron layer and outputs the final decision (probabilities, whichever number in this array is the highest is the action that it takes)

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
    #Learning
    def train_step(self, state, action, reward, next_state, dead):
        #model prediction
        state = torch.tensor(state, dtype=torch.float, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=device)
        action = torch.tensor(action, dtype=torch.long, device=device)
        reward = torch.tensor(reward, dtype=torch.float, device=device)

        #turning 1d list into 2d array
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            dead = (dead,) #creates iterable tuple for each game in a batch

        pred = self.model(state) #takes in input vector and using current weights and biases calculates best move. I think

        target = pred.clone() #creates copy of prediction variable
        with torch.no_grad():
            next_q = self.model(next_state).max(dim=1).values
        dead_tensor = torch.tensor(dead, dtype=torch.bool, device=device)
        Q_new = reward + self.gamma * next_q * (~dead_tensor)
        action_indices = torch.argmax(action, dim=1)
        target[torch.arange(len(dead_tensor), device=device), action_indices] = Q_new

        #adjusting weights and biases
        self.optimizer.zero_grad() #clears previous math
        loss = self.criterion(target, pred) #calculates error between predicted reward and actual reward
        loss.backward()
        self.optimizer.step()

class QTrainerCNN:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
    #Learning
    def train_step(self, grids, flats, action, reward, next_grids, next_flats, dead):
        #model prediction
        grids = torch.tensor(np.array(grids), dtype=torch.float, device=device)
        flats = torch.tensor(np.array(flats), dtype=torch.float, device=device)
        next_grids = torch.tensor(np.array(next_grids), dtype=torch.float, device=device)
        next_flats = torch.tensor(np.array(next_flats), dtype=torch.float, device=device)
        action = torch.tensor(np.array(action), dtype=torch.long, device=device)
        reward = torch.tensor(np.array(reward), dtype=torch.float, device=device)

        if grids.dim() == 3:
            grids = grids.unsqueeze(0)
            next_grids = next_grids.unsqueeze(0)
            flats = flats.unsqueeze(0)
            next_flats = next_flats.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            dead = (dead,)

        pred = self.model(grids, flats)

        with torch.no_grad():
            next_q = self.model(next_grids, next_flats).max(dim=1).values
        dead_tensor = torch.tensor(dead, dtype=torch.bool, device=device)
        Q_new = reward + self.gamma * next_q * (~dead_tensor)
        action_indices = torch.argmax(action, dim=1)

        target = pred.clone() #creates copy of prediction variable
        action_indices = torch.argmax(action, dim=1)
        target[torch.arange(len(dead_tensor), device=device), action_indices] = Q_new

        #adjusting weights and biases
        self.optimizer.zero_grad() #clears previous math
        loss = self.criterion(target, pred) #calculates error between predicted reward and actual reward
        loss.backward()
        #print(f"loss: {loss.item():.4f}")
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

class Conv_QNet(nn.Module):
    def __init__(self, res, flat_input_size, hidden_size, output_size):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)

        conv_flat_size = 64 * (res * res)

        self.fc_flat = nn.Linear(flat_input_size, 64)

        self.fc1 = nn.Linear(conv_flat_size + 64, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

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
