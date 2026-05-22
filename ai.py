import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os

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
    