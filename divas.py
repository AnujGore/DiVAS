import torch
from torch.nn import init
import math

class divas:
    def __init__(self, input_dim, hidden_dim, **kwargs):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.device = kwargs["device"] if kwargs["device"] is not None else "cpu"

        self.input_to_hidden = torch.empty((self.input_dim, self.hidden_dim)).to(device = self.device)
        self.first_hidden = torch.empty((self.hidden_dim, self.hidden_dim)).to(device = self.device)
        self.second_hidden = torch.empty((self.hidden_dim, self.hidden_dim)).to(device = self.device)
        self.hidden_to_output = torch.empty((self.hidden_dim, self.input_dim)).to(device = self.device)

        self.fitness = torch.nan

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.input_to_hidden, a=math.sqrt(5))
        init.kaiming_uniform_(self.first_hidden, a=math.sqrt(5))
        init.kaiming_uniform_(self.second_hidden, a=math.sqrt(5))
        init.kaiming_uniform_(self.hidden_to_output, a=math.sqrt(5))

    def predict(self, input):
        x = torch.matmul(input, self.input_to_hidden)
        x = torch.matmul(x, self.first_hidden)
        x = torch.matmul(x, self.second_hidden)
        y = torch.matmul(x, self.hidden_to_output)

        return y

    def mutate(self):
        self.input_to_hidden += torch.normal(0.0, 1.0, size = self.input_to_hidden.shape)
        self.first_hidden += torch.normal(0.0, 1.0, size = self.first_hidden.shape)
        self.second_hidden += torch.normal(0.0, 1.0, size = self.second_hidden.shape)
        self.hidden_to_output += torch.normal(0.0, 1.0, size = self.hidden_to_output.shape)    