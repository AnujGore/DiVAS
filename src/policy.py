import torch
import torch.nn as nn
import numpy as np

STALE_REWARD_THRESHOLD = 1e-2  
STALE_REWARD_WINDOW = 10       
SPIKE_STD = 0.2

class FeedForwardNN(nn.Module):
	"""
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""
	def __init__(self, input_size, hidden_layers, output_size):
		"""
		input_size: int, number of input features
		hidden_layers: list of ints, each int is the number of neurons in that hidden layer
		output_size: int, number of output neurons
		"""
		super(FeedForwardNN, self).__init__()
		
		layers = []
		in_features = input_size
		
		# Create hidden layers
		for hidden_size in hidden_layers:
			layers.append(nn.Linear(in_features, hidden_size))
			layers.append(nn.ReLU())
			in_features = hidden_size
		
		# Output layer
		layers.append(nn.Linear(in_features, output_size))
		
		# Combine all layers in a sequential container
		self.network = nn.Sequential(*layers)

	def forward(self, obs):
		"""
			Runs a forward pass on the neural network.

			Parameters:
				obs - observation to pass as input

			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs)

		obs = obs.to(dtype=torch.float32)

		output = self.network(obs)

		return output

class PPO:
	def __init__(self, env, **kwargs):
		self.env = env
		self.num_slots = 24*4
		self.hidden_dim = [128, 256, 128]
		self.num_actions = env.num_tasks
		self.lr = kwargs["learning_rate"]

		input_dim = self.num_slots + self.num_slots

		self.actor = FeedForwardNN(input_dim, self.hidden_dim, self.num_actions)
		self.critic = FeedForwardNN(input_dim, self.hidden_dim, 1)

		self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

def is_stale(reward_history):
    if len(reward_history) < STALE_REWARD_WINDOW:
        return False
    recent_rewards = reward_history[-STALE_REWARD_WINDOW:]
    return torch.std(torch.tensor(recent_rewards)) < STALE_REWARD_THRESHOLD

def spike_actor_network(actor):
    with torch.no_grad():
        for param in actor.parameters():
            param.mul_(SPIKE_STD)
