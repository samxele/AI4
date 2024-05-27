import numpy as np
from game import Game
import heuristics

import random
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

# Instantiate the neural network
class DeepQNetworkConnect4(nn.Module):
    # def __init__(self, env):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, kernel_size=4, stride=1, padding=0)
        self.fc1 = nn.Linear(10*3*4, 42)
        self.fc2 = nn.Linear(42, 20)
        self.fc3 = nn.Linear(20, 7)


    def forward(self, x):
        x = x.unsqueeze(0)  # Add an extra dimension for the channels
        x = F.relu(self.conv(x))
        x = x.view(-1, 10*3*4)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(0)
  
# Instantiate the replay buffer
class ReplayBuffer:

    def __init__(self, max_frames):
        self.max_frames = max_frames
        self.buffer = []

    def add(self, frame):
        self.buffer.append(frame)
        if len(self.buffer) > self.max_frames:
            del self.buffer[0:(len(self.buffer)-self.max_frames)]

    def sample(self, num_samples):
        # Ensure we don't pick the same frame twice
        # Record the random indices picked from elements in the buffer
        sample_nums = set() 
        while len(sample_nums) < num_samples:
            sample_nums.add(random.randrange(len(self.buffer)))
        experiences = [self.buffer[i] for i in sample_nums]
        return experiences

def calculate_epsilon(step, epsilon_start, epsilon_finish, total_timesteps, exploration_fraction):
    finish_step = total_timesteps * exploration_fraction
    if step > finish_step:
        return epsilon_finish
    epsilon_range = epsilon_start - epsilon_finish
    return epsilon_finish + (((finish_step - step) / finish_step) * epsilon_range)

def QLoss(state_evals, next_state_evals):
  loss = 1e6 * torch.mean((torch.stack(state_evals) - torch.stack(next_state_evals)) ** 2)
  return loss

# HYPERPARAMETERS
seed = 0
buffer_size = 10
learning_rate = 2.5e-2
pretrain_games = 100
train_games = 20
ideal_batch_size = 10
total_timesteps = 1e6
epsilon_start = 0.9
epsilon_finish = 0
exploration_fraction = 0.8

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True if seed > 0 else False

    # Initialise replay memory D to capacity N
    buffer = ReplayBuffer(buffer_size)

    # Initialize action-value function Q and target network
    q_network = DeepQNetworkConnect4().to(device)
    target_network = DeepQNetworkConnect4().to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimiser = torch.optim.Adam(q_network.parameters(), learning_rate)

    for step in range(int(pretrain_games)):
        # Generate games and add experiences
        g = Game()
        # TODO: Add options for more agent pairs
        g.playGame(agent1 = 1, agent2 = 3)
        g.playGame(agent1 = 3, agent2 = 1)
        for experience in g.experiences:
            buffer.add(experience)
        print(step)

    for iter in range(int(total_timesteps)):
        
        for step in range(int(train_games)):
            # Generate games and add experiences
            g = Game()
            # TODO: Add options for more agent pairs
            to_display = iter % 50 == 0
            g.playGame(agent1 = q_network, agent2 = q_network, pick_display = to_display)
            for experience in g.experiences:
                buffer.add(experience)

        batch = buffer.sample(min(len(buffer.buffer), ideal_batch_size))
        # states and next states should be floats (same as the OUTPUT)
        # brackets are required to turn generator into a list
        states = torch.stack([torch.from_numpy(exp.state) for exp in batch]).float()
        next_states = torch.stack([torch.from_numpy(exp.next_state) for exp in batch]).float()
        # get q-network outputs
        state_q_values = [q_network(state) for state in states]
        next_state_q_values = [target_network(next_state) for next_state in next_states]

        # get loss
        loss = QLoss(state_q_values, next_state_q_values)
        # backprop
        with torch.no_grad():
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        
        target_network = q_network

        print(iter, loss)

train()