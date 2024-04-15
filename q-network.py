import numpy as np
import game
import heuristics

import random
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

# Instantiate the neural network
class DeepQNetworkConnect4(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(126, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 7),
        )
        print(self.network)

    def forward(self, x):
        return self.network(x)
  
# Instantiate the replay buffer
class ReplayBuffer:

    def __init__(self, max_frames):
        self.max_frames = max_frames
        self.buffer = []

    def add(self, frame):
        self.buffer.append(frame)
        if len(self.buffer) > self.max_frames:
            del self.buffer[0:len(self.buffer)-self.max_frames]

    def sample(self, num_samples):
        # Ensure we don't pick the same frame twice
        # Record the random indices picked from elements in the buffer
        sample_nums = set() 
        while len(sample_nums) < num_samples:
            sample_nums.add(random.randrange(len(self.buffer)))
        experiences = [self.buffer[i] for i in sample_nums]
        return {"obs":      torch.stack([torch.as_tensor(experience["obs"],      dtype=torch.float32) for experience in experiences], 0),
                "next_obs": torch.stack([torch.as_tensor(experience["next_obs"], dtype=torch.float32) for experience in experiences], 0),
                "actions":  torch.stack([torch.as_tensor(experience["action"],   dtype=torch.int64)   for experience in experiences], 0),
                "rewards":  torch.stack([torch.as_tensor(experience["reward"],   dtype=torch.float32) for experience in experiences], 0),
                "dones":    torch.stack([torch.as_tensor(experience["done"],     dtype=torch.float32) for experience in experiences], 0)}
    
def get_rolling_average(items, num):
    rolling = []
    for i in range(len(items)-num):
        rolling.append(sum(items[i:i+num])/num)
    return rolling

def calculate_epsilon(step, epsilon_start, epsilon_finish, total_timesteps, exploration_fraction):
    finish_step = total_timesteps * exploration_fraction
    if step > finish_step:
        return epsilon_finish
    epsilon_range = epsilon_start - epsilon_finish
    return epsilon_finish + (((finish_step - step) / finish_step) * epsilon_range)

'''
def train(env, total_timesteps, batch_size, buffer_size, train_frequency, seed,
          target_network_update_frequency,  gamma, learning_rate, epsilon_start,
          epsilon_finish, exploration_fraction, learn_start_size, noop_size):

    env = gym.make(env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True if seed > 0 else False

    # Initialise replay memory D to capacity N
    buffer = ReplayBuffer(buffer_size)

    # Initialize action-value function Q and target network
    q_network = DeepQNetworkCartpole(env).to(device)
    target_network = DeepQNetworkCartpole(env).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimiser = torch.optim.Adam(q_network.parameters(), learning_rate)

    obs = env.reset()
    episode_end_steps = []
    episode_rewards = []
    last_episode_end_step = 0

    for step in range(total_timesteps):

        # Select random action with p (epsilon), else argmax (q)
        epsilon = calculate_epsilon(step, epsilon_start, epsilon_finish, total_timesteps, exploration_fraction)

        if random.random() < epsilon:
            # Perform a random action
            action = env.action_space.sample()
        else:
            # Perform the best action available
            logits = q_network(torch.Tensor(obs).to(device))
            action = torch.argmax(logits, dim=0).cpu().numpy().tolist()

        # Perform the action, then keep track of everything
        next_obs, reward, done, info = env.step(action) # FIX THIS!
        next_obs_copy = next_obs.copy()

        # Store transition in the buffer D
        buffer.add({"obs": obs, "next_obs": next_obs_copy, "action": [action], 
                    "reward": [reward], "done": [1 if done else 0], "info": info})

        if done: # If episode is done
            print(f"Finished {step} steps, episode returns {step-last_episode_end_step}")
            episode_end_steps.append(step)
            episode_rewards.append(step-last_episode_end_step)
            last_episode_end_step = step
            obs = env.reset()
            continue

        obs = next_obs

        if step > learn_start_size and step % train_frequency == 0:

            # Sample replay experiences
            experiences = buffer.sample(batch_size)

            with torch.no_grad(): # We want to sample the Q-network here, but not update it YET

                # The _ removes the indices we don't want, leaving only the values
                target_max, _ = target_network(experiences["next_obs"]).max(dim=1) 
                # If done, sets second term to 0, since we just want rewards
                td_target = experiences["rewards"].flatten() + gamma * target_max * (1 - experiences["dones"].flatten())

            # Collect tensor of predicted rewards associated with the actions taken
            old_val = q_network(experiences["obs"]).gather(1, experiences["actions"]).squeeze()

            # Calculate loss
            loss = F.mse_loss(td_target, old_val)

            # Gradient descent
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        # Update target network
        if step % target_network_update_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

    env.close()
    plot_results(episode_end_steps, episode_rewards, total_timesteps)
'''