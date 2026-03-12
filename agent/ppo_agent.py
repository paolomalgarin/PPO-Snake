# Modello PPO
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam
import numpy as np
from tools.beautyful_progress_bar import PBar
import os

class FeedForwardNN(nn.Module):

    def __init__(self, obs_shape, action_dim):
        super().__init__()

        # Get the channels from the obs shape
        C, H, W = obs_shape
        
        self.conv1 = nn.Conv2d(C, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the shape the tensor will be after the convolutions
        # To manage various game dimentions
        flattened_size = self._get_conv_output(obs_shape)
        
        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        
        self.policy_head = nn.Linear(32, action_dim)
        self.value_head = nn.Linear(32, 1)

    def forward(self, obs):
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        
        x = self.conv1(obs)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)
        # x = self.pool(x)
        
        x = self.conv3(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        x = self.relu(x)

        logits = self.policy_head(x)
        value = self.value_head(x)

        return logits, value
    
    def _get_conv_output(self, shape):
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            x = self.conv1(dummy)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            # x = self.pool(x)
            x = self.conv3(x)
            x = self.relu(x)
            return x.view(1, -1).size(1)


# CREDITS:
# https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-2-4-f9d8b8aa938a
# PPO ALGO (pseudo code):
# https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg

class PPOAgent:

    def __init__(self, env):
        # Extract environment information
        self.env = env
        self.obs_shape = env.observation_space.shape
        self.act_dim = env.action_space.n # Now ONLY handles Discrete actions

        self._init_hyperparameters()

        # PPO ALG STEP 1
        # Initialize actor and critic networks
        self.actor = FeedForwardNN(self.obs_shape, self.act_dim)
        self.critic = FeedForwardNN(self.obs_shape, 1)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # (useless for discrete action spaces)
        # # Create the covariance matrix
        # # Note that I chose 0.5 for stdev arbitrarily.
        # self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        # self.cov_mat = torch.diag(self.cov_var)
    
    def learn(self, total_timesteps):
        t_so_far = 0 # Timesteps simulated so far
        batch_n = 0  # Batch number
        pbar = PBar(total_timesteps, preset="training")
        
        # PPO ALG STEP 2
        while(t_so_far < total_timesteps):
            # PPO ALG STEP 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_rews, batch_lens, _ = self.rollout()
            batch_n += 1

            # Calculate how many timesteps we collected this batch
            total_batch_steps = np.sum(batch_lens)
            t_so_far += total_batch_steps
            pbar.update(total_batch_steps)

            # Display batch resoults
            self._print_stats(total_batch_steps, batch_rews, batch_lens, batch_n)

            # Calculate V_{phi, k}
            V, _, _ = self.evaluate(batch_obs, batch_acts)

            # PPO ALG STEP 5 (step 4 is in rollout function)
            # Calculate advantage
            A_k = batch_rtgs - V.detach()

            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                # Epoch code (where we perform multiple updates on the actor and critic networks)

                # Calculate pi_theta(a_t | s_t)
                V, curr_log_probs, entropy = self.evaluate(batch_obs, batch_acts)
                
                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean() - self.entropy_coef * entropy.mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)
                
                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
        
        pbar.close()

        # return the specific number of timesteps the model was trained for (not necessary for the project)
        return t_so_far


    def _init_hyperparameters(self):
        # Default values for hyperparameters
        self.timesteps_per_batch = 12800       # timesteps per batch (a batch is a number of timesteps before updating PPO's policy)
        self.max_timesteps_per_episode = 3200  # timesteps per episode (an episode is a game inside the env)
        self.gamma = 0.99
        self.n_updates_per_iteration = 10      # Number of epoch, used to perform multiple updates on the actor and critic networks
        self.clip = 0.2
        self.lr = 4e-4
        self.entropy_coef = 0.01

    def rollout(self):
        # Data of a batch
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch
        apples = []            # episodic lengths in batch


        # Number of timesteps run so far this batch
        t = 0

        # => BATCH <=
        while t < self.timesteps_per_batch:
            
            # Rewards of this episode
            ep_rewards = []

            obs, _ = self.env.reset()
            done = False

            # => EPISODE <=
            for ep_t in range(self.max_timesteps_per_episode):

                t += 1

                # Collect current observation
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Collect current reward, action, and log prob
                ep_rewards.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if(done):
                    apples.append(info['score'])
                    break
            
            # Collect episode length and rewards
            batch_lens.append(ep_t + 1)  # +1 because timestep starts at 0
            batch_rews.append(ep_rewards)
        
        # Reshape data as tensors before returning
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.long)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float)

        # PPO ALG STEP 4
        batch_rtgs = self.compute_rtgs(batch_rews)

        # Return the data of the batch 
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_rews, batch_lens, apples

    def get_action(self, obs, deterministic=False):
        # N.B: deterministic is a flag to use (set true) when wanting to choose always the best action the model thinks there is (during play or evaluation)

        # Convert to tensor
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

        # Add batch dimention to obs if not already added to perform convolutions without errors
        if obs.dim() == 3:
            obs = obs.unsqueeze(0) 

        # Query the actor network for a mean action
        mean, _ = self.actor.forward(obs)

        # Create a Categorical Distribution
        dist = Categorical(logits=mean)
        

        if deterministic:
            action = torch.argmax(mean, dim=-1)
            log_prob = dist.log_prob(action)
        else:
            # Sample an action from the distribution and get its log prob
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob  
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return action.detach().numpy(), log_prob.squeeze(0).detach()

    def compute_rtgs(self, batch_rewards):
        # Calculate the rewards-to-go (rtg) per episode per batch to return
        batch_rtgs = []

        # Iterate through each episode backwards to maintain the same order in batch_rtgs
        for ep_rews in reversed(batch_rewards):

            discounted_reward = 0  # The discounted reward so far

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
            
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs
    
    def evaluate(self, batch_obs, batch_acts):
        # Asking critic network for a value V for each obs in batch_obs
        V, _ = self.critic.forward(batch_obs)
        V = V.squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network
        mean, _ = self.actor.forward(batch_obs)
        dist = Categorical(logits=mean)
        log_probs = dist.log_prob(batch_acts.squeeze())
        entropy = dist.entropy()

        # Return predicted values V and log probs log_probs
        return V, log_probs, entropy
    
    def save(self, total_timesteps = None, path = os.path.join('results', 'model'), file_name = 'final_mnodel.pth'):
        saving_path = os.path.join(path, file_name)

        # Create folders if not exists
        os.makedirs(path, exist_ok=True)

        # Create checkpoint
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optim_state_dict': self.actor_optim.state_dict(),
            'critic_optim_state_dict': self.critic_optim.state_dict(),
            'timesteps_trained': total_timesteps
        }

        # Save checkpoint
        torch.save(checkpoint, saving_path)
        print(f'Model saved in [ {saving_path} ]')

    def load(self, path, device=None, load_optimizers=True):
        # Fetch the states
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Load actor and critic
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

        # Load optimizers if requested
        if load_optimizers and 'actor_optim_state_dict' in checkpoint:
            self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
        if load_optimizers and 'critic_optim_state_dict' in checkpoint:
            self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])

        # Returns model's timesteps
        ts = checkpoint.get('timesteps_trained', None)
        print('Model loaded successfully!')
        print(f'This model was trained for {f"{ts:,}" if ts != None else "???"} timesteps')
        return ts
    
    def _print_stats(self, batch_steps, batch_rews, batch_lens, batch_n = None):

        # Calculate values to print
        mean_reward = sum(sum(ep_rews) for ep_rews in batch_rews) / len(batch_rews)
        mean_ep_len = sum(batch_lens) / len(batch_lens)
        max_ep_rew = max(sum(ep_rews) for ep_rews in batch_rews)

        strings = [
            f"| ", 
            f"| Mean reward       |  {mean_reward:.2f} ",
            f"| Max episode rew   |  {max_ep_rew} ", 
            f"| Mean episode len  |  {mean_ep_len:.1f} ",
            f"| Total batch steps |  {batch_steps} ",
        ]

        # Calculate max string length
        max_len = max(len(s) for s in strings)

        # Create separator string
        separator = "-" * (max_len + 2)

        # Create 'title' string
        title_separator_len = int((max_len - len(f"|  BATCH{f' {batch_n}' if batch_n != None else ''} ")) / 2)
        title = (
            "| " + 
            " " * title_separator_len +
            f" BATCH{f' {batch_n}' if batch_n != None else ''} " +
            " " * title_separator_len +
            (' ' if (max_len - len(f"|  BATCH{f' {batch_n}' if batch_n != None else ''} ")) % 2 != 0 else '') +
            " |"
        )

        # Print stats
        print(separator)
        
        print(title)
        for s in strings:
            print(s, end=(" " * (max_len - len(s)) + " |\n"))

        print(separator)
        print()