import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, size, device):
        self.max_size = size
        self.ptr = 0
        self.size = 0
        self.device = device
        
        # Pre-allocate huge tensors on GPU
        self.obs      = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.actions  = torch.zeros((size, action_dim), dtype=torch.float32, device=device)
        self.rewards  = torch.zeros((size, 1), dtype=torch.float32, device=device)
        self.dones    = torch.zeros((size, 1), dtype=torch.float32, device=device)

    def add(self, obs, action, reward, next_obs, done):
        """Single sample add"""
        if isinstance(obs, np.ndarray): obs = torch.from_numpy(obs).to(self.device)
        if isinstance(action, np.ndarray): action = torch.from_numpy(action).to(self.device)
        if isinstance(reward, np.ndarray): reward = torch.from_numpy(reward).to(self.device)
        if isinstance(next_obs, np.ndarray): next_obs = torch.from_numpy(next_obs).to(self.device)
        
        d_val = float(done)
        
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = d_val

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, obs, action, reward, next_obs, done):
        """
        Add a BATCH of transitions (from vectorized envs) to the buffer.
        Input shapes: (num_envs, dim)
        """
        if isinstance(obs, np.ndarray): 
            obs = torch.from_numpy(obs).float().to(self.device)
        
        if isinstance(action, np.ndarray): 
            action = torch.from_numpy(action).float().to(self.device)
            
        if isinstance(reward, np.ndarray): 
            reward = torch.from_numpy(reward).float().to(self.device)
            
        if isinstance(next_obs, np.ndarray): 
            next_obs = torch.from_numpy(next_obs).float().to(self.device)
            
        if isinstance(done, np.ndarray): 
            done = torch.from_numpy(done).float().to(self.device)

        batch_num = obs.shape[0]
        
        # Calculate indices (circular buffer logic)
        indices = torch.arange(self.ptr, self.ptr + batch_num, device=self.device) % self.max_size
        
        # Store data
        self.obs[indices] = obs
        self.actions[indices] = action
        self.next_obs[indices] = next_obs
        
        # Ensure correct shapes for reward/done (N, 1)
        if reward.ndim == 1: reward = reward.unsqueeze(1)
        if done.ndim == 1: done = done.unsqueeze(1)
        
        self.rewards[indices] = reward
        self.dones[indices] = done

        # Update pointer and size
        self.ptr = (self.ptr + batch_num) % self.max_size
        self.size = min(self.size + batch_num, self.max_size)

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        return self.size
