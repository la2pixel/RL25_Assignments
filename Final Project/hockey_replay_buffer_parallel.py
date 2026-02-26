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
        """Single sample add (Compatibility mode)"""
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

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay buffer.
    
    Samples transitions with probability proportional to their TD error.
    
    Parameters
    ----------
    size : int
        Maximum buffer size
    alpha : float
        Priority exponent (0 = uniform, 1 = full prioritization)
    beta : float
        Importance sampling exponent (annealed to 1)
    """
    
    def __init__(self, size, alpha=0.6, beta=0.4):
        super().__init__(size)
        self.alpha = alpha
        self.beta = beta
        self._priorities = np.zeros(size, dtype=np.float32)
        self._max_priority = 1.0
    
    def add(self, obs, action, reward, next_obs, done):
        """Add transition with max priority"""
        # Ensure observations are flattened
        if isinstance(obs, np.ndarray) and obs.ndim == 2 and obs.shape[0] == 1:
            obs = obs.squeeze(0)
        if isinstance(next_obs, np.ndarray) and next_obs.ndim == 2 and next_obs.shape[0] == 1:
            next_obs = next_obs.squeeze(0)
        
        if not isinstance(action, np.ndarray):
            action = np.asarray(action)
        
        # Set max priority for new transitions
        self._priorities[self._next_idx] = self._max_priority
        
        data = (obs, action, reward, next_obs, float(done))
        
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        
        self._next_idx = (self._next_idx + 1) % self._maxsize
    
    def sample(self, batch_size):
        """
        Sample based on priorities.
        
        Returns
        -------
        tuple
            (obs, actions, rewards, next_obs, dones, weights, indices)
            weights are importance sampling weights
        """
        n = len(self._storage)
        
        # Compute sampling probabilities
        priorities = self._priorities[:n] ** self.alpha
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(n, size=batch_size, p=probs)
        
        weights = (n * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  
        
        # Get samples
        obs, actions, rewards, next_obs, dones = self._encode_sample(indices)
        
        return obs, actions, rewards, next_obs, dones, weights.astype(np.float32), indices
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD errors.
        
        Parameters
        ----------
        indices : np.array
            Indices of transitions to update
        td_errors : np.array
            TD errors for each transition
        """
        priorities = np.abs(td_errors) + 1e-6  
        
        for idx, priority in zip(indices, priorities):
            self._priorities[idx] = priority
            self._max_priority = max(self._max_priority, priority)
    
    def anneal_beta(self, current_step, total_steps, final_beta=1.0):
        """Anneal beta towards 1.0"""
        self.beta = self.beta + (final_beta - self.beta) * (current_step / total_steps)