"""
Soft Actor-Critic (SAC) Model for Hockey Environment

Implements:
1. Gaussian Actor (policy network)
2. Twin Critics (Q-networks)
3. Automatic temperature tuning

Fixed issues:
- Proper alpha initialization
- Gradient clipping option
- Better numerical stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class Actor(nn.Module):
    """
    Gaussian policy network (actor)
    
    Outputs mean and log_std for each action dimension
    Actions are sampled from Gaussian distribution and squashed with tanh
    """
    
    def __init__(self, obs_dim, action_dim, hidden_sizes=[256, 256], 
                 log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build network
        layers = []
        prev_size = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.net = nn.Sequential(*layers)
        
        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(prev_size, action_dim)
        self.log_std_layer = nn.Linear(prev_size, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # Special initialization for output layers
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
        
    def forward(self, obs):
        """
        Forward pass
        
        Returns
        -------
        mean : torch.Tensor
            Mean of action distribution
        log_std : torch.Tensor
            Log standard deviation of action distribution
        """
        x = self.net(obs)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, obs, deterministic=False, with_logprob=True):
        """
        Sample action from policy
        
        Parameters
        ----------
        obs : torch.Tensor
            Observation
        deterministic : bool
            If True, return mean (no noise)
        with_logprob : bool
            If True, return log probability
            
        Returns
        -------
        action : torch.Tensor
            Sampled action (squashed to [-1, 1])
        log_prob : torch.Tensor or None
            Log probability of action (None if with_logprob=False)
        """
        mean, log_std = self.forward(obs)
        
        if deterministic:
            # Deterministic action (for evaluation)
            action = torch.tanh(mean)
            return action, None
        
        # Sample from Gaussian
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        
        # Reparameterization trick: x = mean + std * noise
        x = normal.rsample()  # rsample() enables gradient flow
        action = torch.tanh(x)  # Squash to [-1, 1]
        
        if with_logprob:
            # Compute log probability with tanh correction
            log_prob = normal.log_prob(x)
            
            # Correction for tanh squashing (more numerically stable version)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
            return action, log_prob
        else:
            return action, None


class Critic(nn.Module):
    """
    Q-network (critic)
    
    Takes observation and action as input, outputs Q-value
    """
    
    def __init__(self, obs_dim, action_dim, hidden_sizes=[256, 256]):
        super(Critic, self).__init__()
        
        # Build network
        layers = []
        prev_size = obs_dim + action_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output Q-value
        layers.append(nn.Linear(prev_size, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
    def forward(self, obs, action):
        """
        Forward pass
        
        Parameters
        ----------
        obs : torch.Tensor
            Observation
        action : torch.Tensor
            Action
            
        Returns
        -------
        q_value : torch.Tensor
            Q-value for (observation, action) pair
        """
        # Concatenate observation and action
        x = torch.cat([obs, action], dim=-1)
        q_value = self.net(x)
        return q_value


class TwinCritic(nn.Module):
    """
    Twin Q-networks
    
    Two critics to reduce overestimation bias
    """
    
    def __init__(self, obs_dim, action_dim, hidden_sizes=[256, 256]):
        super(TwinCritic, self).__init__()
        
        self.critic1 = Critic(obs_dim, action_dim, hidden_sizes)
        self.critic2 = Critic(obs_dim, action_dim, hidden_sizes)
        
    def forward(self, obs, action):
        """
        Forward pass through both critics
        
        Returns
        -------
        q1 : torch.Tensor
            Q-value from critic 1
        q2 : torch.Tensor
            Q-value from critic 2
        """
        q1 = self.critic1(obs, action)
        q2 = self.critic2(obs, action)
        return q1, q2
    
    def q1(self, obs, action):
        """Get Q-value from critic 1 only"""
        return self.critic1(obs, action)
    
    def q2(self, obs, action):
        """Get Q-value from critic 2 only"""
        return self.critic2(obs, action)


class SAC:
    """
    Soft Actor-Critic Agent
    
    Implements SAC with:
    - Gaussian actor (policy)
    - Twin critics (Q-functions)
    - Automatic temperature tuning
    - Soft target updates
    """
    
    def __init__(self,
                 obs_dim,
                 action_dim,
                 device,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 alpha_lr=3e-4,
                 hidden_sizes=[256, 256],
                 gamma=0.99,
                 tau=0.005,
                 alpha=0.2,
                 auto_entropy_tuning=True,
                 target_entropy=None,
                 grad_clip=None):
        """
        Initialize SAC agent
        
        Parameters
        ----------
        obs_dim : int
            Observation dimension
        action_dim : int
            Action dimension
        device : torch.device
            Device (CPU or GPU)
        actor_lr : float
            Actor learning rate
        critic_lr : float
            Critic learning rate
        alpha_lr : float
            Temperature learning rate
        hidden_sizes : list
            Hidden layer sizes
        gamma : float
            Discount factor
        tau : float
            Soft update coefficient (0.005 = 0.5% update per step)
        alpha : float
            Initial temperature
        auto_entropy_tuning : bool
            Whether to automatically tune temperature
        target_entropy : float
            Target entropy (default: -action_dim)
        grad_clip : float or None
            Gradient clipping value (None = no clipping)
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.grad_clip = grad_clip
        
        # Initialize alpha first (before any property access)
        self._alpha = alpha
        self.auto_entropy_tuning = auto_entropy_tuning
        
        # Create networks
        self.actor = Actor(obs_dim, action_dim, hidden_sizes).to(device)
        self.critic = TwinCritic(obs_dim, action_dim, hidden_sizes).to(device)
        self.critic_target = TwinCritic(obs_dim, action_dim, hidden_sizes).to(device)
        
        # Initialize target networks 
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Freeze target networks 
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Automatic entropy tuning
        if auto_entropy_tuning:
            if target_entropy is None:
                # Default heuristic: -action_dim
                self.target_entropy = -action_dim
            else:
                self.target_entropy = target_entropy
            
            self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, 
                                         requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
    
    @property
    def alpha(self):
        """Get current temperature"""
        if self.auto_entropy_tuning:
            return self.log_alpha.exp().item()
        else:
            return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        """Set temperature (only when not auto-tuning)"""
        self._alpha = value
    
    def select_action(self, obs, deterministic=False):
        """
        Select action
        
        Parameters
        ----------
        obs : np.array
            Observation
        deterministic : bool
            If True, use mean (for evaluation)
            
        Returns
        -------
        action : np.array
            Selected action
        """
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                if obs.ndim == 1:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                else:
                    obs_tensor = torch.FloatTensor(obs).to(self.device)
            else:
                obs_tensor = obs.unsqueeze(0).to(self.device) if obs.dim() == 1 else obs.to(self.device)
            
            action_tensor, _ = self.actor.sample(obs_tensor, deterministic=deterministic, with_logprob=False)
            
            action = action_tensor.cpu().numpy().flatten()
            
        return action
    
    def update(self, replay_buffer, batch_size):
        """
        Update actor, critics, and temperature
        """
        # Sample batch
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = replay_buffer.sample(batch_size)
        
        if isinstance(obs_batch, torch.Tensor):

            obs = obs_batch
            actions = act_batch
            next_obs = next_obs_batch
            rewards = rew_batch
            dones = done_batch
        else:
            obs = torch.FloatTensor(obs_batch).to(self.device)
            actions = torch.FloatTensor(act_batch).to(self.device)
            next_obs = torch.FloatTensor(next_obs_batch).to(self.device)
            rewards = torch.FloatTensor(rew_batch).unsqueeze(1).to(self.device)
            dones = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)
        
        # ================== Update Critics ==================
        
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs = self.actor.sample(next_obs)
            
            # Compute target Q-values using target networks
            q1_next_target, q2_next_target = self.critic_target(next_obs, next_actions)
            q_next_target = torch.min(q1_next_target, q2_next_target)
            
            # Add entropy bonus
            q_next_target = q_next_target - self.alpha * next_log_probs
            
            q_target = rewards + (1 - dones) * self.gamma * q_next_target
        
        # Get current Q-values
        q1, q2 = self.critic(obs, actions)
        
        # Critic losses (MSE)
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)
        critic_loss = critic1_loss + critic2_loss
        
        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()
        
        # ================== Update Actor ==================
        
        # Sample actions from current policy
        new_actions, log_probs = self.actor.sample(obs)
        
        # Compute Q-values for new actions
        q1_new, q2_new = self.critic(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()
        
        # ================== Update Temperature ==================
        
        if self.auto_entropy_tuning:
            # If entropy < target, increase α to encourage exploration
            # If entropy > target, decrease α
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0)
        
        # ================== Soft Update Target Networks ==================
        
        self._soft_update(self.critic, self.critic_target)
        
        # Return losses for logging
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else alpha_loss,
            'alpha': self.alpha,
            'mean_q1': q1.mean().item(),
            'mean_q2': q2.mean().item(),
            'mean_log_prob': log_probs.mean().item(),
        }
    
    def _soft_update(self, source, target):
        """
        Soft update target network
        
        target = tau * source + (1-tau) * target
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, filepath):
        """Save model"""
        save_dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'auto_entropy_tuning': self.auto_entropy_tuning,
        }
        
        if self.auto_entropy_tuning:
            save_dict['log_alpha'] = self.log_alpha
            save_dict['alpha_optimizer'] = self.alpha_optimizer.state_dict()
            save_dict['target_entropy'] = self.target_entropy
        else:
            save_dict['alpha'] = self._alpha
            
        torch.save(save_dict, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        if self.auto_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha.data = checkpoint['log_alpha'].data
            if 'alpha_optimizer' in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        elif not self.auto_entropy_tuning and 'alpha' in checkpoint:
            self._alpha = checkpoint['alpha']